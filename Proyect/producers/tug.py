import cv2
import time
import uuid
import threading
import queue
import csv
from io import BytesIO, StringIO
from minio import Minio
from minio.error import S3Error
import serial

# Configuración de la cámara
VIDEO_URL = "http://192.168.0.48:8080/video"
FPS_UPLOAD = 5
JPEG_QUALITY = 85
ROTAR_FRAME = False

# Configuración del bucket
MINIO_HOST = "localhost:9000"
MINIO_USER = "minioadmin"
MINIO_PASSWORD = "minioadmin123"
BUCKET = "source"
minio_client = Minio(
    MINIO_HOST,
    access_key=MINIO_USER,
    secret_key=MINIO_PASSWORD,
    secure=False
)

# Configuración del IMU
SERIAL_PORT = "/dev/rfcomm0"   
SERIAL_BAUDRATE = 115200
IMU_CHUNK_SIZE = 20
IMU_ENABLED = True
# Cabecera de los datos del IMU
IMU_HEADER = ["frame_timestamp_ms", "imu_timestamp_ms",
              "accel_x_g", "accel_y_g", "accel_z_g",
              "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]


# Función para comprobar si el bucket esta creado
def check_bucket(nombre: str) -> None:
    try:
        if not minio_client.bucket_exists(nombre):
            minio_client.make_bucket(nombre)
            print(f"[MinIO] Bucket '{nombre}' creado.")
        else:
            print(f"[MinIO] Bucket '{nombre}' ya existe.")
    except S3Error as e:
        print(f"[MinIO] Error comprobando bucket: {e}")
        raise

# Cola para la subida de datos
upload_queue  = queue.Queue()
frames_subidos = 0
frames_lock = threading.Lock()

# última muestra disponible del IMU
ultima_muestra_imu: dict = {
    "timestamp_ms": None,
    "accel_x_g":    None, "accel_y_g": None, "accel_z_g": None,
    "gyro_x_dps":   None, "gyro_y_dps": None, "gyro_z_dps": None,
}
imu_muestra_lock = threading.Lock()
stop_imu  = threading.Event()


# Subir frames capturados en una cola
def upload_worker() -> None:
    global frames_subidos
    while True:
        item = upload_queue.get()

        # Si recibe un item nulo de la cola, se termina la subida de datos
        if item is None:
            upload_queue.task_done()
            break

        prueba_id, object_name, data_bytes = item
        try:
            buffer = BytesIO(data_bytes)
            # Detectar tipo de contenido por la carpeta de destino
            content_type = "text/csv" if object_name.startswith("imu/") else "image/jpeg"
            minio_client.put_object(
                bucket_name=BUCKET,
                object_name=object_name,
                data=buffer,
                length=len(data_bytes),
                content_type=content_type
            )
            # Solo contar frames de cámara en el contador principal
            if not object_name.startswith("imu/"):
                with frames_lock:
                    frames_subidos += 1
        except S3Error as e:
            print(f"[MinIO] Error subiendo '{object_name}': {e}")
        finally:
            upload_queue.task_done()

# Función para subir los datos del IMU cuando pasan de la logitud del chunck
def _flush_imu_buffer(rows: list, prueba_id: str, chunk_idx: int) -> None:
    if not rows:
        return
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(IMU_HEADER)
    writer.writerows(rows)
    csv_bytes = buf.getvalue().encode("utf-8")
    timestamp_ms = int(time.time() * 1000)
    object_name = (
        f"imu/prueba_id={prueba_id}/"
        f"imu_{timestamp_ms:016d}_chunk{chunk_idx:04d}.csv"
    )
    upload_queue.put((prueba_id, object_name, csv_bytes))


def imu_reader() -> None:
    """
    Hilo que lee el puerto serie continuamente y actualiza 'ultima_muestra_imu'
    con la lectura más reciente. El bucle principal la consulta al capturar cada
    frame, asociando la muestra al mismo timestamp que la imagen.
    """
    try:
        ser = serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=1)
        print(f"[IMU] Conectado a {SERIAL_PORT} @ {SERIAL_BAUDRATE} baudios")
    except Exception as e:
        print(f"[IMU] No se pudo abrir {SERIAL_PORT}: {e}")
        print("[IMU] Continuando sin IMU.")
        return

    try:
        while not stop_imu.is_set():
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 7:
                continue
            try:
                ts = int(parts[0])
                ax,ay,az= float(parts[1]), float(parts[2]), float(parts[3])
                gx, gy, gz = float(parts[4]), float(parts[5]), float(parts[6])
            except ValueError:
                continue
            # Actualizar la última muestra disponible
            with imu_muestra_lock:
                ultima_muestra_imu["timestamp_ms"] = ts
                ultima_muestra_imu["accel_x_g"]    = ax
                ultima_muestra_imu["accel_y_g"]    = ay
                ultima_muestra_imu["accel_z_g"]    = az
                ultima_muestra_imu["gyro_x_dps"]   = gx
                ultima_muestra_imu["gyro_y_dps"]   = gy
                ultima_muestra_imu["gyro_z_dps"]   = gz
    except Exception as e:
        print(f"[IMU] Error en el hilo lector: {e}")
    finally:
        ser.close()
        print("[IMU] Puerto serie cerrado.")


# Función para añadir una interfaz gráfica a la prueba
def dibujar_panel(frame, prueba_activa: bool, prueba_id: str | None,
                  pruebas_totales: int, frames_en_cola: int,
                  imu_muestras: int = 0) -> None:

    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if prueba_activa:
        # Prueba en curso
        cv2.putText(frame, "● PRUEBA EN CURSO",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 80), 2)
        id_corto = prueba_id[:18] + "..." if prueba_id else "?"
        cv2.putText(frame, f"ID: {id_corto}",
                    (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
        with frames_lock:
            subidos = frames_subidos
        cv2.putText(frame,
                    f"Camara — Frames subidos: {subidos}   |   Cola: {frames_en_cola}",
                    (12, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)
        # Línea IMU
        if IMU_ENABLED:
            cv2.putText(frame, f"IMU   — Muestras capturadas: {imu_muestras}",
                        (12, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 200, 255), 1)
        # Tecla de acción en la esquina derecha
        cv2.putText(frame, "[F] Finalizar",
                    (w - 210, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 100, 255), 2)
    else:
        if pruebas_totales == 0:
            msg   = "Sin prueba activa"
            color = (0, 200, 255)
        else:
            msg   = f"Prueba #{pruebas_totales} finalizada"
            color = (0, 200, 255)
            with frames_lock:
                subidos = frames_subidos
            cv2.putText(frame, f"Frames guardados en esta prueba: {subidos}",
                        (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (160, 160, 160), 1)
            if frames_en_cola > 0:
                cv2.putText(frame, f"Subiendo cola: {frames_en_cola} frames restantes...",
                            (12, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 160, 255), 1)

        cv2.putText(frame, f"◌ {msg}",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(frame, "[I] Iniciar nueva prueba",
                    (w - 370, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 220, 80), 2)

    # Tecla para salir
    cv2.putText(frame, "[ESC] Salir",
                (w - 165, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)

# Código primcipal
def main() -> None:
    global frames_subidos

    # Comprobar si existe el bucket
    check_bucket(BUCKET)

    # Hilo de subida de datos
    worker = threading.Thread(target=upload_worker, daemon=True, name="upload-worker")
    worker.start()

    # Hilo de lectura de datos del IMU
    imu_worker = None
    if IMU_ENABLED:
        imu_worker = threading.Thread(target=imu_reader, daemon=True, name="imu-reader")
        imu_worker.start()

    # Estado de la prueba
    prueba_activa = False
    prueba_id = None
    pruebas_totales = 0
    ultimo_upload = 0.0
    intervalo_upload = 1.0 / FPS_UPLOAD

    # Buffer de filas IMU (una fila por frame capturado)
    imu_buffer = []
    imu_chunk_idx = 0
    imu_muestras_count = 0

    # Capturar video de la cámara
    cap = cv2.VideoCapture(VIDEO_URL)

    # Comprobar si hay video
    if not cap.isOpened():
        print(f"[Error] No se puede abrir el stream: {VIDEO_URL}")
        upload_queue.put(None)
        stop_imu.set()
        worker.join()
        if imu_worker:
            imu_worker.join()
        return

    # Nombrar a la ventana
    cv2.namedWindow("TFM – Captura", cv2.WINDOW_NORMAL)

    # Bucle principal
    while True:
        # Obtener el frame de la cámara.
        ret, frame = cap.read()
        if not ret:
            print("[Aviso] Sin frame de la cámara")
            time.sleep(0.1)
            continue

        # Rotar el frame si es necesario
        if ROTAR_FRAME:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Tiempo actual
        ahora = time.time()

        # Captura y poner en la cola la imagen
        # Subir a la cola las imágenes que hayan pasado el intervalo
        if prueba_activa and (ahora - ultimo_upload) >= intervalo_upload:
            ok, img_encoded = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if ok:
                timestamp_ms = int(ahora * 1000)

                # Añadir a la cola los datos de la cámara
                object_name = (
                    f"camera/prueba_id={prueba_id}/"
                    f"frame_{timestamp_ms:016d}.jpg"
                )
                upload_queue.put((prueba_id, object_name, img_encoded.tobytes()))

                
                # Añadir a la cola los datos de la cámara
                if IMU_ENABLED:
                    with imu_muestra_lock:
                        muestra = dict(ultima_muestra_imu)
                    if muestra["timestamp_ms"] is not None:
                        imu_buffer.append([
                            timestamp_ms, muestra["timestamp_ms"],
                            muestra["accel_x_g"], muestra["accel_y_g"], muestra["accel_z_g"],
                            muestra["gyro_x_dps"], muestra["gyro_y_dps"],muestra["gyro_z_dps"],
                        ])
                        imu_muestras_count += 1
                        if len(imu_buffer) >= IMU_CHUNK_SIZE:
                            _flush_imu_buffer(imu_buffer, prueba_id, imu_chunk_idx)
                            imu_chunk_idx += 1
                            imu_buffer = []

                ultimo_upload = ahora

        # Interfaz de la prueba
        dibujar_panel(
            frame,
            prueba_activa = prueba_activa,
            prueba_id = prueba_id,
            pruebas_totales = pruebas_totales,
            frames_en_cola = upload_queue.qsize(),
            imu_muestras = imu_muestras_count,
        )

        cv2.imshow("TUG – Captura", frame)

        # Capturar teclado
        key = cv2.waitKey(1) & 0xFF

        # Presionar Escape → vaciar buffer IMU y salir
        if key == 27:
            if imu_buffer and prueba_id:
                _flush_imu_buffer(imu_buffer, prueba_id, imu_chunk_idx)
                imu_buffer = []
            break

        # Presionar I o i → iniciar una nueva prueba
        elif key in (ord('i'), ord('I')):
            if not prueba_activa:
                prueba_id = str(uuid.uuid4())
                prueba_activa = True
                pruebas_totales += 1
                ultimo_upload = 0.0
                imu_buffer = []
                imu_chunk_idx = 0
                imu_muestras_count = 0
                with frames_lock:
                    frames_subidos = 0
                print(f"[INFO] Prueba #{pruebas_totales} iniciada.")
                print(f"       ID     : {prueba_id}")
                print(f"       Camara : {BUCKET}/camera/prueba_id={prueba_id}/")
                print(f"       IMU    : {BUCKET}/imu/prueba_id={prueba_id}/")

        # Presionar F → terminar la prueba y vaciar buffer IMU restante
        elif key in (ord('f'), ord('F')):
            if prueba_activa:
                prueba_activa = False
                if imu_buffer and prueba_id:
                    _flush_imu_buffer(imu_buffer, prueba_id, imu_chunk_idx)
                    imu_chunk_idx += 1
                    imu_buffer = []
                with frames_lock:
                    subidos = frames_subidos
                print(f"[INFO] Prueba #{pruebas_totales} finalizada.")
                print(f"       Frames cámara  : {subidos}")
                print(f"       Muestras IMU   : {imu_muestras_count}")
                print(f"       Cola pendiente : {upload_queue.qsize()}")

    # Cerrar y subir frames restantes
    cap.release()
    cv2.destroyAllWindows()

    stop_imu.set()

    pendientes = upload_queue.qsize()
    if pendientes > 0:
        print(f"\n[INFO] Esperando a subir {pendientes} objetos restantes...")

    upload_queue.put(None)
    worker.join()
    if imu_worker:
        imu_worker.join()

    with frames_lock:
        print(f"[INFO] Subida completada. Frames de cámara: {frames_subidos}")
    print("[INFO] Programa terminado.")


if __name__ == "__main__":
    main()
