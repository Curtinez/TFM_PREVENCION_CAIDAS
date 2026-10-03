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
VIDEO_URL = "http://192.168.0.17:8080/video"
FPS_UPLOAD = 5
JPEG_QUALITY = 85
ROTAR_FRAME = False

# Configuración de la ventana de vídeo
WINDOW_NAME = "TFM - Captura TUG"
ESCALA_VENTANA_INICIAL = 0.6   # fracción del tamaño nativo del vídeo
ESCALA_VENTANA_PASO = 0.1
ESCALA_VENTANA_MIN = 0.3
ESCALA_VENTANA_MAX = 1.5

# Configuración del bucket de MinIO
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
IMU_CHUNK_SIZE = 500  # ~5s de datos por chunk a ~100Hz, para no crear ficheros diminutos
IMU_ENABLED = True
# host_timestamp_ms: reloj de este PC al recibir la muestra.
# imu_timestamp_ms: millis() del propio M5 desde que arrancó.
IMU_HEADER = ["host_timestamp_ms", "imu_timestamp_ms",
              "accel_x_g", "accel_y_g", "accel_z_g",
              "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]


# Comprueba que el bucket de destino existe en MinIO, y lo crea si no
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

# Cola para la subida de datos (cámara e IMU) a MinIO
upload_queue = queue.Queue()
frames_subidos = 0
frames_lock = threading.Lock()

# Estado de la prueba visible para el hilo del IMU
estado_imu_lock = threading.Lock()
estado_imu = {"activa": False, "prueba_id": None}

# Buffer de muestras del IMU pendientes de subir. Lo rellena imu_reader()
# a su propio ritmo, desacoplado de la cámara.
imu_buffer_lock = threading.Lock()
imu_buffer: list = []
imu_chunk_idx = 0
imu_muestras_count = 0

stop_imu = threading.Event()


# Hilo que va subiendo a MinIO lo que llega por upload_queue
def upload_worker() -> None:
    global frames_subidos
    while True:
        item = upload_queue.get()

        # Un item nulo es la señal de parada
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

# Empaqueta el buffer de muestras del IMU en un CSV y lo manda a la cola de subida
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
    Lee el puerto serie continuamente y añade cada muestra al buffer de
    subida al ritmo real del M5 (~100Hz), desacoplado de la cámara.
    """
    global imu_chunk_idx, imu_muestras_count

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
                ax, ay, az = float(parts[1]), float(parts[2]), float(parts[3])
                gx, gy, gz = float(parts[4]), float(parts[5]), float(parts[6])
            except ValueError:
                continue

            with estado_imu_lock:
                activa, prueba_id = estado_imu["activa"], estado_imu["prueba_id"]
            if not activa:
                continue  # sin prueba en curso, se descarta la muestra

            host_timestamp_ms = time.time() * 1000
            with imu_buffer_lock:
                imu_buffer.append([host_timestamp_ms, ts, ax, ay, az, gx, gy, gz])
                imu_muestras_count += 1
                if len(imu_buffer) >= IMU_CHUNK_SIZE:
                    _flush_imu_buffer(imu_buffer, prueba_id, imu_chunk_idx)
                    imu_chunk_idx += 1
                    imu_buffer.clear()
    except Exception as e:
        print(f"[IMU] Error en el hilo lector: {e}")
    finally:
        ser.close()
        print("[IMU] Puerto serie cerrado.")


# Dibuja el panel de estado (prueba en curso, contadores, teclas) sobre el frame
def dibujar_panel(
    frame,
    prueba_activa: bool,
    prueba_id: str | None,
    pruebas_totales: int,
    frames_en_cola: int,
    imu_muestras: int = 0,
    tiempo_transcurrido: float = 0.0,
) -> None:

    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 150), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    if prueba_activa:
        cv2.putText(frame, f"● PRUEBA EN CURSO — {tiempo_transcurrido:.1f}s",
                    (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 220, 80), 2)
        id_corto = prueba_id[:18] + "..." if prueba_id else "?"
        cv2.putText(frame, f"ID: {id_corto}",
                    (12, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1)
        with frames_lock:
            subidos = frames_subidos
        cv2.putText(frame,
                    f"Camara — Frames subidos: {subidos}   |   Cola: {frames_en_cola}",
                    (12, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)
        if IMU_ENABLED:
            cv2.putText(frame, f"IMU   — Muestras capturadas: {imu_muestras}",
                        (12, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (100, 200, 255), 1)
        cv2.putText(frame, "[F] Finalizar",
                    (w - 210, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 100, 255), 2)
    else:
        if pruebas_totales == 0:
            msg = "Sin prueba activa"
            color = (0, 200, 255)
        else:
            msg = f"Prueba #{pruebas_totales} finalizada"
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

    cv2.putText(frame, "[+/-] Tamano ventana",
                (12, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)
    cv2.putText(frame, "[ESC] Salir",
                (w - 165, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)


# Marca la prueba como inactiva para el hilo del IMU y sube lo que quede en el buffer
def _finalizar_prueba_imu(prueba_id: str) -> None:
    global imu_chunk_idx

    with estado_imu_lock:
        estado_imu["activa"] = False

    with imu_buffer_lock:
        if imu_buffer:
            _flush_imu_buffer(imu_buffer, prueba_id, imu_chunk_idx)
            imu_chunk_idx += 1
            imu_buffer.clear()


def main() -> None:
    global frames_subidos, imu_chunk_idx, imu_muestras_count

    check_bucket(BUCKET)

    worker = threading.Thread(target=upload_worker, daemon=True, name="upload-worker")
    worker.start()

    imu_worker = None
    if IMU_ENABLED:
        imu_worker = threading.Thread(target=imu_reader, daemon=True, name="imu-reader")
        imu_worker.start()

    # Estado de la prueba
    prueba_activa = False
    prueba_id = None
    pruebas_totales = 0
    tiempo_inicio_prueba = 0.0
    ultimo_upload = 0.0
    intervalo_upload = 1.0 / FPS_UPLOAD

    # Estado de la ventana de vídeo
    ventana_configurada = False
    escala_ventana = ESCALA_VENTANA_INICIAL

    cap = cv2.VideoCapture(VIDEO_URL)
    if not cap.isOpened():
        print(f"[Error] No se puede abrir el stream: {VIDEO_URL}")
        upload_queue.put(None)
        stop_imu.set()
        worker.join()
        if imu_worker:
            imu_worker.join()
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Aviso] Sin frame de la cámara")
            time.sleep(0.1)
            continue

        if ROTAR_FRAME:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        frame_h, frame_w = frame.shape[:2]

        # Tamaño inicial de la ventana, calculado en cuanto se conoce la
        # resolución real del vídeo (no se puede saber antes del primer frame)
        if not ventana_configurada:
            cv2.resizeWindow(WINDOW_NAME, int(frame_w * escala_ventana), int(frame_h * escala_ventana))
            ventana_configurada = True

        ahora = time.time()

        # Subir a la cola un frame de cámara cuando toque según FPS_UPLOAD
        # (el IMU se sube aparte, en su propio hilo)
        if prueba_activa and (ahora - ultimo_upload) >= intervalo_upload:
            ok, img_encoded = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            )
            if ok:
                timestamp_ms = int(ahora * 1000)
                object_name = (
                    f"camera/prueba_id={prueba_id}/"
                    f"frame_{timestamp_ms:016d}.jpg"
                )
                upload_queue.put((prueba_id, object_name, img_encoded.tobytes()))
                ultimo_upload = ahora

        with imu_buffer_lock:
            muestras_imu_actual = imu_muestras_count
        dibujar_panel(
            frame,
            prueba_activa=prueba_activa,
            prueba_id=prueba_id,
            pruebas_totales=pruebas_totales,
            frames_en_cola=upload_queue.qsize(),
            imu_muestras=muestras_imu_actual,
            tiempo_transcurrido=(ahora - tiempo_inicio_prueba) if prueba_activa else 0.0,
        )

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF

        # ESC: vaciar buffer IMU y salir
        if key == 27:
            if prueba_activa and prueba_id:
                _finalizar_prueba_imu(prueba_id)
            break

        # I: iniciar una nueva prueba
        elif key in (ord('i'), ord('I')):
            if not prueba_activa:
                prueba_id = str(uuid.uuid4())
                prueba_activa = True
                pruebas_totales += 1
                tiempo_inicio_prueba = ahora
                ultimo_upload = 0.0
                with frames_lock:
                    frames_subidos = 0
                with imu_buffer_lock:
                    imu_buffer.clear()
                    imu_chunk_idx = 0
                    imu_muestras_count = 0
                with estado_imu_lock:
                    estado_imu["activa"] = True
                    estado_imu["prueba_id"] = prueba_id
                print(f"[INFO] Prueba #{pruebas_totales} iniciada.")
                print(f"       ID     : {prueba_id}")
                print(f"       Camara : {BUCKET}/camera/prueba_id={prueba_id}/")
                print(f"       IMU    : {BUCKET}/imu/prueba_id={prueba_id}/")

        # F: terminar la prueba y vaciar buffer IMU restante
        elif key in (ord('f'), ord('F')):
            if prueba_activa:
                prueba_activa = False
                _finalizar_prueba_imu(prueba_id)
                with frames_lock:
                    subidos = frames_subidos
                with imu_buffer_lock:
                    total_imu = imu_muestras_count
                print(f"[INFO] Prueba #{pruebas_totales} finalizada.")
                print(f"       Frames cámara  : {subidos}")
                print(f"       Muestras IMU   : {total_imu}")
                print(f"       Cola pendiente : {upload_queue.qsize()}")

        # +/-: agrandar o encoger la ventana de vídeo
        elif key in (ord('+'), ord('=')):
            escala_ventana = min(ESCALA_VENTANA_MAX, escala_ventana + ESCALA_VENTANA_PASO)
            cv2.resizeWindow(WINDOW_NAME, int(frame_w * escala_ventana), int(frame_h * escala_ventana))
        elif key == ord('-'):
            escala_ventana = max(ESCALA_VENTANA_MIN, escala_ventana - ESCALA_VENTANA_PASO)
            cv2.resizeWindow(WINDOW_NAME, int(frame_w * escala_ventana), int(frame_h * escala_ventana))

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
