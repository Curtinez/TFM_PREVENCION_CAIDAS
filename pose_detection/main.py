import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
import time
import numpy as np
import math

# Lista de resultados
result_list = []


# Función para calcular el ángulo
def calcular_angulo(hombro, cadera, rodilla):

    hombro = np.array(hombro)
    cadera = np.array(cadera)
    rodilla = np.array(rodilla)

    angulo = math.degrees(math.atan2(rodilla[1] - cadera[1], rodilla[0] - cadera[0])
                          - math.atan2(hombro[1] - cadera[1], hombro[0] - cadera[0]))

    if angulo < 0:
        angulo += 360

    return angulo


# Función callback para procesar los resultados
def res_callback(result, output_image, timestamp_ms):
    result_list.append(result)


# Configuración del modelo
options = vision.PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_lite.task"),
    running_mode=vision.RunningMode.LIVE_STREAM,
    result_callback=res_callback
)
landmarker = vision.PoseLandmarker.create_from_options(options)

# Procesar el video
# Cámara del ordenador
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Video del teléfono
video_url = "http://192.168.0.20:8080/video"
cap = cv2.VideoCapture(video_url)

# Bucle para procesar frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rotamos el video si grabamos verticalmente
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    # Altura y ancho
    h, w, _ = frame.shape

    # Cambiamos el formato de la imagen
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    landmarker.detect_async(frame_rgb, time.time_ns() // 1_000_000)

    # Obtenemos el resultado del modelo
    if result_list:
        for lm in result_list[0].pose_landmarks:
            # lado izquierdo del cuerpo
            x_hombro = int(lm[11].x * w)
            y_hombro = int(lm[11].y * h)
            x_cadera = int(lm[23].x * w)
            y_cadera = int(lm[23].y * h)
            x_rodilla = int(lm[25].x * w)
            y_rodilla = int(lm[25].y * h)
            p_hombro = [x_hombro, y_hombro]
            p_cadera = [x_cadera, y_cadera]
            p_rodilla = [x_rodilla, y_rodilla]
            angulo_sentado_iz = calcular_angulo(p_hombro, p_cadera, p_rodilla)

            # lado_derecho del cuerpo
            x_hombro = int(lm[12].x * w)
            y_hombro = int(lm[12].y * h)
            x_cadera = int(lm[24].x * w)
            y_cadera = int(lm[24].y * h)
            x_rodilla = int(lm[26].x * w)
            y_rodilla = int(lm[26].y * h)
            p_hombro = [x_hombro, y_hombro]
            p_cadera = [x_cadera, y_cadera]
            p_rodilla = [x_rodilla, y_rodilla]
            angulo_sentado_der = calcular_angulo(p_hombro, p_cadera, p_rodilla)

            # Seleccionamos el ángulo más pequeño
            angulo_sentado = np.min([angulo_sentado_der, angulo_sentado_iz])

            # Si el ánulo está por debajo de 140 grados consideramos que está sentado
            if angulo_sentado < 120:
                 estado = "Sentado"
                 color_texto = (0, 255, 255)
            else:
                estado = "De pie"
                color_texto = (0, 255, 0)

            # Mostramos el estado y el ángulo en la imagen
            cv2.putText(frame, f"Estado: {estado}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)
            cv2.putText(frame, f"Angulo: {int(angulo_sentado)}", (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color_texto, 2)

            # Dibujamos todos los puntos en la imagen
            for each_lm in lm:
                if each_lm.visibility > 0.8:
                    x_lm = int(each_lm.x * w)
                    y_lm = int(each_lm.y * h)
                    cv2.circle(frame, (x_lm, y_lm), 8, (0, 0, 255), -1)

        # Borramos el resultado del modelo en este frame
        result_list.clear()

    # Creamos la ventana de video en OpenCV
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Liberamos la ventana
cap.release()
cv2.destroyAllWindows()
