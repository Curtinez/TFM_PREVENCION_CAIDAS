import os
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# Configuración
STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
    "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", "http://minio:9000"),
    "AWS_REGION": os.environ.get("AWS_REGION", "us-east-1"),
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

BRONZE_PATH = "s3://bronze/camera/"
SILVER_PATH = "s3://silver/camera/"

MODEL_PATH = Path(os.environ.get(
    "MODELO_PATH", str(Path(__file__).resolve().parent / "models" / "pose_landmarker_lite.task")
))
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

SILVER_SCHEMA = pa.schema([
    ("prueba_id", pa.string()),
    ("frame_timestamp_ms", pa.int64()),
    ("landmarks", pa.list_(pa.struct([
        ("landmark_id", pa.int64()),
        ("x", pa.float32()),
        ("y", pa.float32()),
        ("z", pa.float32()),
        ("visibility", pa.float32()),
        ("presence", pa.float32()),
    ]))),
])


# Modelo MediaPipe
def asegurar_modelo() -> None:
    if MODEL_PATH.exists():
        return
    print(f"[Silver Camara Pose] Descargando modelo MediaPipe en {MODEL_PATH}...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def crear_landmarker() -> vision.PoseLandmarker:
    opciones = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.PoseLandmarker.create_from_options(opciones)


def procesar_frame(landmarker: vision.PoseLandmarker, img_bytes: bytes) -> list:
    nparr = np.frombuffer(bytes(img_bytes), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    resultado = landmarker.detect(mp_image)

    if not resultado.pose_landmarks:
        return []

    return [
        {
            "landmark_id": idx,
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility) if lm.visibility is not None else 0.0,
            "presence": float(lm.presence) if lm.presence is not None else 0.0,
        }
        for idx, lm in enumerate(resultado.pose_landmarks[0])
    ]


# Delta Lake (sin Spark)
def pruebas_procesadas() -> set:
    try:
        dt = DeltaTable(SILVER_PATH, storage_options=STORAGE_OPTIONS)
    except Exception:
        return set()
    return set(dt.to_pyarrow_table(columns=["prueba_id"]).column("prueba_id").to_pylist())


def leer_bronze_pendiente(procesadas: set) -> pd.DataFrame:
    try:
        dt = DeltaTable(BRONZE_PATH, storage_options=STORAGE_OPTIONS)
    except Exception:
        print("[Silver Camara Pose] La tabla Bronze no existe aún. Finalizando.")
        return pd.DataFrame(columns=["prueba_id", "frame_timestamp_ms", "content"])

    filtro = [("prueba_id", "not in", list(procesadas))] if procesadas else None
    tabla = dt.to_pyarrow_table(
        columns=["prueba_id", "frame_timestamp_ms", "content"], partitions=filtro
    )
    return tabla.to_pandas()


def guardar_silver(filas: list) -> None:
    tabla_salida = pa.Table.from_pylist(filas, schema=SILVER_SCHEMA)
    write_deltalake(
        SILVER_PATH,
        tabla_salida,
        mode="append",
        partition_by=["prueba_id"],
        storage_options=STORAGE_OPTIONS,
    )


# Código principal
def main() -> None:
    print("[Silver Camara Pose] Iniciando pipeline...")
    asegurar_modelo()

    procesadas = pruebas_procesadas()
    print(f"[Silver Camara Pose] Pruebas ya en Silver: {sorted(procesadas)}")

    df_bronze = leer_bronze_pendiente(procesadas)
    if df_bronze.empty:
        print("[Silver Camara Pose] Sin frames pendientes. Pipeline completado.")
        return

    print(f"[Silver Camara Pose] Frames pendientes: {len(df_bronze)}")

    landmarker = crear_landmarker()
    try:
        for prueba_id, grupo in df_bronze.groupby("prueba_id"):
            print(f"[Silver Camara Pose] Procesando prueba '{prueba_id}' ({len(grupo)} frames)...")
            filas = []
            for _, fila in grupo.iterrows():
                try:
                    landmarks = procesar_frame(landmarker, fila["content"])
                except Exception as err:
                    print(f"[Silver Camara Pose] WARN frame {fila['frame_timestamp_ms']}: {err}", file=sys.stderr)
                    landmarks = []

                filas.append({
                    "prueba_id": str(prueba_id),
                    "frame_timestamp_ms": int(fila["frame_timestamp_ms"]),
                    "landmarks": landmarks,
                })

            guardar_silver(filas)
            print(f"[Silver Camara Pose] Prueba '{prueba_id}' escrita en Silver ({len(filas)} frames).")
    finally:
        landmarker.close()

    print("[Silver Camara Pose] Pipeline completado con éxito.")


if __name__ == "__main__":
    main()
