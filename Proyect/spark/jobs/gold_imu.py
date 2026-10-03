import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, BooleanType
from delta.tables import DeltaTable

# Rutas de origen y destino
GOLD_CAMERA_PATH = "s3a://gold/camera/"
SILVER_IMU_PATH = "s3a://silver/imu/"
GOLD_PATH = "s3a://gold/imu/"

# Ventana mínima (segundos) para considerar un tramo de sentado/sobrepasado
MIN_DWELL_S = 1.0

# Un pico de magnitud de aceleración por encima de este valor (g) se
# cuenta como paso.
UMBRAL_PASO_G = 1.1

# Separación mínima (s) entre dos pasos aceptados
DISTANCIA_MIN_PASO_S = 0.3

# Corte clínico estándar del TUG: >=12s se considera riesgo alto de caída / fragilidad
FRAGIL_UMBRAL_S = 12.0

EVENTOS_SCHEMA = StructType([
    StructField("prueba_id", StringType()),
    StructField("finalizada", BooleanType()),
    StructField("fragil", BooleanType()),
    StructField("intentos_levantarse", IntegerType()),
    StructField("duracion_prueba_s", DoubleType()),
    StructField("tiempo_levantarse_s", DoubleType()),
    StructField("tiempo_hasta_marca_s", DoubleType()),
    StructField("tiempo_giro_y_vuelta_marca_s", DoubleType()),
    StructField("tiempo_vuelta_silla_s", DoubleType()),
    # Columnas internas (instantes absolutos) para acotar las ventanas del
    # IMU en la siguiente etapa; no se escriben en la tabla final.
    StructField("t0_abs_ms", DoubleType()),
    StructField("t1_abs_ms", DoubleType()),
    StructField("t4_abs_ms", DoubleType()),
])

IMU_METRICS_SCHEMA = StructType([
    StructField("prueba_id", StringType()),
    StructField("pico_levantarse_dps", DoubleType()),
    StructField("n_pasos_detectados", IntegerType()),
    StructField("cv_intervalo_pasos", DoubleType()),
])


def crear_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Gold_IMU")
        .master("spark://spark-master:7077")
        .config("spark.submit.deployMode", "client")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.hadoop.fs.s3a.access.key",             "minioadmin")
        .config("spark.hadoop.fs.s3a.secret.key",             "minioadmin123")
        .config("spark.hadoop.fs.s3a.endpoint",               "http://minio:9000")
        .config("spark.hadoop.fs.s3a.path.style.access",      "true")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.impl",                   "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .getOrCreate()
    )


# Obtiene la lista de 'prueba_id' únicos que ya existen en Gold
def obtener_pruebas_existentes(spark):
    if not DeltaTable.isDeltaTable(spark, GOLD_PATH):
        return None

    return (
        spark.read.format("delta").load(GOLD_PATH)
        .select("prueba_id")
        .distinct()
    )


# Suaviza una señal booleana ruidosa: para cada instante, mira la media de
# los valores en una ventana de MIN_DWELL_S centrada en él (voto por
# mayoría temporal). Un pico/hueco más corto que la ventana no la cambia.
def suavizar_booleano(valores: pd.Series, tiempos_s: np.ndarray, ventana_s: float) -> np.ndarray:
    indice = pd.to_datetime(tiempos_s, unit="s")
    serie = pd.Series(valores.astype(float).to_numpy(), index=indice)
    suavizada = serie.rolling(f"{int(ventana_s * 1000)}ms", center=True, min_periods=1).mean()
    return (suavizada.to_numpy() > 0.5)


# Divide una señal booleana en tramos consecutivos: lista de (valor, t_inicio, t_fin)
def tramos(valores_bool: np.ndarray, tiempos_s: np.ndarray) -> list:
    resultado = []
    inicio = 0
    for i in range(1, len(valores_bool) + 1):
        if i == len(valores_bool) or valores_bool[i] != valores_bool[inicio]:
            resultado.append((bool(valores_bool[inicio]), float(tiempos_s[inicio]), float(tiempos_s[i - 1])))
            inicio = i
    return resultado


# Detecta picos locales (mayor que ambos vecinos) por encima de un umbral,
# descartando los que caen a menos de distancia_min del último aceptado
def detectar_picos(valores: np.ndarray, tiempos: np.ndarray, umbral: float, distancia_min: float) -> np.ndarray:
    idx_candidatos = [
        i for i in range(1, len(valores) - 1)
        if valores[i] > valores[i - 1] and valores[i] > valores[i + 1] and valores[i] > umbral
    ]

    picos = []
    ultimo_t = None
    for i in idx_candidatos:
        t = tiempos[i]
        if ultimo_t is not None and (t - ultimo_t) < distancia_min:
            continue
        picos.append(t)
        ultimo_t = t

    return np.array(picos)


# --- Etapa 1: eventos de la prueba a partir de Gold Camera ----------------
def calcular_eventos_prueba(pdf: pd.DataFrame) -> pd.DataFrame:
    prueba_id = pdf["prueba_id"].iloc[0]
    fila = {
        "prueba_id": prueba_id,
        "finalizada": False,
        "fragil": None,
        "intentos_levantarse": 0,
        "duracion_prueba_s": None,
        "tiempo_levantarse_s": None,
        "tiempo_hasta_marca_s": None,
        "tiempo_giro_y_vuelta_marca_s": None,
        "tiempo_vuelta_silla_s": None,
        "t0_abs_ms": None,
        "t1_abs_ms": None,
        "t4_abs_ms": None,
    }

    pdf = pdf.dropna(subset=["sentado", "sobrepasado"]).sort_values("frame_timestamp_ms").reset_index(drop=True)
    if pdf.empty:
        return pd.DataFrame([fila])

    t_abs = pdf["frame_timestamp_ms"].to_numpy(dtype=float)
    t0_abs = t_abs[0]
    fila["t0_abs_ms"] = float(t0_abs)
    t_s = (t_abs - t0_abs) / 1000.0

    sentado = suavizar_booleano(pdf["sentado"], t_s, MIN_DWELL_S)
    sobrepasado = suavizar_booleano(pdf["sobrepasado"], t_s, MIN_DWELL_S)

    tramos_sentado = tramos(sentado, t_s)

    # Intentos de levantarse: nº de transiciones sentado(True) -> de pie(False)
    intentos = sum(
        1 for i in range(1, len(tramos_sentado))
        if tramos_sentado[i - 1][0] and not tramos_sentado[i][0]
    )
    fila["intentos_levantarse"] = intentos

    if intentos == 0:
        return pd.DataFrame([fila])

    # El último tramo de pie es el intento final: el que, si la prueba se
    # completa, es el que recorre los 3 metros.
    t1_s = next(t_ini for valor, t_ini, _ in reversed(tramos_sentado) if not valor)
    fila["tiempo_levantarse_s"] = t1_s
    fila["t1_abs_ms"] = float(t0_abs + t1_s * 1000.0)

    # t2: primer instante sobrepasado tras quedarse de pie
    candidatos = np.where((t_s >= t1_s) & sobrepasado)[0]
    if len(candidatos) == 0:
        return pd.DataFrame([fila])
    t2_s = float(t_s[candidatos[0]])

    # t3: primer instante, tras t2, en que deja de estar sobrepasado (ya de vuelta)
    candidatos = np.where((t_s > t2_s) & ~sobrepasado)[0]
    if len(candidatos) == 0:
        return pd.DataFrame([fila])
    t3_s = float(t_s[candidatos[0]])

    # t4: primer instante, tras t3, en que vuelve a estar sentado
    candidatos = np.where((t_s > t3_s) & sentado)[0]
    if len(candidatos) == 0:
        return pd.DataFrame([fila])
    t4_s = float(t_s[candidatos[0]])

    fila["finalizada"] = True
    fila["fragil"] = t4_s >= FRAGIL_UMBRAL_S
    fila["duracion_prueba_s"] = t4_s
    fila["tiempo_hasta_marca_s"] = t2_s - t1_s
    fila["tiempo_giro_y_vuelta_marca_s"] = t3_s - t2_s
    fila["tiempo_vuelta_silla_s"] = t4_s - t3_s
    fila["t4_abs_ms"] = float(t0_abs + t4_s * 1000.0)

    return pd.DataFrame([fila])


# --- Etapa 2: métricas del IMU, acotadas por los instantes de la etapa 1 --
def calcular_imu_prueba(pdf: pd.DataFrame) -> pd.DataFrame:
    prueba_id = pdf["prueba_id"].iloc[0]
    t0_abs = pdf["t0_abs_ms"].iloc[0]
    t1_abs = pdf["t1_abs_ms"].iloc[0]
    t4_abs = pdf["t4_abs_ms"].iloc[0]

    fila = {
        "prueba_id": prueba_id,
        "pico_levantarse_dps": None,
        "n_pasos_detectados": 0,
        "cv_intervalo_pasos": None,
    }

    if pd.isna(t1_abs):
        # Nunca llegó a quedarse de pie: no hay ventanas que analizar
        return pd.DataFrame([fila])

    pdf = pdf.sort_values("host_timestamp_ms").reset_index(drop=True)
    t_abs_imu = pdf["host_timestamp_ms"].to_numpy(dtype=float)
    t_s_imu = (t_abs_imu - t0_abs) / 1000.0

    gyro_mag = np.sqrt(pdf["gyro_x_dps"] ** 2 + pdf["gyro_y_dps"] ** 2 + pdf["gyro_z_dps"] ** 2).to_numpy()
    accel_mag = np.sqrt(pdf["accel_x_g"] ** 2 + pdf["accel_y_g"] ** 2 + pdf["accel_z_g"] ** 2).to_numpy()

    ventana_levantarse = (t_abs_imu >= t0_abs) & (t_abs_imu <= t1_abs)
    if ventana_levantarse.any():
        fila["pico_levantarse_dps"] = float(gyro_mag[ventana_levantarse].max())

    # Ventana de "de pie": de levantarse a sentarse (o hasta el final de la
    # grabación, si la prueba no llegó a finalizar)
    if pd.notna(t4_abs):
        ventana_de_pie = (t_abs_imu >= t1_abs) & (t_abs_imu <= t4_abs)
    else:
        ventana_de_pie = t_abs_imu >= t1_abs

    tiempos_pico = detectar_picos(
        accel_mag[ventana_de_pie], t_s_imu[ventana_de_pie], UMBRAL_PASO_G, DISTANCIA_MIN_PASO_S
    )
    intervalos = np.diff(tiempos_pico)
    fila["n_pasos_detectados"] = int(len(tiempos_pico))
    if len(intervalos) >= 2:
        fila["cv_intervalo_pasos"] = float(np.std(intervalos) / np.mean(intervalos))

    return pd.DataFrame([fila])


def main():
    spark = crear_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("[Gold IMU] Iniciando pipeline de fusión...")

        if not DeltaTable.isDeltaTable(spark, GOLD_CAMERA_PATH):
            print("[Gold IMU] La tabla Gold de cámara no existe aún. Finalizando.")
            return
        if not DeltaTable.isDeltaTable(spark, SILVER_IMU_PATH):
            print("[Gold IMU] La tabla Silver de IMU no existe aún. Finalizando.")
            return

        df_camera = spark.read.format("delta").load(GOLD_CAMERA_PATH)
        # Silver ya descarta las filas que no pasan las reglas de calidad
        # (ver silver_imu.py), así que aquí no hace falta volver a filtrar.
        df_imu = spark.read.format("delta").load(SILVER_IMU_PATH)

        df_pruebas_existentes = obtener_pruebas_existentes(spark)
        if df_pruebas_existentes is not None:
            df_camera_pendiente = df_camera.join(df_pruebas_existentes, on="prueba_id", how="left_anti")
        else:
            df_camera_pendiente = df_camera

        pruebas_pendientes = [row["prueba_id"] for row in df_camera_pendiente.select("prueba_id").distinct().collect()]
        if not pruebas_pendientes:
            print("[Gold IMU] Sin pruebas pendientes. Pipeline completado sin cambios.")
            return

        print(f"[Gold IMU] Pruebas pendientes: {pruebas_pendientes}")

        df_eventos = df_camera_pendiente.groupBy("prueba_id").applyInPandas(calcular_eventos_prueba, schema=EVENTOS_SCHEMA)
        df_eventos.cache()

        df_imu_con_eventos = df_imu.join(
            df_eventos.select("prueba_id", "t0_abs_ms", "t1_abs_ms", "t4_abs_ms"),
            on="prueba_id",
            how="inner",
        )
        df_imu_metrics = df_imu_con_eventos.groupBy("prueba_id").applyInPandas(calcular_imu_prueba, schema=IMU_METRICS_SCHEMA)

        df_final = (
            df_eventos
            .drop("t0_abs_ms", "t1_abs_ms", "t4_abs_ms")
            .join(df_imu_metrics, on="prueba_id", how="left")
        )

        (
            df_final.write
            .format("delta")
            .mode("append")
            .save(GOLD_PATH)
        )

        print(f"[Gold IMU] Ingesta en {GOLD_PATH} completada con éxito.")

    except Exception as e:
        print(f"[Gold IMU] Error en el pipeline: {e}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
