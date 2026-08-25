from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from delta.tables import DeltaTable

# Rutas de origen y destino
BRONZE_PATH = "s3a://bronze/imu/"
SILVER_PATH = "s3a://silver/imu/"

# Rango físico del MPU6886 (M5StickC Plus)
ACCEL_RANGE_G = 8.0
GYRO_RANGE_DPS = 2000.0

ACCEL_COLS = ["accel_x_g", "accel_y_g", "accel_z_g"]
GYRO_COLS = ["gyro_x_dps", "gyro_y_dps", "gyro_z_dps"]


def crear_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Silver_IMU")
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


# Obtiene la lista de 'prueba_id' únicos que ya existen en Silver
def obtener_pruebas_existentes(spark):
    if not DeltaTable.isDeltaTable(spark, SILVER_PATH):
        return None

    return (
        spark.read.format("delta").load(SILVER_PATH)
        .select("prueba_id")
        .distinct()
    )


# Marca las filas que incumplen alguna regla de calidad; se descartan en
# main(). ±8g/±2000°/s es el límite físico del sensor (MPU6886): superarlo
# es una lectura corrupta, no una señal real más extrema.
def aplicar_reglas_calidad(df):
    # Entre duplicados de (prueba_id, host_timestamp_ms) nos quedamos con
    # uno, desempatando por imu_timestamp_ms (reloj propio del M5).
    w_duplicados = Window.partitionBy("prueba_id", "host_timestamp_ms").orderBy("imu_timestamp_ms")

    valor_nulo = F.lit(False)
    for c in ACCEL_COLS + GYRO_COLS:
        valor_nulo = valor_nulo | F.col(c).isNull()

    accel_fuera_rango = F.lit(False)
    for c in ACCEL_COLS:
        accel_fuera_rango = accel_fuera_rango | (F.abs(F.col(c)) > ACCEL_RANGE_G)

    gyro_fuera_rango = F.lit(False)
    for c in GYRO_COLS:
        gyro_fuera_rango = gyro_fuera_rango | (F.abs(F.col(c)) > GYRO_RANGE_DPS)

    timestamp_duplicado = F.row_number().over(w_duplicados) > 1

    return (
        df
        .withColumn("motivos_descarte", F.array_compact(F.array(
            F.when(valor_nulo, F.lit("valor_nulo")),
            F.when(accel_fuera_rango, F.lit("accel_fuera_rango")),
            F.when(gyro_fuera_rango, F.lit("gyro_fuera_rango")),
            F.when(timestamp_duplicado, F.lit("timestamp_duplicado")),
        )))
        .withColumn("calidad_ok", F.size("motivos_descarte") == 0)
    )


# Código principal
def main():
    spark = crear_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("[Silver IMU] Iniciando pipeline...")

        if not DeltaTable.isDeltaTable(spark, BRONZE_PATH):
            print("[Silver IMU] La tabla Bronze no existe aún. Finalizando.")
            return

        df_bronze = spark.read.format("delta").load(BRONZE_PATH)

        # Filtrar para mantener solo las pruebas que aún no están en Silver
        df_pruebas_existentes = obtener_pruebas_existentes(spark)
        if df_pruebas_existentes is not None:
            df_pendiente = df_bronze.join(
                df_pruebas_existentes,
                on="prueba_id",
                how="left_anti"
            )
        else:
            df_pendiente = df_bronze

        count_pendientes = df_pendiente.count()

        if count_pendientes == 0:
            print("[Silver IMU] Sin filas pendientes. Pipeline completado sin cambios.")
            return

        print(f"[Silver IMU] Filas pendientes: {count_pendientes}")

        df_marcado = aplicar_reglas_calidad(df_pendiente)

        n_descartadas = df_marcado.filter(~F.col("calidad_ok")).count()
        print(f"[Silver IMU] Filas descartadas por calidad: {n_descartadas} / {count_pendientes}")

        df_silver = (
            df_marcado
            .filter(F.col("calidad_ok"))
            .drop("calidad_ok", "motivos_descarte")
        )

        (
            df_silver.write
            .format("delta")
            .mode("append")
            .partitionBy("prueba_id")
            .save(SILVER_PATH)
        )

        print(f"[Silver IMU] Ingesta en {SILVER_PATH} completada con éxito.")

    except Exception as e:
        print(f"[Silver IMU] Error en el pipeline: {e}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
