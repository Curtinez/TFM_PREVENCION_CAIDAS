from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, LongType, DoubleType
from delta.tables import DeltaTable

# RUtas de origen y destino
SOURCE_PATH = "s3a://source/imu/"
BRONZE_PATH = "s3a://bronze/imu/"

# Schema de los datos del IMU
IMU_SCHEMA = StructType([
    StructField("host_timestamp_ms", DoubleType(), nullable=True),
    StructField("imu_timestamp_ms", LongType(), nullable=True),
    StructField("accel_x_g", DoubleType(), nullable=True),
    StructField("accel_y_g", DoubleType(), nullable=True),
    StructField("accel_z_g", DoubleType(), nullable=True),
    StructField("gyro_x_dps", DoubleType(), nullable=True),
    StructField("gyro_y_dps", DoubleType(), nullable=True),
    StructField("gyro_z_dps", DoubleType(), nullable=True),
])


def crear_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Bronze_IMU")
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

# Obtiene la lista de 'prueba_id' únicos que ya existen en Bronze
def obtener_pruebas_existentes(spark):
    if not DeltaTable.isDeltaTable(spark, BRONZE_PATH):
        return None

    return (
        spark.read.format("delta").load(BRONZE_PATH)
        .select("prueba_id")
        .distinct()
    )

# Leer los datos de la fuente
def leer_source(spark):
    return (
        spark.read
        .schema(IMU_SCHEMA)
        .option("header", "true")
        .option("basePath", SOURCE_PATH)
        .csv(SOURCE_PATH)
    )

# Código principal
def main():
    spark = crear_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("[Bronze IMU] Iniciando pipeline de ingesta...")

        # Leer los datos de la fuente
        df_source = leer_source(spark)

        # Obtener los IDs ya existentes en la capa Bronze
        df_pruebas_existentes = obtener_pruebas_existentes(spark)

        # Filtrar para mantener solo las pruebas que NO están en Bronze
        if df_pruebas_existentes is not None:
            # left_anti mantiene filas de df_source cuyo prueba_id NO existe en df_pruebas_existentes
            df_new = df_source.join(
                df_pruebas_existentes,
                on="prueba_id",
                how="left_anti"
            )
        else:
            df_new = df_source

        count_new = df_new.count()

        if count_new == 0:
            print("[Bronze IMU] Sin datos nuevos. Pipeline completado sin cambios.")
            return

        print(f"[Bronze IMU] Filas nuevas a ingestar: {count_new}")

        # Escribir los datos en la capa bronze
        df_new.write.format("delta").mode("append").partitionBy("prueba_id").save(BRONZE_PATH)
        

        print(f"[Bronze IMU] {count_new} filas escritas en {BRONZE_PATH}")
        print("[Bronze IMU] Pipeline completado correctamente.")

    except Exception as e:
        print(f"[Bronze IMU] Error en el pipeline: {e}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()