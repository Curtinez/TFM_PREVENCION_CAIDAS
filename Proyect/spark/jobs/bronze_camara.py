from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract
from delta.tables import DeltaTable

# Rutas de origen y destino
SOURCE_PATH = "s3a://source/camera/"
BRONZE_PATH = "s3a://bronze/camera/"

def crear_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Bronze_Camera")
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
    
    # Seleccionamos solo la columna prueba_id de manera única
    return (
        spark.read.format("delta").load(BRONZE_PATH)
        .select("prueba_id")
        .distinct()
    )

# Leer las imágenes desde la fuente extrayendo los metadatos dinámicamente
def leer_source(spark):
    df_raw = (
        spark.read
        .format("binaryFile")
        .option("basePath", SOURCE_PATH)
        .load(SOURCE_PATH)
    )
    
    df_procesado = df_raw.select(
        # Extraer el id de la prueba buscando "prueba_id=VALOR"
        regexp_extract(col("path"), r"prueba_id=([^/]+)", 1).alias("prueba_id"),
        
        # Extraer el timestamp numérico del título de la imagen
        regexp_extract(col("path"), r"frame_(\d+)\.jpg", 1).cast("long").alias("frame_timestamp_ms"),
        
        # Guardar los bytes de la imagen
        col("content"),
    )
    
    return df_procesado

# Código principal
def main():
    spark = crear_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("[Bronze Camara] Iniciando pipeline de ingesta...")

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
            print("[Bronze Camara] Sin pruebas nuevas para ingestar. Pipeline completado sin cambios.")
            return

        print(f"[Bronze Camara] Imágenes nuevas pertenecientes a nuevas pruebas a ingestar: {count_new}")

        # Escribir las imágenes en la capa bronze usando Delta y particionando por prueba_id
        df_new.write.format("delta").mode("append").partitionBy("prueba_id").save(BRONZE_PATH)
        
        print(f"[Bronze Camara] {count_new} imágenes escritas en {BRONZE_PATH}")
        print("[Bronze Camara] Pipeline completado correctamente.")

    except Exception as e:
        print(f"[Bronze Camara] Error en el pipeline: {e}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()