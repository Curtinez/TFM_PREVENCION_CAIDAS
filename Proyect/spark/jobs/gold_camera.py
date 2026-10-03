from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from delta.tables import DeltaTable

# Rutas de origen y destino
SILVER_PATH = "s3a://silver/camera/"
GOLD_PATH = "s3a://gold/camera/"

# Posición del objeto que marca los 3 metros (normalizada, 0-1)
MARCA_X = 0.818
MARCA_Y = 0.735

# Dirección (dx, dy) de la recta de corte que pasa por el objeto
DIR_MARCHA_X = 1.0
DIR_MARCHA_Y = -2.10

# Bajo este ángulo de rodilla (grados) consideramos que está sentado
SENTADO_ANGULO_MAX = 120.0

# Visibilidad mínima de un landmark para confiar en él
VISIBILIDAD_MIN = 0.5

# landmark_id de MediaPipe Pose
LEFT_HIP, RIGHT_HIP = 23, 24
LEFT_KNEE, RIGHT_KNEE = 25, 26
LEFT_ANKLE, RIGHT_ANKLE = 27, 28


def crear_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("Gold_Camera")
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


# Añade columnas <prefijo>_x, <prefijo>_y, <prefijo>_vis extraídas del
# array de landmarks para un landmark_id concreto (sin UDF: filter + element_at)
def con_landmark(df, landmark_id: int, prefijo: str):
    punto = F.element_at(
        F.filter("landmarks", lambda l: l["landmark_id"] == F.lit(landmark_id)), 1
    )
    return (
        df
        .withColumn(f"{prefijo}_x", punto["x"])
        .withColumn(f"{prefijo}_y", punto["y"])
        .withColumn(f"{prefijo}_vis", punto["visibility"])
    )


# Ángulo cadera-rodilla-tobillo (grados) para un lado ("izq"/"der").
# Null si algún punto no es suficientemente visible.
def con_angulo_rodilla(df, lado: str):
    hip_x, hip_y = F.col(f"{lado}_cadera_x"), F.col(f"{lado}_cadera_y")
    knee_x, knee_y = F.col(f"{lado}_rodilla_x"), F.col(f"{lado}_rodilla_y")
    ank_x, ank_y = F.col(f"{lado}_tobillo_x"), F.col(f"{lado}_tobillo_y")

    v1x, v1y = hip_x - knee_x, hip_y - knee_y
    v2x, v2y = ank_x - knee_x, ank_y - knee_y

    producto_punto = v1x * v2x + v1y * v2y
    mag1 = F.sqrt(v1x ** 2 + v1y ** 2)
    mag2 = F.sqrt(v2x ** 2 + v2y ** 2)
    cos_angulo = F.greatest(F.lit(-1.0), F.least(F.lit(1.0), producto_punto / (mag1 * mag2)))

    vis_ok = (
        (F.col(f"{lado}_cadera_vis") > VISIBILIDAD_MIN) &
        (F.col(f"{lado}_rodilla_vis") > VISIBILIDAD_MIN) &
        (F.col(f"{lado}_tobillo_vis") > VISIBILIDAD_MIN)
    )

    return df.withColumn(
        f"angulo_rodilla_{lado}",
        F.when(vis_ok, F.degrees(F.acos(cos_angulo)))
    )


def calcular_caracteristicas(df):
    df = con_landmark(df, LEFT_HIP, "izq_cadera")
    df = con_landmark(df, RIGHT_HIP, "der_cadera")
    df = con_landmark(df, LEFT_KNEE, "izq_rodilla")
    df = con_landmark(df, RIGHT_KNEE, "der_rodilla")
    df = con_landmark(df, LEFT_ANKLE, "izq_tobillo")
    df = con_landmark(df, RIGHT_ANKLE, "der_tobillo")

    df = con_angulo_rodilla(df, "izq")
    df = con_angulo_rodilla(df, "der")

    # Promedio de los ángulos disponibles (uno o los dos lados)
    angulos = F.filter(F.array("angulo_rodilla_izq", "angulo_rodilla_der"), lambda a: a.isNotNull())
    df = df.withColumn(
        "angulo_rodilla",
        F.when(F.size(angulos) > 0, F.aggregate(angulos, F.lit(0.0), lambda acc, x: acc + x) / F.size(angulos))
    )
    df = df.withColumn(
        "sentado",
        F.when(F.col("angulo_rodilla").isNotNull(), F.col("angulo_rodilla") < SENTADO_ANGULO_MAX)
    )

    # Pie más avanzado (mayor x) entre los tobillos suficientemente visibles
    izq_ok = F.col("izq_tobillo_vis") > VISIBILIDAD_MIN
    der_ok = F.col("der_tobillo_vis") > VISIBILIDAD_MIN
    izq_mas_avanzado = F.col("izq_tobillo_x") >= F.col("der_tobillo_x")

    pie_x = (
        F.when(izq_ok & der_ok, F.when(izq_mas_avanzado, F.col("izq_tobillo_x")).otherwise(F.col("der_tobillo_x")))
        .when(izq_ok, F.col("izq_tobillo_x"))
        .when(der_ok, F.col("der_tobillo_x"))
    )
    pie_y = (
        F.when(izq_ok & der_ok, F.when(izq_mas_avanzado, F.col("izq_tobillo_y")).otherwise(F.col("der_tobillo_y")))
        .when(izq_ok, F.col("izq_tobillo_y"))
        .when(der_ok, F.col("der_tobillo_y"))
    )

    # Avance a lo largo de la dirección de marcha, medido desde el objeto.
    avance = (pie_x - F.lit(MARCA_X)) * F.lit(DIR_MARCHA_X) + (pie_y - F.lit(MARCA_Y)) * F.lit(DIR_MARCHA_Y)

    df = df.withColumn("sobrepasado", F.when(pie_x.isNotNull(), avance > 0))

    return df.select(
        "prueba_id",
        "frame_timestamp_ms",
        "sentado",
        "sobrepasado",
    )


def main():
    spark = crear_spark_session()
    spark.sparkContext.setLogLevel("WARN")

    try:
        print("[Gold Camara] Iniciando pipeline...")

        if not DeltaTable.isDeltaTable(spark, SILVER_PATH):
            print("[Gold Camara] La tabla Silver no existe aún. Finalizando.")
            return

        df_silver = spark.read.format("delta").load(SILVER_PATH)

        # Filtrar para mantener solo las pruebas que aún no están en Gold
        df_pruebas_existentes = obtener_pruebas_existentes(spark)
        if df_pruebas_existentes is not None:
            df_pendiente = df_silver.join(
                df_pruebas_existentes,
                on="prueba_id",
                how="left_anti"
            )
        else:
            df_pendiente = df_silver

        count_pendientes = df_pendiente.count()

        if count_pendientes == 0:
            print("[Gold Camara] Sin frames pendientes. Pipeline completado sin cambios.")
            return

        print(f"[Gold Camara] Frames pendientes: {count_pendientes}")

        df_gold = calcular_caracteristicas(df_pendiente)

        (
            df_gold.write
            .format("delta")
            .mode("append")
            .partitionBy("prueba_id")
            .save(GOLD_PATH)
        )

        print(f"[Gold Camara] Ingesta en {GOLD_PATH} completada con éxito.")

    except Exception as e:
        print(f"[Gold Camara] Error en el pipeline: {e}")
        raise

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
