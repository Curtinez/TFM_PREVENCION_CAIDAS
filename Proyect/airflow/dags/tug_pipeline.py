from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.sensors.python import PythonSensor
from deltalake import DeltaTable
from minio import Minio
from minio.error import S3Error

# Configuración de MinIO
MINIO_HOST = "minio:9000"
MINIO_USER = "minioadmin"
MINIO_PASSWORD = "minioadmin123"
BUCKET_A_COMPROBAR = "source"

# Configuración de los jobs de Spark
SPARK_JOBS_DIR = "/opt/airflow/spark-jobs"
JOBS_DIR = "/opt/airflow/jobs"
SPARK_PACKAGES = "io.delta:delta-spark_2.12:3.2.1,org.apache.hadoop:hadoop-aws:3.3.4"
SPARK_CONF = {
    "spark.hadoop.fs.s3a.access.key": MINIO_USER,
    "spark.hadoop.fs.s3a.secret.key": MINIO_PASSWORD,
    "spark.hadoop.fs.s3a.endpoint": f"http://{MINIO_HOST}",
    "spark.hadoop.fs.s3a.path.style.access": "true",
    "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
    "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
}


def check_minio_bucket(**context) -> bool:
    try:
        client = Minio(MINIO_HOST, access_key=MINIO_USER, secret_key=MINIO_PASSWORD, secure=False)
        if client.bucket_exists(BUCKET_A_COMPROBAR):
            print(f"Bucket '{BUCKET_A_COMPROBAR}' existe en MinIO")
            return True
        print(f"Bucket '{BUCKET_A_COMPROBAR}' NO existe en MinIO todavía")
        return False
    except S3Error as e:
        print(f"Error de S3: {e}")
        return False
    except Exception as e:
        print(f"Error al conectar con MinIO: {e}")
        return False


DELTALAKE_STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID": MINIO_USER,
    "AWS_SECRET_ACCESS_KEY": MINIO_PASSWORD,
    "AWS_ENDPOINT_URL": f"http://{MINIO_HOST}",
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}


def _prueba_ids_en_source(client: Minio, prefix: str) -> set:
    ids = set()
    for obj in client.list_objects(BUCKET_A_COMPROBAR, prefix=prefix, recursive=False):
        nombre = obj.object_name.rstrip("/")
        if "prueba_id=" in nombre:
            ids.add(nombre.split("prueba_id=")[-1])
    return ids


def hay_pruebas_pendientes(**context) -> bool:
    """
    Comprobación barata (sin arrancar Spark) de si queda algo por procesar:
    compara las pruebas en origen (source/) contra las que ya llegaron a
    gold/imu, el último eslabón del pipeline.
    """
    client = Minio(MINIO_HOST, access_key=MINIO_USER, secret_key=MINIO_PASSWORD, secure=False)
    pruebas_origen = _prueba_ids_en_source(client, "camera/") | _prueba_ids_en_source(client, "imu/")

    try:
        dt = DeltaTable("s3://gold/imu/", storage_options=DELTALAKE_STORAGE_OPTIONS)
        pruebas_terminadas = set(dt.to_pyarrow_table(columns=["prueba_id"]).column("prueba_id").to_pylist())
    except Exception:
        pruebas_terminadas = set()

    pendientes = pruebas_origen - pruebas_terminadas
    if pendientes:
        print(f"Pruebas pendientes de procesar: {sorted(pendientes)}")
        return True

    print("No hay pruebas nuevas: se salta el resto del DAG.")
    return False


with DAG(
    dag_id="tug_pipeline",
    description="Pipeline TUG completo: bronze -> silver -> gold de cámara e IMU, y fusión final",
    start_date=datetime(2025, 1, 1),
    # Sondeo periódico: los jobs son incrementales/idempotentes (cada uno
    # solo procesa las pruebas que aún no tiene), así que ejecutar de más
    # no hace daño — simplemente no encuentra pruebas pendientes y termina.
    schedule="*/15 * * * *",
    catchup=False,
    tags=["tug", "spark", "mediapipe", "minio"],
) as dag:

    # Espera a que exista el bucket 'source' en MinIO antes de arrancar nada
    check_bucket = PythonSensor(
        task_id="check_minio_bucket",
        python_callable=check_minio_bucket,
        timeout=300,
        poke_interval=30,
        mode="poke",
    )

    # Corta el DAG aquí si no hay pruebas nuevas que procesar
    hay_pendientes = ShortCircuitOperator(
        task_id="hay_pruebas_pendientes",
        python_callable=hay_pruebas_pendientes,
    )

    # Ingesta cruda del CSV del IMU (capa Bronze)
    bronze_imu = SparkSubmitOperator(
        task_id="bronze_imu",
        conn_id="spark_default",
        application=f"{SPARK_JOBS_DIR}/bronze_imu.py",
        name="Bronze_IMU",
        packages=SPARK_PACKAGES,
        conf=SPARK_CONF,
        verbose=True,
    )

    # Filtra y limpia el IMU aplicando las reglas de calidad (capa Silver)
    silver_imu = SparkSubmitOperator(
        task_id="silver_imu",
        conn_id="spark_default",
        application=f"{SPARK_JOBS_DIR}/silver_imu.py",
        name="Silver_IMU",
        packages=SPARK_PACKAGES,
        conf=SPARK_CONF,
        verbose=True,
    )

    # Ingesta cruda de los frames de cámara (capa Bronze)
    bronze_camera = SparkSubmitOperator(
        task_id="bronze_camera",
        conn_id="spark_default",
        application=f"{SPARK_JOBS_DIR}/bronze_camara.py",
        name="Bronze_Camera",
        packages=SPARK_PACKAGES,
        conf=SPARK_CONF,
        verbose=True,
    )

    # Detección de pose con MediaPipe (capa Silver). No es un job de Spark
    # (ver jobs/silver_camera_pose.py); corre con el Python de Airflow.
    silver_camera = BashOperator(
        task_id="silver_camera",
        bash_command=f"python {JOBS_DIR}/silver_camera_pose.py",
    )

    # Calcula sentado/sobrepasado por frame a partir de los landmarks (capa Gold)
    gold_camera = SparkSubmitOperator(
        task_id="gold_camera",
        conn_id="spark_default",
        application=f"{SPARK_JOBS_DIR}/gold_camera.py",
        name="Gold_Camera",
        packages=SPARK_PACKAGES,
        conf=SPARK_CONF,
        verbose=True,
    )

    # Job de fusión final: cruza silver_imu con gold_camera y saca el
    # resumen de la prueba TUG completa
    gold_imu = SparkSubmitOperator(
        task_id="gold_imu",
        conn_id="spark_default",
        application=f"{SPARK_JOBS_DIR}/gold_imu.py",
        name="Gold_IMU_Fusion",
        packages=SPARK_PACKAGES,
        conf=SPARK_CONF,
        verbose=True,
    )

    check_bucket >> hay_pendientes >> [bronze_imu, bronze_camera]
    bronze_imu >> silver_imu
    bronze_camera >> silver_camera >> gold_camera
    [silver_imu, gold_camera] >> gold_imu
