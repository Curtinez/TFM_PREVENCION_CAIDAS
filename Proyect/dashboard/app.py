"""
app.py — Dashboard de resultados del pipeline TUG.

Lee directamente las tablas Delta (sin Spark, igual que jobs/silver_camera_pose.py
y tests/pose_detection/verificar_gold_camara.py) y las muestra en Streamlit:
un resumen de todas las pruebas y el detalle de una en concreto.

Arranque local:
    pip install -r dashboard/requirements.txt
    streamlit run dashboard/app.py

Por defecto apunta a MinIO en localhost:9000 (pensado para correr fuera de
Docker, como los demás scripts de verificación). Si se lanza dentro de la
red de docker-compose, sobreescribe AWS_ENDPOINT_URL=http://minio:9000.
"""

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from deltalake import DeltaTable

# ── Paleta (dataviz skill: references/palette.md, ya validada) ──────
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

FASE_COLORES = {
    "Levantarse": BLUE,
    "Hasta la marca": ORANGE,
    "Giro y vuelta a la marca": AQUA,
    "Vuelta a la silla": YELLOW,
}

# ── Configuración ────────────────────────────────────────────────────
STORAGE_OPTIONS = {
    "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", "minioadmin"),
    "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", "minioadmin123"),
    "AWS_ENDPOINT_URL": os.environ.get("AWS_ENDPOINT_URL", "http://localhost:9000"),
    "AWS_REGION": "us-east-1",
    "AWS_ALLOW_HTTP": "true",
    "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
}

FRAGIL_UMBRAL_S = 12.0

st.set_page_config(page_title="TUG - Dashboard", page_icon="🚶", layout="wide")


# ── Carga de datos ───────────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner="Cargando resultados...")
def cargar_tabla(path: str, columnas: list | None = None) -> pd.DataFrame:
    try:
        dt = DeltaTable(path, storage_options=STORAGE_OPTIONS)
    except Exception:
        return pd.DataFrame()
    tabla = dt.to_pyarrow_table(columns=columnas) if columnas else dt.to_pyarrow_table()
    return tabla.to_pandas()


def cargar_gold_imu() -> pd.DataFrame:
    df = cargar_tabla("s3://gold/imu/")
    if df.empty:
        return df
    df = df.sort_values("duracion_prueba_s", ascending=False, na_position="last").reset_index(drop=True)
    df["prueba_corta"] = df["prueba_id"].str[:8]
    return df


def cargar_gold_camera(prueba_id: str) -> pd.DataFrame:
    df = cargar_tabla("s3://gold/camera/")
    if df.empty:
        return df
    df = df[df["prueba_id"] == prueba_id].sort_values("frame_timestamp_ms").reset_index(drop=True)
    if df.empty:
        return df
    df["t_s"] = (df["frame_timestamp_ms"] - df["frame_timestamp_ms"].iloc[0]) / 1000.0
    # sentado/sobrepasado pueden ser None en algún frame suelto (mediapipe sin
    # detección ese frame): el frame hereda el estado del anterior en vez de
    # romper la gráfica.
    df["sentado"] = df["sentado"].ffill().bfill()
    df["sobrepasado"] = df["sobrepasado"].ffill().bfill()
    return df


def cargar_silver_imu(prueba_id: str, t0_abs_ms: float | None = None) -> pd.DataFrame:
    df = cargar_tabla("s3://silver/imu/")
    if df.empty:
        return df
    df = df[(df["prueba_id"] == prueba_id)].sort_values("host_timestamp_ms").reset_index(drop=True)
    if df.empty:
        return df
    # Mismo origen de tiempo que la cámara (t0 = primer frame de Gold Camera),
    # para que las marcas de fase (calculadas sobre ese origen en gold_imu.py)
    # caigan en el instante correcto también en el eje del IMU.
    origen = t0_abs_ms if t0_abs_ms is not None else df["host_timestamp_ms"].iloc[0]
    df["t_s"] = (df["host_timestamp_ms"] - origen) / 1000.0
    df["accel_mag"] = np.sqrt(df["accel_x_g"] ** 2 + df["accel_y_g"] ** 2 + df["accel_z_g"] ** 2)
    df["gyro_mag"] = np.sqrt(df["gyro_x_dps"] ** 2 + df["gyro_y_dps"] ** 2 + df["gyro_z_dps"] ** 2)
    return df


# Instantes (mismo origen que t_s) en los que termina cada fase, a partir de
# las duraciones de Gold IMU. Se corta en la primera fase sin dato (prueba
# no finalizada) en vez de mostrar marcas a medias.
def calcular_marcas_fase(fila: pd.Series) -> list[dict]:
    etiquetas = {
        "Levantarse": "Fin levantarse",
        "Hasta la marca": "Marca (ida)",
        "Giro y vuelta a la marca": "Marca (vuelta)",
        "Vuelta a la silla": "Fin prueba",
    }
    marcas = []
    t_acum = 0.0
    for fase, duracion in fases.items():
        if pd.isna(duracion):
            break
        t_acum += duracion
        marcas.append({"t": t_acum, "fase": etiquetas[fase], "color": FASE_COLORES[fase]})
    return marcas


def anadir_marcas_fase(fig: go.Figure, marcas: list[dict], df: pd.DataFrame, columna_y: str) -> None:
    for marca in marcas:
        y_valor = np.interp(marca["t"], df["t_s"], df[columna_y])
        fig.add_vline(
            x=marca["t"], line=dict(color=marca["color"], width=1, dash="dot"),
            annotation=dict(text=marca["fase"], font=dict(size=10, color=marca["color"]), yshift=8),
            annotation_position="top",
        )
        fig.add_scatter(
            x=[marca["t"]], y=[y_valor], mode="markers", showlegend=False,
            marker=dict(color=marca["color"], size=11, line=dict(color=SURFACE, width=1.5)),
            hovertemplate=f"{marca['fase']}<br>t=%{{x:.1f}}s<extra></extra>",
        )


# Mismos valores que spark/jobs/gold_imu.py: hay que mantenerlos iguales
# para que el número de pasos coincida con la tarjeta de métricas.
UMBRAL_PASO_G = 1.1
DISTANCIA_MIN_PASO_S = 0.3


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



# ── Estilo común de las figuras ──────────────────────────────────────
def figura_base() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font=dict(color=TEXT_PRIMARY, family="system-ui, -apple-system, 'Segoe UI', sans-serif"),
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, color=TEXT_MUTED),
        yaxis=dict(gridcolor=GRIDLINE, zerolinecolor=GRIDLINE, color=TEXT_MUTED),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


# ── Tarjetas de métricas (más vistosas que st.metric) ────────────────
def render_metricas(items: list[dict], vertical: bool = False) -> None:
    """items: lista de {label, value, color} — color es el acento izquierdo.
    Si vertical=True, las tarjetas se apilan una debajo de otra en vez de en fila."""
    st.markdown(
        """
        <style>
        .tug-metric-card {
            background: %(surface)s;
            border: 1px solid %(gridline)s;
            border-left: 4px solid var(--tug-accent, %(blue)s);
            border-radius: 10px;
            padding: 14px 18px;
            height: 100%%;
        }
        .tug-metric-label {
            font-size: 11px;
            font-weight: 600;
            letter-spacing: .06em;
            text-transform: uppercase;
            color: %(muted)s;
            margin-bottom: 8px;
        }
        .tug-metric-value {
            font-size: 26px;
            font-weight: 700;
            color: %(primary)s;
            line-height: 1.1;
        }
        </style>
        """
        % {
            "surface": SURFACE,
            "gridline": GRIDLINE,
            "blue": BLUE,
            "muted": TEXT_MUTED,
            "primary": TEXT_PRIMARY,
        },
        unsafe_allow_html=True,
    )
    cols = st.columns(len(items)) if not vertical else [st.container() for _ in items]
    for col, item in zip(cols, items):
        with col:
            st.markdown(
                f"""
                <div class="tug-metric-card" style="--tug-accent: {item.get('color', BLUE)}; margin-bottom: 12px;">
                    <div class="tug-metric-label">{item['label']}</div>
                    <div class="tug-metric-value">{item['value']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Página ────────────────────────────────────────────────────────────
col_1, col_2 = st.columns([0.75, 0.25])
with col_1:
    st.title("Resultados del TUG")


df_imu = cargar_gold_imu()

if df_imu.empty:
    st.warning(
        "Todavía no hay ninguna prueba en `gold/imu/`. Ejecuta el pipeline "
        "(o el DAG de Airflow) y vuelve a refrescar."
    )
    st.stop()


etiquetas = {
    row.prueba_id: (
        f"{row.prueba_corta} — "
        f"{row.duracion_prueba_s:.1f}s" if row.finalizada else f"{row.prueba_corta} — sin finalizar"
    ) + (" ⚠️" if row.fragil else "")
    for row in df_imu.itertuples()
}
with col_2:

    prueba_id = st.selectbox(
        "Selecciona una Prueba", options=df_imu["prueba_id"], format_func=lambda pid: etiquetas[pid]
    )
fila = df_imu[df_imu["prueba_id"] == prueba_id].iloc[0]

if not fila["finalizada"]:
    st.warning("Esta prueba no se completó.")

if fila["fragil"] is None:
    fragil_texto, fragil_color = "—", TEXT_MUTED
elif fila["fragil"]:
    fragil_texto, fragil_color = "Sí", STATUS_CRITICAL
else:
    fragil_texto, fragil_color = "No", STATUS_GOOD

metricas = [
    {
        "label": "Duración total",
        "value": f"{fila['duracion_prueba_s']:.1f}s" if pd.notna(fila["duracion_prueba_s"]) else "—",
        "color": BLUE,
    },
    {"label": "Frágil", "value": fragil_texto, "color": fragil_color},
    {
        "label": "Intentos para levantarse",
        "value": int(fila["intentos_levantarse"]),
        "color": ORANGE,
    },
    {
        "label": "Pasos detectados",
        "value": int(fila["n_pasos_detectados"]) if pd.notna(fila["n_pasos_detectados"]) else "—",
        "color": AQUA,
    },
    {
        "label": "Pico al levantarse",
        "value": f"{fila['pico_levantarse_dps']:.0f}°/s" if pd.notna(fila["pico_levantarse_dps"]) else "—",
        "color": YELLOW,
    },
]


# Datos necesarios para las gráficas y las tablas de la prueba seleccionada
fases = {
    "Levantarse": fila["tiempo_levantarse_s"],
    "Hasta la marca": fila["tiempo_hasta_marca_s"],
    "Giro y vuelta a la marca": fila["tiempo_giro_y_vuelta_marca_s"],
    "Vuelta a la silla": fila["tiempo_vuelta_silla_s"],
}
fases_completas = all(pd.notna(v) for v in fases.values())
df_cam = cargar_gold_camera(prueba_id)
t0_abs_ms = df_cam["frame_timestamp_ms"].iloc[0] if not df_cam.empty else None
df_s_imu = cargar_silver_imu(prueba_id, t0_abs_ms=t0_abs_ms)
marcas_fase = calcular_marcas_fase(fila)

ALTURA_GRAFICA = 400

st.write("")
st.divider()

col_3, _, col_4, _, col_5 = st.columns([0.15,0.05, 0.4,0.05, 0.35])

# ── Columna 3: selector único de gráfica ─────────────────────────────
with col_4:
    st.subheader("Gráficas")

    opciones_graficas = []
    if fases_completas:
        opciones_graficas.append("Desglose por fases")
    if not df_cam.empty:
        opciones_graficas.append("Cámara: sentado / sobrepasado la marca")
    if not df_s_imu.empty:
        opciones_graficas.append("IMU: magnitud de aceleración")
        opciones_graficas.append("IMU: magnitud de giroscopio")


    if not opciones_graficas:
        st.info("No hay datos suficientes para mostrar gráficas de esta prueba.")
    else:
        grafica_sel = st.selectbox("Selecciona una gráfica", opciones_graficas)

        if grafica_sel == "Desglose por fases":
            fig_fases = figura_base()
            fig_fases.add_pie(
                labels=list(fases.keys()),
                values=list(fases.values()),
                hole=0.55,
                marker=dict(colors=[FASE_COLORES[nombre] for nombre in fases]),
                text=[f"{valor:.1f}s" for valor in fases.values()],
                textinfo="text",
                hovertemplate="%{label}: %{value:.1f}s (%{percent})<extra></extra>",
            )
            fig_fases.update_layout(
                height=ALTURA_GRAFICA,
                xaxis_visible=False,
                yaxis_visible=False,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="right",
                    x=-0.1,
                ),
                annotations=[dict(
                    text=f"{fila['duracion_prueba_s']:.1f}s",
                    x=0.5, y=0.5,
                    font=dict(size=18, color=TEXT_PRIMARY),
                    showarrow=False,
                )],
            )
            st.plotly_chart(fig_fases, width='stretch')

        elif grafica_sel == "Cámara: sentado / sobrepasado la marca":
            fig_cam = figura_base()
            fig_cam.add_scatter(
                x=df_cam["t_s"], y=df_cam["sentado"].astype(int), mode="lines", name="Sentado",
                line=dict(color=BLUE, width=2, shape="hv"), hovertemplate="t=%{x:.1f}s<extra></extra>",
            )
            fig_cam.add_scatter(
                x=df_cam["t_s"], y=df_cam["sobrepasado"].astype(int) - 1.3, mode="lines", name="Sobrepasado la marca",
                line=dict(color=ORANGE, width=2, shape="hv"), hovertemplate="t=%{x:.1f}s<extra></extra>",
            )
            fig_cam.update_layout(
                height=ALTURA_GRAFICA, xaxis_title="segundos desde el inicio",
                yaxis=dict(tickvals=[0, 1, -1.3, -0.3], ticktext=["No", "Sí", "No", "Sí"], gridcolor=GRIDLINE),
            )
            st.plotly_chart(fig_cam, width='stretch')

        elif grafica_sel == "IMU: magnitud de aceleración":
            fig_accel = figura_base()
            fig_accel.add_scatter(
                x=df_s_imu["t_s"], y=df_s_imu["accel_mag"], mode="lines", name="Aceleración",
                line=dict(color=BLUE, width=2), showlegend=False,
                hovertemplate="t=%{x:.1f}s<br>%{y:.2f} g<extra></extra>",
            )
            anadir_marcas_fase(fig_accel, marcas_fase, df_s_imu, "accel_mag")
            fig_accel.update_layout(height=ALTURA_GRAFICA, xaxis_title="segundos", yaxis_title="g")
            st.plotly_chart(fig_accel, width='stretch')

        elif grafica_sel == "IMU: magnitud de giroscopio":
            fig_gyro = figura_base()
            fig_gyro.add_scatter(
                x=df_s_imu["t_s"], y=df_s_imu["gyro_mag"], mode="lines", name="Giroscopio",
                line=dict(color=BLUE, width=2), showlegend=False,
                hovertemplate="t=%{x:.1f}s<br>%{y:.0f} °/s<extra></extra>",
            )
            anadir_marcas_fase(fig_gyro, marcas_fase, df_s_imu, "gyro_mag")
            fig_gyro.update_layout(height=ALTURA_GRAFICA, xaxis_title="segundos", yaxis_title="°/s")
            st.plotly_chart(fig_gyro, width='stretch')



# ── Columna 4: tablas de datos (Gold + Silver, como antes) ───────────
with col_5:
    st.subheader("Datos")

    st.caption("Fila de resultados")
    st.dataframe(
        fila.drop(labels="prueba_corta").to_frame().T,
        width="stretch",
            hide_index=True,
    )
    st.caption("Datos IMU")
    if not df_s_imu.empty:
        st.dataframe(df_s_imu, width='stretch', height=270,     hide_index=True,)
    else:
        st.info("Sin datos de IMU (Silver, calidad_ok) para esta prueba.")

# ── Columna 5: métricas apiladas verticalmente ───────────────────────
with col_3:
    render_metricas(metricas, vertical=True)