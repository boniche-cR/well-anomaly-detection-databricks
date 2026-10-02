# Databricks notebook source
# MAGIC %md
# MAGIC 🏠 **Índice:** `00_LEEME`   ·   **Siguiente ➡️:** `01_ingestion_incremental`
# MAGIC
# MAGIC *(Navegá abriendo cada notebook desde el árbol de archivos, en orden numérico.)*

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "G": {"col": 0, "row": 2, "label": "Generador\nsintético"},
        "D": {"col": 1, "row": 0, "label": "dim_pozo"},
        "H": {"col": 1, "row": 1, "label": "lecturas_historico\n(75 días · Delta)"},
        "V": {"col": 1, "row": 2, "label": "Volume landing\n(15 días · JSON)"},
        "E": {"col": 1, "row": 3, "label": "eventos_falla\n(etiquetas)"},
        "C": {"col": 1, "row": 4, "label": "comentarios\noperativos (ES)"},
    },
    [("G", "D"), ("G", "H"), ("G", "V"), ("G", "E"), ("G", "C")],
    highlight_ids=["G"],
    caption="00 · Setup — genera datos sintéticos de pozos en Unity Catalog",
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 00 · Setup y generación de datos sintéticos de pozos
# MAGIC
# MAGIC **Solution Accelerator — Detección de Anomalías en Pozos**
# MAGIC
# MAGIC Este notebook prepara el entorno y genera datos realistas (sintéticos) para todo el workshop:
# MAGIC
# MAGIC 1. Crea el **catálogo**, **esquema** y un **Volume** de Unity Catalog.
# MAGIC 2. Genera la dimensión de **pozos** (dos yacimientos de ejemplo, no convencional).
# MAGIC 3. Genera series de tiempo horarias de **sensores** con *firmas de falla* inyectadas
# MAGIC    (sobrecalentamiento de ESP, candado de gas, arenamiento, declinación).
# MAGIC 4. Genera **comentarios operativos** en español (para la parte de IA generativa).
# MAGIC 5. Genera **eventos de falla** (etiquetas) para evaluar los modelos.
# MAGIC
# MAGIC El histórico se escribe como tabla Delta; los últimos días se dejan como **archivos**
# MAGIC en el Volume para que el notebook `01` los ingiera de forma **incremental** con Auto Loader.
# MAGIC
# MAGIC > Correr una sola vez (o vía el job `generar_datos`). Es idempotente.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parámetros

# COMMAND ----------

dbutils.widgets.text("catalog", "oilgas_anomaly_demo", "Catálogo")
dbutils.widgets.text("schema", "pozos", "Esquema")
dbutils.widgets.text("landing_volume", "landing", "Volume de landing")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
LANDING_VOLUME = dbutils.widgets.get("landing_volume")

print(f"Catálogo        : {CATALOG}")
print(f"Esquema         : {SCHEMA}")
print(f"Volume landing  : {LANDING_VOLUME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Unity Catalog: catálogo, esquema y volume
# MAGIC
# MAGIC Todo vive bajo UC desde el minuto cero — gobierno, linaje y permisos salen gratis.

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{LANDING_VOLUME}")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{LANDING_VOLUME}"
RAW_PATH = f"{VOLUME_PATH}/sensores_raw"      # archivos crudos para Auto Loader (notebook 01)
print(f"Volume path     : {VOLUME_PATH}")
print(f"Raw landing     : {RAW_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Dimensión de pozos
# MAGIC
# MAGIC 20 pozos repartidos en dos yacimientos de ejemplo, con distintos sistemas de
# MAGIC levantamiento artificial. El tipo de lifting condiciona qué fallas aplican.

# COMMAND ----------

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

rng = np.random.default_rng(42)  # reproducible

N_POZOS = 20
CAMPOS = ["Yacimiento Norte", "Yacimiento Sur"]
TIPOS_LIFTING = ["ESP", "PCP", "Gas Lift"]

pozos = []
for i in range(1, N_POZOS + 1):
    campo = CAMPOS[0] if i <= 13 else CAMPOS[1]
    pad = f"Pad-{((i - 1) // 4) + 1:02d}"
    tipo = TIPOS_LIFTING[(i - 1) % 3]
    pozos.append({
        "pozo_id": f"POZO-{i:03d}",
        "nombre": f"YN.{i:03d}" if campo == CAMPOS[0] else f"YS.{i:03d}",
        "pad": pad,
        "campo": campo,
        "cuenca": "Cuenca no convencional (ejemplo)",
        "tipo_lifting": tipo,
        "profundidad_m": int(rng.integers(2600, 3200)),
        "fecha_alta": (datetime(2023, 1, 1) + timedelta(days=int(rng.integers(0, 500)))).date(),
        "latitud": round(-37.8 + rng.normal(0, 0.15), 5),
        "longitud": round(-68.9 + rng.normal(0, 0.15), 5),
    })

df_pozos = pd.DataFrame(pozos)
(spark.createDataFrame(df_pozos)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{CATALOG}.{SCHEMA}.dim_pozo"))

display(spark.table(f"{CATALOG}.{SCHEMA}.dim_pozo"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Series de sensores con firmas de falla
# MAGIC
# MAGIC Lecturas horarias por 90 días. Variables típicas de pozo + equipo de lifting:
# MAGIC presión de tubing/casing, temperatura y corriente del motor, vibración, frecuencia del
# MAGIC variador (VSD), caudal bruto/neto, corte de agua, GOR, posición de choke y tasa de arena.
# MAGIC
# MAGIC Inyectamos **4 modos de falla** con firmas temporales distintas (para enseñar reglas,
# MAGIC `MATCH_RECOGNIZE`, modelos ML y clasificación con IA):
# MAGIC
# MAGIC | Modo | Firma | Dónde se detecta mejor |
# MAGIC |---|---|---|
# MAGIC | **Sobrecalentamiento ESP** | temp motor ↑ sostenida + corriente ↑ → trip | reglas / MATCH_RECOGNIZE |
# MAGIC | **Candado de gas (gas lock)** | GOR ↑, caudal ↓, presión oscila | ML multivariado |
# MAGIC | **Arenamiento** | tasa de arena ↑, erosión de choke, presión ↓ | reglas + ML |
# MAGIC | **Declinación / fuga** | caída estructural lenta de presión y caudal | cambio estructural |

# COMMAND ----------

DIAS = 90
FREQ_H = 1
ts_inicio = datetime(2026, 7, 1, 0, 0, 0)
n_pasos = int(DIAS * 24 / FREQ_H)
timestamps = [ts_inicio + timedelta(hours=FREQ_H * k) for k in range(n_pasos)]

# Firmas de falla: (pozo_idx, modo, paso_inicio, duracion_pasos)
eventos = [
    (0,  "sobrecalentamiento_esp", 1400, 60),
    (3,  "sobrecalentamiento_esp", 1800, 48),
    (6,  "candado_gas",            1200, 72),
    (9,  "candado_gas",            1600, 60),
    (1,  "arenamiento",            1500, 90),
    (12, "arenamiento",            1300, 80),
    (15, "declinacion_fuga",        900, 400),
    (4,  "declinacion_fuga",       1000, 500),
]

def base_series(pozo, n):
    """Línea base realista por pozo, con ciclo diario leve y ruido."""
    hora = np.arange(n) % 24
    ciclo = np.sin(2 * np.pi * hora / 24)
    base = {
        "presion_tubing_psi":   1800 + 120 * rng.standard_normal() + 40 * ciclo + rng.normal(0, 15, n),
        "presion_casing_psi":   2200 + 120 * rng.standard_normal() + 30 * ciclo + rng.normal(0, 15, n),
        "temperatura_motor_c":    82 +   4 * rng.standard_normal() +  2 * ciclo + rng.normal(0, 1.2, n),
        "corriente_motor_a":      45 +   3 * rng.standard_normal() +  1 * ciclo + rng.normal(0, 0.8, n),
        "vibracion_mm_s":        2.5 + 0.3 * rng.standard_normal() + rng.normal(0, 0.15, n),
        "frecuencia_vsd_hz":      52 +   2 * rng.standard_normal() + rng.normal(0, 0.3, n),
        "caudal_bruto_m3d":      320 +  40 * rng.standard_normal() + 15 * ciclo + rng.normal(0, 8, n),
        "corte_agua_pct":         35 +   8 * rng.standard_normal() + rng.normal(0, 1.5, n),
        "gor_m3m3":              110 +  20 * rng.standard_normal() + rng.normal(0, 4, n),
        "posicion_choke_pct":     60 +   5 * rng.standard_normal() + rng.normal(0, 1.0, n),
        "tasa_arena_ppm":          8 +   2 * rng.standard_normal() + rng.normal(0, 0.6, n),
    }
    return {k: v.astype(float) for k, v in base.items()}

def aplica_falla(s, modo, ini, dur, n):
    """Suma la firma de falla sobre la línea base, en el rango [ini, ini+dur)."""
    fin = min(ini + dur, n)
    rampa = np.zeros(n)
    rampa[ini:fin] = np.linspace(0, 1, fin - ini)
    if modo == "sobrecalentamiento_esp":
        s["temperatura_motor_c"] += 35 * rampa
        s["corriente_motor_a"]   += 18 * rampa
        s["vibracion_mm_s"]      += 2.0 * rampa
    elif modo == "candado_gas":
        osc = np.zeros(n); osc[ini:fin] = np.sin(np.linspace(0, 20 * np.pi, fin - ini))
        s["gor_m3m3"]            += 180 * rampa
        s["caudal_bruto_m3d"]    -= 160 * rampa
        s["presion_tubing_psi"]  += 300 * osc * rampa
    elif modo == "arenamiento":
        s["tasa_arena_ppm"]      += 90 * rampa
        s["presion_tubing_psi"]  -= 250 * rampa
        s["posicion_choke_pct"]  += 20 * rampa
    elif modo == "declinacion_fuga":
        s["presion_tubing_psi"]  -= 400 * rampa
        s["presion_casing_psi"]  -= 300 * rampa
        s["caudal_bruto_m3d"]    -= 180 * rampa
    return s

# mapa pozo_idx -> lista de (modo, ini, dur)
mapa_eventos = {}
for pozo_idx, modo, ini, dur in eventos:
    mapa_eventos.setdefault(pozo_idx, []).append((modo, ini, dur))

filas = []
etiquetas = []
for idx, row in df_pozos.reset_index(drop=True).iterrows():
    s = base_series(row, n_pasos)
    for (modo, ini, dur) in mapa_eventos.get(idx, []):
        s = aplica_falla(s, modo, ini, dur, n_pasos)
        etiquetas.append({
            "pozo_id": row["pozo_id"],
            "modo_falla": modo,
            "ts_inicio": timestamps[ini],
            "ts_fin": timestamps[min(ini + dur, n_pasos) - 1],
        })
    # caudal neto a partir de bruto y corte de agua
    neto = s["caudal_bruto_m3d"] * (1 - np.clip(s["corte_agua_pct"], 0, 100) / 100)
    for k in range(n_pasos):
        filas.append({
            "pozo_id": row["pozo_id"],
            "ts": timestamps[k],
            "presion_tubing_psi": round(float(s["presion_tubing_psi"][k]), 2),
            "presion_casing_psi": round(float(s["presion_casing_psi"][k]), 2),
            "temperatura_motor_c": round(float(s["temperatura_motor_c"][k]), 2),
            "corriente_motor_a": round(float(s["corriente_motor_a"][k]), 2),
            "vibracion_mm_s": round(float(s["vibracion_mm_s"][k]), 3),
            "frecuencia_vsd_hz": round(float(s["frecuencia_vsd_hz"][k]), 2),
            "caudal_bruto_m3d": round(float(s["caudal_bruto_m3d"][k]), 2),
            "caudal_neto_m3d": round(float(neto[k]), 2),
            "corte_agua_pct": round(float(np.clip(s["corte_agua_pct"][k], 0, 100)), 2),
            "gor_m3m3": round(float(s["gor_m3m3"][k]), 2),
            "posicion_choke_pct": round(float(np.clip(s["posicion_choke_pct"][k], 0, 100)), 2),
            "tasa_arena_ppm": round(float(max(s["tasa_arena_ppm"][k], 0)), 2),
        })

df_lecturas = pd.DataFrame(filas)
df_eventos = pd.DataFrame(etiquetas)
print(f"Lecturas generadas: {len(df_lecturas):,}  |  Eventos de falla: {len(df_eventos)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Partir histórico vs incremental
# MAGIC
# MAGIC - **Histórico** (primeros 75 días) → tabla Delta `lecturas_historico`.
# MAGIC - **Incremental** (últimos 15 días) → archivos JSON por día en el Volume, para que
# MAGIC   el notebook `01` los ingiera con Auto Loader simulando llegada continua.

# COMMAND ----------

corte = ts_inicio + timedelta(days=75)
df_hist = df_lecturas[df_lecturas["ts"] < corte].copy()
df_inc = df_lecturas[df_lecturas["ts"] >= corte].copy()

# Histórico como Delta
(spark.createDataFrame(df_hist)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{CATALOG}.{SCHEMA}.lecturas_historico"))
print(f"lecturas_historico: {len(df_hist):,} filas")

# Eventos de falla (etiquetas ground-truth)
(spark.createDataFrame(df_eventos)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{CATALOG}.{SCHEMA}.eventos_falla"))
display(spark.table(f"{CATALOG}.{SCHEMA}.eventos_falla"))

# COMMAND ----------

# Incremental → archivos JSON por día en el Volume (uno por fecha)
dbutils.fs.rm(RAW_PATH, recurse=True)
dbutils.fs.mkdirs(RAW_PATH)

df_inc["fecha"] = df_inc["ts"].dt.date.astype(str)
for fecha, grupo in df_inc.groupby("fecha"):
    sdf = spark.createDataFrame(grupo.drop(columns=["fecha"]))
    (sdf.coalesce(1).write.mode("overwrite").format("json")
        .save(f"{RAW_PATH}/fecha={fecha}"))

print(f"Archivos incrementales escritos en {RAW_PATH} ({df_inc['fecha'].nunique()} días)")
display(dbutils.fs.ls(RAW_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Comentarios operativos (español) — para IA generativa
# MAGIC
# MAGIC Texto libre del RTOC / operadores de campo. Mezclamos comentarios de rutina con otros
# MAGIC que describen las fallas inyectadas. El notebook `05` usará AI Functions para
# MAGIC clasificarlos en modo de falla + severidad, sin que nadie los lea a mano.

# COMMAND ----------

plantillas = {
    "sobrecalentamiento_esp": [
        "Motor de la ESP operando por encima de 110°C, corriente subiendo. Se recomienda bajar frecuencia del variador.",
        "Alarma de alta temperatura en motor del {pozo}. Vibración en aumento, posible desgaste de rodamientos.",
        "ESP del {pozo} con sobrecarga térmica sostenida; evaluar parada controlada antes del trip.",
    ],
    "candado_gas": [
        "Caudal cayó fuerte en {pozo} con GOR disparado, parece interferencia/candado de gas.",
        "Presión de tubing oscilando y producción errática en {pozo}; sospecha de gas lock en la bomba.",
        "Gas interfiriendo el llenado de la bomba en {pozo}, caudal neto a la baja.",
    ],
    "arenamiento": [
        "Incremento marcado de arena en {pozo}, se observa erosión en el choke y caída de presión.",
        "Producción de arena fuera de rango en {pozo}; riesgo de abrasión en instalaciones de superficie.",
        "Arena por encima del umbral en {pozo}, ajustar choke y monitorear presión de tubing.",
    ],
    "declinacion_fuga": [
        "Declinación progresiva de presión y caudal en {pozo} durante los últimos días.",
        "Caída estructural sostenida en {pozo}; descartar fuga en casing o agotamiento de reservorio.",
        "Tendencia bajista lenta pero continua en presión de {pozo}, agendar revisión de integridad.",
    ],
    "rutina": [
        "Pozo {pozo} operando en parámetros normales, sin novedad en el turno.",
        "Chequeo de rutina {pozo}: presiones y caudal dentro de rango. Todo OK.",
        "Mantenimiento preventivo programado para {pozo} la semana próxima.",
        "{pozo} estable tras ajuste de choke de ayer; producción conforme a plan.",
    ],
}

comentarios = []
cid = 0
# Comentarios ligados a cada evento de falla
for _, ev in df_eventos.iterrows():
    for txt in rng.choice(plantillas[ev["modo_falla"]], size=2, replace=False):
        cid += 1
        ts_c = ev["ts_inicio"] + timedelta(hours=int(rng.integers(0, 24)))
        comentarios.append({
            "comentario_id": cid, "pozo_id": ev["pozo_id"],
            "ts": ts_c, "autor": f"op_{rng.integers(1, 9)}",
            "comentario": txt.format(pozo=ev["pozo_id"]),
        })
# Comentarios de rutina repartidos
for _, row in df_pozos.iterrows():
    for _ in range(rng.integers(2, 5)):
        cid += 1
        ts_c = ts_inicio + timedelta(hours=int(rng.integers(0, n_pasos)))
        comentarios.append({
            "comentario_id": cid, "pozo_id": row["pozo_id"],
            "ts": ts_c, "autor": f"op_{rng.integers(1, 9)}",
            "comentario": rng.choice(plantillas["rutina"]).format(pozo=row["pozo_id"]),
        })

df_coment = pd.DataFrame(comentarios).sort_values("ts")
(spark.createDataFrame(df_coment)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{CATALOG}.{SCHEMA}.comentarios_operativos"))
print(f"comentarios_operativos: {len(df_coment)} filas")
display(spark.table(f"{CATALOG}.{SCHEMA}.comentarios_operativos").limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Resumen
# MAGIC
# MAGIC Objetos creados en Unity Catalog:
# MAGIC
# MAGIC | Objeto | Qué es |
# MAGIC |---|---|
# MAGIC | `dim_pozo` | Dimensión de pozos (20) |
# MAGIC | `lecturas_historico` | Series de sensores, primeros 75 días (Delta) |
# MAGIC | `eventos_falla` | Etiquetas ground-truth de fallas inyectadas |
# MAGIC | `comentarios_operativos` | Texto libre en español para IA generativa |
# MAGIC | Volume `landing/sensores_raw/` | Archivos JSON de los últimos 15 días → Auto Loader (notebook 01) |
# MAGIC
# MAGIC Siguiente: **`01_ingestion_incremental`** — ingesta incremental con Auto Loader.

# COMMAND ----------

for t in ["dim_pozo", "lecturas_historico", "eventos_falla", "comentarios_operativos"]:
    n = spark.table(f"{CATALOG}.{SCHEMA}.{t}").count()
    print(f"{t:28s}: {n:,} filas")
