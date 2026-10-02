# Databricks notebook source
# MAGIC %md
# MAGIC 🏠 **Índice:** `00_LEEME`   ·   ⬅️ **Anterior:** `05_genai_enrichment`   ·   **Siguiente ➡️:** `07_feedback_loop`

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "UC": {"col": 0, "row": 0, "label": "UC Model\n@champion"},
        "EP": {"col": 1, "row": 0, "label": "Serving endpoint\n(scale-to-zero)"},
        "PY": {"col": 2, "row": 0, "label": "Cliente Python\n(app / backend)"},
        "SQ": {"col": 2, "row": 1, "label": "ai_query en SQL\n(analista / BI)"},
    },
    [("UC", "EP"), ("EP", "PY"), ("EP", "SQ")],
    highlight_ids=["EP"],
    caption="06 · Un modelo, dos consumos: REST y SQL",
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 06 · Model Serving + AI Functions
# MAGIC
# MAGIC **Capacidad:** *Uso de Model Serving y AI Functions para consumir modelos desde aplicaciones.*
# MAGIC
# MAGIC Tomamos el detector registrado en UC (notebook 04) y lo exponemos como **endpoint REST**.
# MAGIC Luego lo consumimos de dos formas:
# MAGIC
# MAGIC 1. **Cliente Python/REST** — como lo haría una app o un backend.
# MAGIC 2. **`ai_query` desde SQL** — como lo haría un analista o un dashboard, sin código.
# MAGIC
# MAGIC > Crear el endpoint tarda unos minutos (provisioning). En vivo: lanzarlo al inicio del módulo
# MAGIC > y seguir hablando mientras levanta.

# COMMAND ----------

# Garantizar mlflow + SDK (en notebook serverless interactivo ya están; esto lo hace a prueba de entornos)
%pip install -q mlflow
dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "oilgas_anomaly_demo", "Catálogo")
dbutils.widgets.text("schema", "pozos", "Esquema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

MODELO_UC = f"{CATALOG}.{SCHEMA}.detector_anomalias"
ENDPOINT = "detector-anomalias-pozos"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Crear el endpoint de Model Serving
# MAGIC
# MAGIC Servimos la versión con alias `champion`. `scale_to_zero` apaga el cómputo cuando no hay
# MAGIC tráfico — ideal para demo y para cargas intermitentes.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
# Resolver la versión apuntada por el alias 'champion' (notebook 04)
version_champion = MlflowClient().get_model_version_by_alias(MODELO_UC, "champion").version
print(f"Sirviendo {MODELO_UC} v{version_champion}")

w = WorkspaceClient()
served = ServedEntityInput(
    entity_name=MODELO_UC,
    entity_version=version_champion,
    scale_to_zero_enabled=True,
    workload_size="Small",
)

existentes = [e.name for e in w.serving_endpoints.list()]
if ENDPOINT in existentes:
    w.serving_endpoints.update_config(name=ENDPOINT, served_entities=[served])
    print(f"Endpoint '{ENDPOINT}' actualizado.")
else:
    w.serving_endpoints.create(
        name=ENDPOINT,
        config=EndpointCoreConfigInput(name=ENDPOINT, served_entities=[served]),
    )
    print(f"Endpoint '{ENDPOINT}' creándose… (unos minutos)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Esperar a que quede READY
# MAGIC (En vivo, saltar esta celda y volver cuando el endpoint esté listo.)

# COMMAND ----------

w.serving_endpoints.wait_get_serving_endpoint_not_updating(ENDPOINT)
estado = w.serving_endpoints.get(ENDPOINT).state
print(f"Estado del endpoint: {estado}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Consumo A — cliente Python/REST (como una app)

# COMMAND ----------

import pandas as pd

muestra = (spark.table("lecturas_serie")
           .select("presion_tubing_psi", "presion_casing_psi", "temperatura_motor_c",
                   "corriente_motor_a", "vibracion_mm_s", "frecuencia_vsd_hz",
                   "caudal_bruto_m3d", "corte_agua_pct", "gor_m3m3",
                   "posicion_choke_pct", "tasa_arena_ppm")
           .limit(5).toPandas())

resp = w.serving_endpoints.query(
    name=ENDPOINT,
    dataframe_records=muestra.to_dict(orient="records"),
)
print(resp.predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Consumo B — `ai_query` desde SQL (como un analista)
# MAGIC
# MAGIC La misma predicción, invocando el endpoint directamente en SQL. Esto es lo que permite
# MAGIC puntuar millones de filas o integrarlo en un dashboard sin escribir una app.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT pozo_id, ts,
# MAGIC   ai_query(
# MAGIC     'detector-anomalias-pozos',
# MAGIC     named_struct(
# MAGIC       'presion_tubing_psi', presion_tubing_psi, 'presion_casing_psi', presion_casing_psi,
# MAGIC       'temperatura_motor_c', temperatura_motor_c, 'corriente_motor_a', corriente_motor_a,
# MAGIC       'vibracion_mm_s', vibracion_mm_s, 'frecuencia_vsd_hz', frecuencia_vsd_hz,
# MAGIC       'caudal_bruto_m3d', caudal_bruto_m3d, 'corte_agua_pct', corte_agua_pct,
# MAGIC       'gor_m3m3', gor_m3m3, 'posicion_choke_pct', posicion_choke_pct,
# MAGIC       'tasa_arena_ppm', tasa_arena_ppm
# MAGIC     )
# MAGIC   ) AS score_anomalia
# MAGIC FROM lecturas_serie
# MAGIC WHERE pozo_id = 'POZO-001'
# MAGIC LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Puntos de conversación
# MAGIC - **Un modelo, dos consumos:** la app usa REST; el analista usa SQL. Mismo modelo gobernado en UC.
# MAGIC - **`scale_to_zero`:** pagás cómputo solo cuando hay inferencia.
# MAGIC - **Inference Tables:** el endpoint puede loguear cada request/response a una tabla Delta → base del
# MAGIC   monitoreo y del feedback (notebook 07).
# MAGIC - **La "app" es la evolución:** hoy mostramos el endpoint; una Databricks App encima (fleet view
# MAGIC   por pozo) es el siguiente paso (ver notebook 08).
