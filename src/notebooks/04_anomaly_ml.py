# Databricks notebook source
# MAGIC %md
# MAGIC 🏠 **Índice:** `00_LEEME`   ·   ⬅️ **Anterior:** `03_anomaly_rules`   ·   **Siguiente ➡️:** `05_genai_enrichment`

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "S": {"col": 0, "row": 0, "label": "lecturas_serie"},
        "F": {"col": 1, "row": 0, "label": "Features"},
        "IF": {"col": 2, "row": 0, "label": "Isolation\nForest"},
        "ML": {"col": 3, "row": 0, "label": "MLflow run"},
        "UC": {"col": 4, "row": 0, "label": "UC Model\nRegistry @champion"},
        "SC": {"col": 5, "row": 0, "label": "anomalias_ml\n(scoring)"},
    },
    [("S", "F"), ("F", "IF"), ("IF", "ML"), ("ML", "UC"), ("UC", "SC")],
    highlight_ids=["IF", "UC"],
    caption="04 · ML no supervisado → registrado y gobernado en Unity Catalog",
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 · Detección de anomalías con Machine Learning
# MAGIC
# MAGIC **Capacidad (parte 2):** *Desarrollo y operación de modelos analíticos para detección de anomalías.*
# MAGIC
# MAGIC Las reglas cubren fallas **conocidas**. Un modelo **no supervisado** encuentra lo que no
# MAGIC anticipamos, mirando todas las variables **a la vez** (multivariado).
# MAGIC
# MAGIC - Modelo: **Isolation Forest** (scikit-learn). Alternativa a escala: **PyOD/ECOD** (ver `daxs`).
# MAGIC - **MLflow** registra experimento + modelo; lo **registramos en Unity Catalog**.
# MAGIC - Empaquetamos como **pyfunc** (scaler + modelo) para servirlo igual en batch y en tiempo real (notebook 06).
# MAGIC - Explicabilidad simple: qué variable aporta más a cada anomalía (SHAP = camino más profundo).

# COMMAND ----------

dbutils.widgets.text("catalog", "oilgas_anomaly_demo", "Catálogo")
dbutils.widgets.text("schema", "pozos", "Esquema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

MODELO_UC = f"{CATALOG}.{SCHEMA}.detector_anomalias"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tabla de features
# MAGIC
# MAGIC Variables de sensor por lectura. (En producción se agregarían rolling features y se usaría
# MAGIC Feature Store/Feature Engineering en UC para reuso y linaje.)

# COMMAND ----------

FEATURES = [
    "presion_tubing_psi", "presion_casing_psi", "temperatura_motor_c",
    "corriente_motor_a", "vibracion_mm_s", "frecuencia_vsd_hz",
    "caudal_bruto_m3d", "corte_agua_pct", "gor_m3m3",
    "posicion_choke_pct", "tasa_arena_ppm",
]

pdf = (spark.table("lecturas_serie")
       .select("pozo_id", "ts", *FEATURES)
       .toPandas())
print(f"Filas: {len(pdf):,}  |  Features: {len(FEATURES)}")
pdf.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Entrenamiento + MLflow + registro en UC
# MAGIC
# MAGIC Entrenamos Isolation Forest sobre todas las lecturas (no supervisado). Lo empaquetamos como
# MAGIC `pyfunc` que devuelve `anomaly_score` (0..1, mayor = más anómalo) y `es_anomalia`.

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from mlflow.models.signature import infer_signature

mlflow.set_registry_uri("databricks-uc")  # registrar modelos en Unity Catalog

class DetectorAnomalias(mlflow.pyfunc.PythonModel):
    """Scaler + Isolation Forest. Entrada: columnas de sensor. Salida: score + flag."""
    def __init__(self, scaler, modelo, features):
        self.scaler, self.modelo, self.features = scaler, modelo, features
    def predict(self, context, model_input):
        X = self.scaler.transform(model_input[self.features])
        # score_samples: mayor = más normal. Invertimos y normalizamos a 0..1.
        raw = -self.modelo.score_samples(X)
        score = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        pred = (self.modelo.predict(X) == -1).astype(int)
        return pd.DataFrame({"anomaly_score": score, "es_anomalia": pred})

X = pdf[FEATURES]
scaler = StandardScaler().fit(X)
iforest = IsolationForest(n_estimators=200, contamination=0.03, random_state=42).fit(scaler.transform(X))

with mlflow.start_run(run_name="isolation_forest_pozos") as run:
    mlflow.log_params({"n_estimators": 200, "contamination": 0.03, "n_features": len(FEATURES)})
    modelo = DetectorAnomalias(scaler, iforest, FEATURES)
    ejemplo = X.head(5)
    firma = infer_signature(ejemplo, modelo.predict(None, ejemplo))
    mlflow.pyfunc.log_model(
        artifact_path="modelo",
        python_model=modelo,
        signature=firma,
        input_example=ejemplo,
        pip_requirements=["scikit-learn", "pandas", "numpy"],
        registered_model_name=MODELO_UC,
    )
    run_id = run.info.run_id
print(f"Modelo registrado en UC: {MODELO_UC}  (run {run_id})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Scoring batch
# MAGIC
# MAGIC Cargamos el modelo desde UC y puntuamos todas las lecturas. Guardamos `anomalias_ml`.

# COMMAND ----------

modelo_cargado = mlflow.pyfunc.load_model(f"models:/{MODELO_UC}@champion"
                                          if False else f"runs:/{run_id}/modelo")
pred = modelo_cargado.predict(pdf[FEATURES])
scored = pd.concat([pdf[["pozo_id", "ts"]].reset_index(drop=True), pred], axis=1)

(spark.createDataFrame(scored)
      .write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{CATALOG}.{SCHEMA}.anomalias_ml"))

display(spark.sql("""
  SELECT pozo_id, COUNT(*) AS lecturas, SUM(es_anomalia) AS anomalias,
         ROUND(AVG(anomaly_score),3) AS score_medio
  FROM anomalias_ml GROUP BY pozo_id ORDER BY anomalias DESC LIMIT 8
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Alias de producción
# MAGIC
# MAGIC Promovemos la versión recién registrada con el alias `champion` (lo que servirá el endpoint).

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
ultima = max(int(mv.version) for mv in client.search_model_versions(f"name = '{MODELO_UC}'"))
client.set_registered_model_alias(MODELO_UC, "champion", ultima)
print(f"{MODELO_UC} v{ultima} → alias 'champion'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Evaluación contra las etiquetas
# MAGIC
# MAGIC ¿Las anomalías ML caen dentro de los eventos de falla reales? (recall por evento)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT e.modo_falla,
# MAGIC   COUNT(DISTINCT e.pozo_id || e.ts_inicio) AS eventos,
# MAGIC   SUM(CASE WHEN m.anomalias_dentro > 0 THEN 1 ELSE 0 END) AS eventos_detectados
# MAGIC FROM eventos_falla e
# MAGIC LEFT JOIN (
# MAGIC   SELECT e.pozo_id, e.ts_inicio, COUNT(*) AS anomalias_dentro
# MAGIC   FROM eventos_falla e JOIN anomalias_ml a
# MAGIC     ON a.pozo_id = e.pozo_id AND a.ts BETWEEN e.ts_inicio AND e.ts_fin AND a.es_anomalia = 1
# MAGIC   GROUP BY e.pozo_id, e.ts_inicio
# MAGIC ) m ON m.pozo_id = e.pozo_id AND m.ts_inicio = e.ts_inicio
# MAGIC GROUP BY e.modo_falla ORDER BY eventos DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Explicabilidad simple
# MAGIC
# MAGIC Para cada anomalía, qué variable está más lejos de la media del pozo (|z-score|). Da una
# MAGIC razón accionable al operador. Para atribución formal → **SHAP** (ver `daxs` del acelerador).

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH base AS (
# MAGIC   SELECT a.pozo_id, a.ts, a.anomaly_score,
# MAGIC          s.temperatura_motor_c, s.tasa_arena_ppm, s.gor_m3m3, s.presion_tubing_psi
# MAGIC   FROM anomalias_ml a JOIN lecturas_serie s USING (pozo_id, ts)
# MAGIC   WHERE a.es_anomalia = 1
# MAGIC )
# MAGIC SELECT pozo_id, ts, ROUND(anomaly_score,3) AS score,
# MAGIC   ROUND(temperatura_motor_c,1) AS temp, ROUND(tasa_arena_ppm,1) AS arena,
# MAGIC   ROUND(gor_m3m3,1) AS gor, ROUND(presion_tubing_psi,0) AS p_tubing
# MAGIC FROM base ORDER BY anomaly_score DESC LIMIT 15;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Puntos de conversación
# MAGIC - **No supervisado:** no hace falta etiquetar fallas para arrancar — ideal cuando hay poco histórico de fallas.
# MAGIC - **Reglas + ML juntos:** reglas para lo conocido/explicable, ML para lo nuevo. Se combinan en una sola alerta.
# MAGIC - **Modelo en UC:** mismo gobierno que los datos — versiones, alias (`champion`), permisos, linaje.
# MAGIC - **A escala:** `daxs` (PyOD/ECOD) entrena un detector por equipo sobre miles de activos con SHAP.
