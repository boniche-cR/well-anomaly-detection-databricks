# Databricks notebook source
# MAGIC %md
# MAGIC 🏠 **Índice:** `00_LEEME`   ·   ⬅️ **Anterior:** `00_setup`   ·   **Siguiente ➡️:** `02_unity_catalog_governance`

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "V": {"col": 0, "row": 0, "label": "Volume\nsensores_raw"},
        "B": {"col": 1, "row": 0, "label": "bronze_lecturas"},
        "S": {"col": 2, "row": 0, "label": "silver_lecturas"},
        "U": {"col": 3, "row": 0, "label": "lecturas_serie\n(vista)"},
        "H": {"col": 1, "row": 1, "label": "lecturas_historico"},
    },
    [("V", "B", "Auto Loader"), ("B", "S", "MERGE + JOIN"), ("S", "U"), ("H", "U")],
    highlight_ids=["B", "S"],
    caption="01 · Ingesta incremental — bronze → silver → serie",
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 · Ingesta y procesamiento incremental
# MAGIC
# MAGIC **Capacidad:** *Ingesta y procesamiento incremental de datos sobre tablas históricas.*
# MAGIC
# MAGIC Mostramos el patrón **medallion** (bronze → silver) con **Auto Loader**:
# MAGIC
# MAGIC - Auto Loader detecta **solo los archivos nuevos** que llegan al Volume (ingesta incremental real).
# MAGIC - Escribe a una tabla **bronze** (crudo + metadatos de ingesta).
# MAGIC - Deriva una tabla **silver** tipada, limpia y unida a la dimensión de pozos.
# MAGIC - La silver se **une con el histórico** (`lecturas_historico`) para tener la serie completa.
# MAGIC
# MAGIC Usamos `trigger(availableNow=True)`: procesa lo disponible y se detiene — ideal para demo.
# MAGIC En producción, el mismo código corre en *streaming* continuo con `.trigger(processingTime=...)`.

# COMMAND ----------

dbutils.widgets.text("catalog", "oilgas_anomaly_demo", "Catálogo")
dbutils.widgets.text("schema", "pozos", "Esquema")
dbutils.widgets.text("landing_volume", "landing", "Volume de landing")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
LANDING_VOLUME = dbutils.widgets.get("landing_volume")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

VOLUME_PATH = f"/Volumes/{CATALOG}/{SCHEMA}/{LANDING_VOLUME}"
RAW_PATH = f"{VOLUME_PATH}/sensores_raw"
CHECKPOINT = f"{VOLUME_PATH}/_checkpoints/bronze_lecturas"
SCHEMA_LOC = f"{VOLUME_PATH}/_schemas/bronze_lecturas"
print(f"Leyendo archivos de: {RAW_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze — Auto Loader
# MAGIC
# MAGIC `cloudFiles` infiere y **evoluciona** el esquema, y lleva control de qué archivos ya procesó
# MAGIC (en el checkpoint). Si llegan archivos nuevos, solo esos se ingieren.

# COMMAND ----------

from pyspark.sql import functions as F

bronze_stream = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("cloudFiles.schemaLocation", SCHEMA_LOC)
    .option("cloudFiles.inferColumnTypes", "true")
    .load(RAW_PATH)
    # metadatos de ingesta — buena práctica para trazabilidad
    .withColumn("_archivo", F.col("_metadata.file_path"))
    .withColumn("_ingerido_ts", F.current_timestamp())
)

(bronze_stream.writeStream
    .format("delta")
    .option("checkpointLocation", CHECKPOINT)
    .trigger(availableNow=True)
    .toTable(f"{CATALOG}.{SCHEMA}.bronze_lecturas"))

# Esperar a que el micro-batch availableNow termine
for q in spark.streams.active:
    q.awaitTermination()

print("Bronze actualizada.")
display(spark.sql("SELECT COUNT(*) AS filas_bronze, COUNT(DISTINCT pozo_id) AS pozos FROM bronze_lecturas"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver — tipada, limpia y enriquecida
# MAGIC
# MAGIC Casteo de `ts`, filtro de nulos clave, y `JOIN` con `dim_pozo` para traer campo/tipo de lifting.
# MAGIC Aquí usamos un `MERGE` para que sea **idempotente** (reprocesable sin duplicar).

# COMMAND ----------

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.silver_lecturas (
  pozo_id STRING, ts TIMESTAMP, campo STRING, tipo_lifting STRING,
  presion_tubing_psi DOUBLE, presion_casing_psi DOUBLE,
  temperatura_motor_c DOUBLE, corriente_motor_a DOUBLE, vibracion_mm_s DOUBLE,
  frecuencia_vsd_hz DOUBLE, caudal_bruto_m3d DOUBLE, caudal_neto_m3d DOUBLE,
  corte_agua_pct DOUBLE, gor_m3m3 DOUBLE, posicion_choke_pct DOUBLE, tasa_arena_ppm DOUBLE
) USING DELTA
""")

spark.sql(f"""
MERGE INTO {CATALOG}.{SCHEMA}.silver_lecturas AS s
USING (
  SELECT b.pozo_id, CAST(b.ts AS TIMESTAMP) AS ts, d.campo, d.tipo_lifting,
         b.presion_tubing_psi, b.presion_casing_psi, b.temperatura_motor_c,
         b.corriente_motor_a, b.vibracion_mm_s, b.frecuencia_vsd_hz,
         b.caudal_bruto_m3d, b.caudal_neto_m3d, b.corte_agua_pct,
         b.gor_m3m3, b.posicion_choke_pct, b.tasa_arena_ppm
  FROM bronze_lecturas b
  JOIN dim_pozo d USING (pozo_id)
  WHERE b.pozo_id IS NOT NULL AND b.ts IS NOT NULL
) AS u
ON s.pozo_id = u.pozo_id AND s.ts = u.ts
WHEN NOT MATCHED THEN INSERT *
""")

display(spark.sql("SELECT COUNT(*) AS filas_silver FROM silver_lecturas"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Serie completa = histórico + incremental
# MAGIC
# MAGIC Vista que une el histórico (75 días) con lo que va llegando de forma incremental.
# MAGIC El resto del workshop (reglas, ML, IA) corre sobre `lecturas_serie`.

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE VIEW {CATALOG}.{SCHEMA}.lecturas_serie AS
SELECT h.pozo_id, h.ts, d.campo, d.tipo_lifting,
       h.presion_tubing_psi, h.presion_casing_psi, h.temperatura_motor_c,
       h.corriente_motor_a, h.vibracion_mm_s, h.frecuencia_vsd_hz,
       h.caudal_bruto_m3d, h.caudal_neto_m3d, h.corte_agua_pct,
       h.gor_m3m3, h.posicion_choke_pct, h.tasa_arena_ppm
FROM lecturas_historico h JOIN dim_pozo d USING (pozo_id)
UNION ALL
SELECT pozo_id, ts, campo, tipo_lifting,
       presion_tubing_psi, presion_casing_psi, temperatura_motor_c,
       corriente_motor_a, vibracion_mm_s, frecuencia_vsd_hz,
       caudal_bruto_m3d, caudal_neto_m3d, corte_agua_pct,
       gor_m3m3, posicion_choke_pct, tasa_arena_ppm
FROM silver_lecturas
""")

display(spark.sql("""
  SELECT pozo_id, MIN(ts) AS desde, MAX(ts) AS hasta, COUNT(*) AS lecturas
  FROM lecturas_serie GROUP BY pozo_id ORDER BY pozo_id LIMIT 5
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Puntos de conversación con el cliente
# MAGIC - **Incremental de verdad:** si volvés a correr este notebook sin archivos nuevos, Auto Loader no reprocesa nada.
# MAGIC - **Schema evolution:** si mañana el sensor agrega una columna, no se rompe la ingesta.
# MAGIC - **Exactly-once + idempotencia:** checkpoint en Auto Loader + `MERGE` en silver.
# MAGIC - **Mismo código, batch o streaming:** solo cambia el `trigger`.
