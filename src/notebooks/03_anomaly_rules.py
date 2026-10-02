# Databricks notebook source
# MAGIC %md
# MAGIC 🏠 **Índice:** `00_LEEME`   ·   ⬅️ **Anterior:** `02_unity_catalog_governance`   ·   **Siguiente ➡️:** `04_anomaly_ml`

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "S": {"col": 0, "row": 1, "label": "lecturas_serie"},
        "U": {"col": 1, "row": 0, "label": "Umbrales"},
        "Z": {"col": 1, "row": 1, "label": "Z-score móvil"},
        "MR": {"col": 1, "row": 2, "label": "MATCH_RECOGNIZE\n(secuencias)"},
        "A": {"col": 2, "row": 1, "label": "anomalias_reglas"},
        "E": {"col": 3, "row": 1, "label": "eventos_falla\n(etiquetas)"},
    },
    [("S", "U"), ("S", "Z"), ("S", "MR"), ("U", "A"), ("Z", "A"), ("MR", "A"),
     ("A", "E", "evalúa", True)],
    highlight_ids=["MR"],
    caption="03 · Reglas: umbrales + z-score + patrones secuenciales en SQL",
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 · Detección de anomalías con reglas analíticas (SQL)
# MAGIC
# MAGIC **Capacidad (parte 1):** *Desarrollo y operación de modelos analíticos para detección de anomalías.*
# MAGIC
# MAGIC Antes de ML, mucho valor se captura con **reglas en SQL**. Tres niveles, de simple a potente:
# MAGIC
# MAGIC 1. **Umbrales** — límites físicos/operativos conocidos por el modo de falla.
# MAGIC 2. **Z-score móvil** — detectar desvíos vs el comportamiento normal reciente del propio pozo.
# MAGIC 3. **`MATCH_RECOGNIZE`** — detectar *secuencias* (firmas temporales), no solo puntos aislados.
# MAGIC
# MAGIC Al final: la opción **sin código** (Anomaly Detection administrado) y `ai_forecast` (preview).

# COMMAND ----------

dbutils.widgets.text("catalog", "oilgas_anomaly_demo", "Catálogo")
dbutils.widgets.text("schema", "pozos", "Esquema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Reglas por umbral
# MAGIC
# MAGIC Rápidas, explicables y auditables. Codifican el conocimiento del dominio.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE anomalias_umbral AS
# MAGIC SELECT pozo_id, ts, campo, tipo_lifting,
# MAGIC   CASE
# MAGIC     WHEN temperatura_motor_c > 110 AND corriente_motor_a > 58 THEN 'sobrecalentamiento_esp'
# MAGIC     WHEN gor_m3m3 > 220 AND caudal_bruto_m3d < 220            THEN 'candado_gas'
# MAGIC     WHEN tasa_arena_ppm > 45                                  THEN 'arenamiento'
# MAGIC     WHEN presion_tubing_psi < 1500                            THEN 'declinacion_fuga'
# MAGIC     ELSE NULL
# MAGIC   END AS regla_falla,
# MAGIC   temperatura_motor_c, corriente_motor_a, gor_m3m3, caudal_bruto_m3d,
# MAGIC   tasa_arena_ppm, presion_tubing_psi
# MAGIC FROM lecturas_serie
# MAGIC WHERE
# MAGIC   (temperatura_motor_c > 110 AND corriente_motor_a > 58)
# MAGIC   OR (gor_m3m3 > 220 AND caudal_bruto_m3d < 220)
# MAGIC   OR tasa_arena_ppm > 45
# MAGIC   OR presion_tubing_psi < 1500;
# MAGIC
# MAGIC SELECT regla_falla, COUNT(*) AS alertas, COUNT(DISTINCT pozo_id) AS pozos
# MAGIC FROM anomalias_umbral GROUP BY regla_falla ORDER BY alertas DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Z-score móvil
# MAGIC
# MAGIC Los umbrales fijos no sirven si cada pozo tiene su propia "normalidad". El z-score móvil mide
# MAGIC cuántos desvíos estándar se aleja una lectura de la media **reciente del mismo pozo**
# MAGIC (ventana de 48 h). `|z| > 3` ≈ anomalía estadística.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE anomalias_zscore AS
# MAGIC WITH stats AS (
# MAGIC   SELECT pozo_id, ts, temperatura_motor_c,
# MAGIC     avg(temperatura_motor_c) OVER w  AS media_movil,
# MAGIC     stddev(temperatura_motor_c) OVER w AS desvio_movil
# MAGIC   FROM lecturas_serie
# MAGIC   WINDOW w AS (PARTITION BY pozo_id ORDER BY ts ROWS BETWEEN 48 PRECEDING AND 1 PRECEDING)
# MAGIC )
# MAGIC SELECT pozo_id, ts, temperatura_motor_c, media_movil,
# MAGIC   (temperatura_motor_c - media_movil) / NULLIF(desvio_movil, 0) AS z_score
# MAGIC FROM stats
# MAGIC WHERE desvio_movil IS NOT NULL
# MAGIC   AND abs((temperatura_motor_c - media_movil) / NULLIF(desvio_movil, 0)) > 3;
# MAGIC
# MAGIC SELECT COUNT(*) AS alertas_zscore, COUNT(DISTINCT pozo_id) AS pozos FROM anomalias_zscore;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `MATCH_RECOGNIZE` — firmas temporales (lo potente)
# MAGIC
# MAGIC Una lectura alta aislada puede ser ruido. Lo que marca una **falla real** es la *secuencia*:
# MAGIC p. ej., temperatura de motor que **sube de forma sostenida** durante varias horas.
# MAGIC `MATCH_RECOGNIZE` expresa eso directo en SQL — sin self-joins ni ventanas enredadas.
# MAGIC
# MAGIC Patrón: un punto de inicio seguido de **≥6 subidas consecutivas** (`subiendo{6,}`).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE anomalias_patron AS
# MAGIC SELECT 'sobrecalentamiento_esp' AS patron_falla, *
# MAGIC FROM lecturas_serie
# MAGIC MATCH_RECOGNIZE (
# MAGIC   PARTITION BY pozo_id
# MAGIC   ORDER BY ts
# MAGIC   MEASURES
# MAGIC     FIRST(ts) AS ts_inicio,
# MAGIC     LAST(ts)  AS ts_fin,
# MAGIC     FIRST(temperatura_motor_c) AS temp_inicial,
# MAGIC     LAST(temperatura_motor_c)  AS temp_final,
# MAGIC     LAST(temperatura_motor_c) - FIRST(temperatura_motor_c) AS delta_temp,
# MAGIC     COUNT(*) AS horas_subida
# MAGIC   ONE ROW PER MATCH
# MAGIC   AFTER MATCH SKIP PAST LAST ROW
# MAGIC   PATTERN (inicio subiendo{6,})
# MAGIC   DEFINE subiendo AS temperatura_motor_c > PREV(temperatura_motor_c)
# MAGIC );
# MAGIC
# MAGIC SELECT pozo_id, ts_inicio, ts_fin, ROUND(delta_temp,1) AS delta_temp, horas_subida
# MAGIC FROM anomalias_patron ORDER BY delta_temp DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC También podemos buscar la firma de **declinación**: presión de tubing que baja de forma
# MAGIC sostenida durante muchas horas (fuga o agotamiento).

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT pozo_id, ts_inicio, ts_fin, ROUND(caida_psi,0) AS caida_psi, horas
# MAGIC FROM lecturas_serie
# MAGIC MATCH_RECOGNIZE (
# MAGIC   PARTITION BY pozo_id ORDER BY ts
# MAGIC   MEASURES FIRST(ts) AS ts_inicio, LAST(ts) AS ts_fin,
# MAGIC            FIRST(presion_tubing_psi) - LAST(presion_tubing_psi) AS caida_psi,
# MAGIC            COUNT(*) AS horas
# MAGIC   ONE ROW PER MATCH
# MAGIC   AFTER MATCH SKIP PAST LAST ROW
# MAGIC   PATTERN (ini bajando{12,})
# MAGIC   DEFINE bajando AS presion_tubing_psi < PREV(presion_tubing_psi)
# MAGIC )
# MAGIC WHERE caida_psi > 150
# MAGIC ORDER BY caida_psi DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Consolidar y evaluar contra las etiquetas
# MAGIC
# MAGIC Unimos las alertas por regla y las comparamos con `eventos_falla` (ground-truth del notebook 00)
# MAGIC para ver qué tan bien detectan las reglas.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE anomalias_reglas AS
# MAGIC SELECT pozo_id, ts, regla_falla AS modo, 'umbral' AS metodo FROM anomalias_umbral WHERE regla_falla IS NOT NULL
# MAGIC UNION ALL
# MAGIC SELECT pozo_id, ts, 'sobrecalentamiento_esp', 'zscore' FROM anomalias_zscore;
# MAGIC
# MAGIC -- Cobertura: ¿cada evento de falla real fue tocado por alguna alerta de regla?
# MAGIC SELECT e.pozo_id, e.modo_falla, e.ts_inicio, e.ts_fin,
# MAGIC   COUNT(a.ts) AS alertas_dentro_del_evento
# MAGIC FROM eventos_falla e
# MAGIC LEFT JOIN anomalias_reglas a
# MAGIC   ON a.pozo_id = e.pozo_id AND a.ts BETWEEN e.ts_inicio AND e.ts_fin
# MAGIC GROUP BY e.pozo_id, e.modo_falla, e.ts_inicio, e.ts_fin
# MAGIC ORDER BY e.pozo_id;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Bonus — opciones "sin/poco código"
# MAGIC
# MAGIC ### 5a. Anomaly Detection administrado (Lakehouse Monitoring) — *sin código*
# MAGIC No es una función SQL: es un **servicio administrado** que vigila una tabla (frescura,
# MAGIC completitud, % de nulos). Se activa en **Catalog Explorer → esquema → Details → Anomaly
# MAGIC detection**, o por API. Los resultados aparecen con estado (Healthy / Unhealthy) sin que
# MAGIC escribas lógica. Buen complemento al monitoreo de calidad de las tablas de este caso.
# MAGIC
# MAGIC Docs: https://docs.databricks.com/en/lakehouse-monitoring/anomaly-detection.html
# MAGIC
# MAGIC ### 5b. `ai_forecast` — esperado vs real (⚠️ Preview)
# MAGIC Permite pronosticar una serie en SQL y comparar lo **esperado** contra lo **real**; la brecha
# MAGIC es una anomalía. Requiere habilitar el preview *Predictive AI Functions* + SQL Pro/Serverless.
# MAGIC Si no está habilitado en el workspace, mostrar el código y seguir con reglas/ML.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ⚠️ PREVIEW — requiere habilitación. Pronóstico de caudal por pozo (ejemplo).
# MAGIC -- SELECT * FROM ai_forecast(
# MAGIC --   TABLE(SELECT pozo_id, ts, caudal_bruto_m3d FROM lecturas_serie WHERE pozo_id = 'POZO-001'),
# MAGIC --   horizon => 48, time_col => 'ts', value_col => 'caudal_bruto_m3d'
# MAGIC -- );

# COMMAND ----------

# MAGIC %md
# MAGIC ### Puntos de conversación
# MAGIC - Reglas = explicables y baratas; cubren fallas **conocidas**. ML (notebook 04) cubre lo **desconocido**.
# MAGIC - `MATCH_RECOGNIZE` detecta la **secuencia**, no el punto — menos falsos positivos por ruido.
# MAGIC - Todo esto corre en SQL → el equipo de operaciones/analistas lo mantiene sin Python.
