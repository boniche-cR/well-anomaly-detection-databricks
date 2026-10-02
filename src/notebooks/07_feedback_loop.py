# Databricks notebook source
# MAGIC %md
# MAGIC 🏠 **Índice:** `00_LEEME`   ·   ⬅️ **Anterior:** `06_model_serving`   ·   **Siguiente ➡️:** `08_reference_architecture`

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "AL": {"col": 0, "row": 0, "label": "Alertas\n(reglas/ML/IA)"},
        "OP": {"col": 1, "row": 0, "label": "Operador\nconfirma/corrige"},
        "FB": {"col": 2, "row": 0, "label": "feedback_operador"},
        "PR": {"col": 3, "row": 0, "label": "Precisión real"},
        "DS": {"col": 3, "row": 1, "label": "dataset\nsupervisado"},
        "M": {"col": 4, "row": 1, "label": "Próximo\nmodelo"},
    },
    [("AL", "OP"), ("OP", "FB"), ("FB", "PR"), ("FB", "DS"), ("DS", "M", "reentrena", True)],
    highlight_ids=["FB"],
    caption="07 · El feedback del operador alimenta la mejora continua",
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 07 · Feedback de usuario para mejorar la calidad
# MAGIC
# MAGIC **Capacidad:** *Incorporar mecanismos de feedback de usuario para mejorar la
# MAGIC calidad de las clasificaciones y predicciones.*
# MAGIC
# MAGIC El operador del RTOC es quien sabe si una alerta fue real. Capturamos ese juicio y lo usamos para:
# MAGIC
# MAGIC 1. **Medir** la precisión real del sistema (sobre lo que el humano confirmó).
# MAGIC 2. **Reetiquetar** → construir un set supervisado para el próximo modelo.
# MAGIC 3. **Mejorar la clasificación por IA** (few-shot: ejemplos corregidos como instrucción).
# MAGIC
# MAGIC Así el sistema **aprende del uso** en vez de quedarse estático.

# COMMAND ----------

dbutils.widgets.text("catalog", "oilgas_anomaly_demo", "Catálogo")
dbutils.widgets.text("schema", "pozos", "Esquema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tabla de feedback
# MAGIC
# MAGIC Una fila por alerta revisada por un operador: ¿era anomalía real? ¿qué modo de falla? El
# MAGIC frontend (app/dashboard) escribe acá. En producción, con un botón "confirmar / descartar".

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS feedback_operador (
# MAGIC   feedback_id   BIGINT GENERATED ALWAYS AS IDENTITY,
# MAGIC   pozo_id       STRING,
# MAGIC   ts            TIMESTAMP,
# MAGIC   fuente        STRING,     -- 'ml' | 'reglas' | 'ia_texto'
# MAGIC   anomalia_real BOOLEAN,    -- juicio del operador
# MAGIC   modo_correcto STRING,     -- modo de falla corregido por el operador
# MAGIC   operador      STRING,
# MAGIC   revisado_ts   TIMESTAMP DEFAULT current_timestamp()
# MAGIC )
# MAGIC TBLPROPERTIES ('delta.feature.allowColumnDefaults' = 'supported')
# MAGIC COMMENT 'Feedback humano sobre alertas de anomalía (RTOC).';

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Simular feedback del RTOC
# MAGIC
# MAGIC Para la demo, generamos feedback coherente con las etiquetas reales: lo que cayó dentro de un
# MAGIC evento de falla se confirma; una fracción de las alertas fuera de evento se marca como falso positivo.

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT INTO feedback_operador (pozo_id, ts, fuente, anomalia_real, modo_correcto, operador)
# MAGIC SELECT a.pozo_id, a.ts, 'ml',
# MAGIC   e.modo_falla IS NOT NULL                                   AS anomalia_real,
# MAGIC   COALESCE(e.modo_falla, 'falso_positivo')                   AS modo_correcto,
# MAGIC   'op_' || CAST(1 + (abs(hash(a.pozo_id)) % 8) AS STRING)    AS operador
# MAGIC FROM anomalias_ml a
# MAGIC LEFT JOIN eventos_falla e
# MAGIC   ON a.pozo_id = e.pozo_id AND a.ts BETWEEN e.ts_inicio AND e.ts_fin
# MAGIC WHERE a.es_anomalia = 1
# MAGIC   AND (e.modo_falla IS NOT NULL OR rand() < 0.3);   -- muestreo de revisiones
# MAGIC
# MAGIC SELECT fuente, anomalia_real, COUNT(*) FROM feedback_operador GROUP BY fuente, anomalia_real;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Medir precisión real (sobre lo revisado)
# MAGIC
# MAGIC De las alertas que el operador revisó, ¿qué proporción eran anomalías reales? Esta métrica
# MAGIC se puede seguir en el tiempo para detectar degradación del modelo.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT fuente,
# MAGIC   COUNT(*)                                             AS revisadas,
# MAGIC   SUM(CASE WHEN anomalia_real THEN 1 ELSE 0 END)       AS verdaderas,
# MAGIC   ROUND(AVG(CASE WHEN anomalia_real THEN 1.0 ELSE 0 END), 3) AS precision
# MAGIC FROM feedback_operador GROUP BY fuente;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Reetiquetar → set supervisado para el próximo modelo
# MAGIC
# MAGIC El feedback confirmado se convierte en **etiquetas**. Con suficientes, el próximo modelo pasa
# MAGIC de no supervisado a **supervisado** (clasificador por modo de falla) — más preciso y accionable.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE dataset_supervisado AS
# MAGIC SELECT s.*, f.modo_correcto AS etiqueta, f.anomalia_real
# MAGIC FROM feedback_operador f
# MAGIC JOIN lecturas_serie s USING (pozo_id, ts);
# MAGIC
# MAGIC SELECT etiqueta, COUNT(*) AS ejemplos FROM dataset_supervisado GROUP BY etiqueta ORDER BY ejemplos DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Mejorar `ai_classify` con few-shot (sin reentrenar)
# MAGIC
# MAGIC Los casos que la IA erró y el operador corrigió se inyectan como **instrucciones/ejemplos**.
# MAGIC Mejora inmediata de la clasificación de texto sin tocar infraestructura.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Ejemplos corregidos (comentario → modo real) para guiar al clasificador
# MAGIC SELECT c.comentario, f.modo_correcto AS etiqueta_correcta
# MAGIC FROM feedback_operador f
# MAGIC JOIN comentarios_operativos c ON c.pozo_id = f.pozo_id
# MAGIC WHERE f.anomalia_real = true
# MAGIC LIMIT 5;
# MAGIC -- Estos pares se pasan en el parámetro 'instructions' de ai_classify (opción avanzada),
# MAGIC -- o se usan para fine-tuning del endpoint si el volumen lo justifica.

# COMMAND ----------

# MAGIC %md
# MAGIC ### El ciclo completo
# MAGIC ```
# MAGIC sensores + texto → detección (reglas/ML/IA) → alerta → operador confirma/corrige
# MAGIC         ↑                                                           │
# MAGIC         └──────────  reentrenar / few-shot / recalibrar  ◄──────────┘
# MAGIC ```
# MAGIC
# MAGIC ### Puntos de conversación
# MAGIC - El feedback es **dato gobernado** en UC — con linaje hacia el modelo que lo generó.
# MAGIC - Se puede automatizar: job que reentrena cuando se acumulan N correcciones o cae la precisión.
# MAGIC - **Inference Tables** del endpoint (notebook 06) + feedback = base de monitoreo de drift.
