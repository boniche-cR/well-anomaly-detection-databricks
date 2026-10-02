# Databricks notebook source
# MAGIC %md
# MAGIC 🏠 **Índice:** `00_LEEME`   ·   ⬅️ **Anterior:** `04_anomaly_ml`   ·   **Siguiente ➡️:** `06_model_serving`

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "C": {"col": 0, "row": 1, "label": "comentarios\noperativos (ES)"},
        "CL": {"col": 1, "row": 0, "label": "ai_classify\n→ modo de falla"},
        "EX": {"col": 1, "row": 1, "label": "ai_extract\n→ campos"},
        "Q": {"col": 1, "row": 2, "label": "ai_query\n→ resumen"},
        "DE": {"col": 1, "row": 3, "label": "ai_decide\n→ prioridad"},
        "EN": {"col": 2, "row": 0, "label": "comentarios\nenriquecidos"},
    },
    [("C", "CL"), ("C", "EX"), ("C", "Q"), ("C", "DE"), ("CL", "EN"), ("EX", "EN")],
    highlight_ids=["CL", "EX"],
    caption="05 · AI Functions en SQL: texto libre → datos accionables",
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 · Enriquecimiento con IA/LLM (AI Functions en SQL)
# MAGIC
# MAGIC **Capacidad:** *Capacidades de AI/LLM dentro de Databricks para enriquecer información
# MAGIC existente — p. ej. comentarios operativos o eventos de los pozos para generar clasificaciones o etiquetas.*
# MAGIC
# MAGIC Los comentarios del RTOC son **texto libre** que hoy nadie estructura. Con **AI Functions en SQL**
# MAGIC los convertimos en datos accionables — **sin Python, sin entrenar nada**:
# MAGIC
# MAGIC | Función | Estado | Qué hace acá |
# MAGIC |---|---|---|
# MAGIC | `ai_classify` | ✅ GA | Clasifica el comentario en un **modo de falla** |
# MAGIC | `ai_extract` | ✅ GA | Extrae **equipo, severidad, acción recomendada** |
# MAGIC | `ai_query` | ✅ GA | Genera un **resumen** corto para el turno |
# MAGIC | `ai_decide` | 🔵 Beta | Decide **prioridad / requiere intervención** |
# MAGIC | `ai_enrich` | 🔵 Beta | Enriquece con conocimiento externo |
# MAGIC
# MAGIC > Las funciones Beta necesitan habilitación en **Previews**. Si no están activas en el workspace,
# MAGIC > corré las celdas GA y mostrá el código Beta como referencia.

# COMMAND ----------

dbutils.widgets.text("catalog", "oilgas_anomaly_demo", "Catálogo")
dbutils.widgets.text("schema", "pozos", "Esquema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. `ai_classify` — comentario → modo de falla  (✅ GA)
# MAGIC
# MAGIC Pasamos la lista de etiquetas y el modelo devuelve la que mejor aplica (STRING). No entrenamos nada.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE comentarios_clasificados AS
# MAGIC SELECT comentario_id, pozo_id, ts, comentario,
# MAGIC   CAST(
# MAGIC     ai_classify(
# MAGIC       comentario,
# MAGIC       ARRAY('sobrecalentamiento_esp', 'candado_gas', 'arenamiento', 'declinacion_fuga', 'rutina')
# MAGIC     ) AS STRING
# MAGIC   ) AS clasificacion
# MAGIC FROM comentarios_operativos;
# MAGIC
# MAGIC SELECT comentario, clasificacion FROM comentarios_clasificados LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ¿Qué tan bien clasifica? Comparamos la etiqueta de IA contra el modo de falla real del evento
# MAGIC (cuando el comentario cae dentro de una ventana de falla).

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT c.clasificacion AS prediccion_ia, e.modo_falla AS real, COUNT(*) AS n
# MAGIC FROM comentarios_clasificados c
# MAGIC LEFT JOIN eventos_falla e
# MAGIC   ON c.pozo_id = e.pozo_id AND c.ts BETWEEN e.ts_inicio AND e.ts_fin
# MAGIC GROUP BY c.clasificacion, e.modo_falla ORDER BY n DESC;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `ai_extract` — campos estructurados del texto  (✅ GA)
# MAGIC
# MAGIC Sacamos equipo afectado, severidad y acción recomendada en un solo paso.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE comentarios_extraidos AS
# MAGIC SELECT comentario_id, pozo_id, comentario,
# MAGIC   CAST(ai_extract(comentario, ARRAY('equipo_afectado', 'severidad', 'accion_recomendada')) AS STRING) AS extraido_json
# MAGIC FROM comentarios_operativos;
# MAGIC
# MAGIC SELECT comentario, extraido_json FROM comentarios_extraidos LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `ai_query` — resumen para el turno  (✅ GA)
# MAGIC
# MAGIC Un resumen en una línea, en español, por pozo con novedades. `ai_query` llama a un
# MAGIC endpoint de Foundation Model (acá un Llama servido por el sistema).

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT pozo_id,
# MAGIC   ai_query(
# MAGIC     'databricks-meta-llama-3-3-70b-instruct',
# MAGIC     'Resumí en una sola frase, en español, el estado operativo de este pozo a partir de estos comentarios: '
# MAGIC       || array_join(collect_list(comentario), ' | ')
# MAGIC   ) AS resumen_turno
# MAGIC FROM comentarios_operativos
# MAGIC GROUP BY pozo_id
# MAGIC LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. `ai_decide` — prioridad / requiere intervención  (🔵 Beta)
# MAGIC
# MAGIC Decisión multi-criterio sobre cada comentario, en un solo paso. Devuelve prioridad +
# MAGIC si requiere intervención de campo. (Beta — habilitado en el workspace.)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT comentario,
# MAGIC   CAST(
# MAGIC     ai_decide(
# MAGIC       comentario,
# MAGIC       '{
# MAGIC         "prioridad": {"type": "choice", "instructions": "Prioridad operativa del comentario",
# MAGIC           "criteria": {"alta": "riesgo de parada o daño de equipo",
# MAGIC                        "media": "atender dentro del turno", "baja": "rutina sin urgencia"}},
# MAGIC         "requiere_intervencion": {"type": "choice", "instructions": "¿Requiere acción de campo inmediata?",
# MAGIC           "criteria": {"si": "hay que intervenir ya", "no": "puede esperar"}}
# MAGIC       }',
# MAGIC       MAP('version', '1.0')
# MAGIC     ) AS STRING
# MAGIC   ) AS decision
# MAGIC FROM comentarios_operativos LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC Fallback GA equivalente (si el preview no estuviera disponible): prioridad con `ai_classify`.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ✅ Fallback GA
# MAGIC SELECT comentario,
# MAGIC   CAST(ai_classify(comentario, ARRAY('alta', 'media', 'baja')) AS STRING) AS prioridad
# MAGIC FROM comentarios_operativos LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. `ai_enrich` — enriquecer con conocimiento externo  (🔵 Beta)
# MAGIC
# MAGIC Trae datos que no están en nuestras tablas. Caso real: specs de un equipo a partir de su
# MAGIC modelo (fabricante, rango de caudal, aplicación). (Beta — habilitado en el workspace.)

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT CAST(
# MAGIC   ai_enrich(
# MAGIC     'Bomba electrosumergible (ESP) modelo REDA Maximus',
# MAGIC     '["fabricante", "rango_caudal_bpd", "aplicacion_tipica"]',
# MAGIC     PARSE_JSON('[{"type": "web_search", "config": {}}]')
# MAGIC   ) AS STRING
# MAGIC ) AS specs_equipo;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Tabla enriquecida final
# MAGIC
# MAGIC Unimos clasificación + extracción → una tabla de comentarios lista para operar y para
# MAGIC alimentar el feedback (notebook 07) y los dashboards/Genie.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE comentarios_enriquecidos AS
# MAGIC SELECT c.comentario_id, c.pozo_id, c.ts, c.comentario,
# MAGIC        c.clasificacion   AS modo_falla_ia,
# MAGIC        e.extraido_json   AS campos_extraidos_ia
# MAGIC FROM comentarios_clasificados c
# MAGIC LEFT JOIN comentarios_extraidos e USING (comentario_id);
# MAGIC
# MAGIC SELECT * FROM comentarios_enriquecidos LIMIT 10;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Puntos de conversación
# MAGIC - **Cero MLOps para arrancar:** `ai_classify`/`ai_extract` corren en SQL contra modelos ya servidos.
# MAGIC - **Gobernado:** el texto nunca sale del perímetro de Databricks; aplica el mismo UC del notebook 02.
# MAGIC - **Combina con señales:** cruzamos lo que **dice el operador** (texto) con lo que **miden los sensores** (ML).
# MAGIC - **Evolución:** las funciones Beta (`ai_decide`, `ai_enrich`) suman decisión y conocimiento externo.
