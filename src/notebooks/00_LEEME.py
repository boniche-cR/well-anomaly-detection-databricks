# Databricks notebook source
# MAGIC %md
# MAGIC # 🛢️ Detección de Anomalías en Pozos — Solution Accelerator
# MAGIC ### Databricks · Oil & Gas Upstream
# MAGIC
# MAGIC Una solución **end-to-end** para detectar anomalías en datos de producción/operación de pozos,
# MAGIC combinando **reglas analíticas + Machine Learning + IA generativa**, toda gobernada por Unity Catalog.
# MAGIC
# MAGIC > **Empezá aquí.** Este notebook es el índice. Corré los notebooks en orden (`00` → `08`) o navegá
# MAGIC > con los enlaces **Siguiente ➡️** al pie de cada uno.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Arquitectura de la solución

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "S": {"col": 0, "row": 0, "label": "SCADA / PI\nsensores"},
        "C": {"col": 0, "row": 2, "label": "Comentarios\noperativos"},
        "I": {"col": 1, "row": 0, "label": "01 · Ingesta\nincremental"},
        "R": {"col": 2, "row": 0, "label": "03 · Reglas +\nMATCH_RECOGNIZE"},
        "M": {"col": 2, "row": 1, "label": "04 · ML\nIsolation Forest"},
        "AI": {"col": 2, "row": 2, "label": "05 · AI\nFunctions"},
        "SV": {"col": 3, "row": 1, "label": "06 · Model\nServing"},
        "AL": {"col": 4, "row": 1, "label": "Alertas de\nanomalía"},
        "FB": {"col": 5, "row": 1, "label": "07 · Feedback\ndel operador"},
    },
    [("S", "I"), ("C", "AI"), ("I", "R"), ("I", "M"), ("I", "AI"), ("M", "SV"),
     ("R", "AL"), ("M", "AL"), ("AI", "AL"), ("SV", "AL"), ("AL", "FB"),
     ("FB", "M", "reentrena", True), ("FB", "AI", "corrige", True)],
    caption="Gobierno (02 · Unity Catalog) es transversal · buenas prácticas en el notebook 08",
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Contenido — orden de ejecución
# MAGIC
# MAGIC | # | Notebook | Qué muestra | Capacidad |
# MAGIC |---|---|---|---|
# MAGIC | 00 | `00_setup` — **Setup y datos** | Catálogo/esquema/Volume + datos sintéticos de 20 pozos con fallas inyectadas | Datos de ejemplo |
# MAGIC | 01 | `01_ingestion_incremental` — **Ingesta incremental** | Auto Loader → bronze → silver → serie histórica | Ingesta incremental |
# MAGIC | 02 | `02_unity_catalog_governance` — **Gobierno** | Tags, grants, column mask, row filter, linaje | Gobierno de datos |
# MAGIC | 03 | `03_anomaly_rules` — **Reglas analíticas** | Umbrales + z-score + `MATCH_RECOGNIZE` | Modelos de detección |
# MAGIC | 04 | `04_anomaly_ml` — **ML Isolation Forest** | Modelo no supervisado + MLflow + registro en UC | Modelos de detección |
# MAGIC | 05 | `05_genai_enrichment` — **Enriquecimiento con IA** | `ai_classify` / `ai_extract` / `ai_query` / `ai_decide` / `ai_enrich` | AI/LLM para enriquecer |
# MAGIC | 06 | `06_model_serving` — **Model Serving** | Endpoint REST + consumo vía `ai_query` | Serving + AI Functions |
# MAGIC | 07 | `07_feedback_loop` — **Feedback loop** | Feedback del operador → mejora continua | Feedback de usuario |
# MAGIC | 08 | `08_reference_architecture` — **Arquitectura de referencia** | Diagrama + buenas prácticas a producción | Arquitectura y buenas prácticas |
# MAGIC
# MAGIC *(Databricks no abre notebooks por links relativos en Markdown — abrí cada uno desde el árbol de archivos, en orden.)*

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerrequisitos
# MAGIC
# MAGIC - Workspace con **Unity Catalog**, **cómputo serverless** y acceso a **Foundation Model APIs** (para AI Functions).
# MAGIC - Permiso para crear catálogo/esquema/Volume y un endpoint de Model Serving.
# MAGIC - Las funciones **Beta** `ai_decide` / `ai_enrich` requieren habilitación en la página **Previews**
# MAGIC   (si no están, el notebook 05 trae fallbacks GA con `ai_classify`).
# MAGIC
# MAGIC ## Cómo correrlo
# MAGIC
# MAGIC **Opción A — interactivo (recomendado para el workshop):** abrí `00_setup` y seguí los enlaces *Siguiente ➡️*.
# MAGIC
# MAGIC **Opción B — como jobs (DAB):**
# MAGIC ```bash
# MAGIC databricks bundle deploy -t dev -p <profile>
# MAGIC databricks bundle run generar_datos    -t dev -p <profile>   # siembra los datos (notebook 00)
# MAGIC databricks bundle run pipeline_anomalias -t dev -p <profile>  # ingesta → reglas → scoring ML
# MAGIC ```
# MAGIC
# MAGIC ## Datos
# MAGIC
# MAGIC Sintéticos y reproducibles: 20 pozos en **dos yacimientos de ejemplo**, lecturas horarias
# MAGIC de sensores con **4 modos de falla** inyectados (sobrecalentamiento ESP, candado de gas, arenamiento,
# MAGIC declinación) + comentarios operativos en español. No depende de datos de ningún cliente.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### ▶️ Empezar: abrí **`00_setup`** en el árbol de archivos (y seguí en orden 00 → 08)
# MAGIC
# MAGIC <sub>Solution Accelerator · se apoya en el acelerador open-source de Databricks
# MAGIC `databricks-industry-solutions/energy-sandbox` (ESP Predictive Maintenance). Entregado AS-IS.</sub>
