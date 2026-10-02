# Databricks notebook source
# MAGIC %md
# MAGIC 🏠 **Índice:** `00_LEEME`   ·   ⬅️ **Anterior:** `07_feedback_loop`   ·   🏁 fin del recorrido

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "F": {"col": 0, "row": 0, "label": "Fuentes\nsensores + texto"},
        "I": {"col": 1, "row": 0, "label": "Ingesta\nincremental"},
        "UC": {"col": 2, "row": 0, "label": "Lakehouse · UC\nbronze→silver→gold"},
        "DET": {"col": 3, "row": 0, "label": "Detección\nreglas + ML + IA"},
        "SV": {"col": 4, "row": 0, "label": "Model Serving\nREST + SQL"},
        "CON": {"col": 5, "row": 0, "label": "AI/BI · Genie\nApp (Fase 2)"},
    },
    [("F", "I"), ("I", "UC"), ("UC", "DET"), ("DET", "SV"), ("SV", "CON"),
     ("CON", "DET", "feedback", True)],
    caption="08 · Arquitectura de referencia end-to-end",
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 08 · Arquitectura de referencia y buenas prácticas
# MAGIC
# MAGIC **Capacidad:** *Arquitectura de referencia y buenas prácticas para llevar una
# MAGIC solución de este tipo a producción.*
# MAGIC
# MAGIC Este notebook cierra el workshop: cómo encajan todas las piezas y qué hace falta para producción.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Arquitectura de referencia
# MAGIC
# MAGIC ```
# MAGIC   FUENTES                INGESTA              LAKEHOUSE (Unity Catalog)           CONSUMO
# MAGIC  ┌──────────┐        ┌──────────────┐    ┌─────────────────────────────┐   ┌───────────────┐
# MAGIC  │ SCADA/PI │─────┐  │ Auto Loader  │    │ bronze → silver → gold      │   │ Model Serving │
# MAGIC  │ Historian│     ├─▶│ (incremental)│───▶│ (Delta + linaje + permisos) │──▶│  (REST + SQL) │
# MAGIC  │ sensores │     │  │ Lakeflow     │    │                             │   └──────┬────────┘
# MAGIC  └──────────┘     │  └──────────────┘    │  Detección:                 │          │
# MAGIC  ┌──────────┐     │                      │   • reglas + MATCH_RECOGNIZE│          ▼
# MAGIC  │ coment.  │─────┘                      │   • ML (Isolation Forest)   │   ┌───────────────┐
# MAGIC  │ operativos│                           │   • IA texto (AI Functions) │   │ AI/BI + Genie │
# MAGIC  └──────────┘                            │  Modelos registrados en UC  │   │ Alertas       │
# MAGIC                                          └──────────────┬──────────────┘   │ App (Fase 2)  │
# MAGIC                                                         │                  └──────┬────────┘
# MAGIC                                                         │   feedback operador     │
# MAGIC                                                         └─────────◄───────────────┘
# MAGIC ```
# MAGIC
# MAGIC Un solo plano gobernado: los datos, los modelos y las funciones de IA viven bajo **Unity Catalog**.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Mapa: capacidad → componente → notebook
# MAGIC
# MAGIC | # | Capacidad | Componente Databricks | Notebook |
# MAGIC |---|---|---|---|
# MAGIC | 1 | Ingesta/procesamiento incremental | Auto Loader, Streaming Tables, Delta, `MERGE` | `01` |
# MAGIC | 2 | Gobierno de datos | Unity Catalog: tags, grants, masking, row filter, linaje | `02` |
# MAGIC | 3 | Modelos de detección | Reglas + `MATCH_RECOGNIZE`; Isolation Forest + MLflow + UC | `03`, `04` |
# MAGIC | 4 | IA/LLM para enriquecer | AI Functions: `ai_classify`/`ai_extract`/`ai_query` | `05` |
# MAGIC | 5 | Model Serving + AI Functions | Serving endpoint; `ai_query` sobre el endpoint | `06` |
# MAGIC | 6 | Feedback de usuario | Tabla de feedback → reentrenar / few-shot | `07` |
# MAGIC | 7 | Arquitectura + buenas prácticas | este notebook | `08` |

# COMMAND ----------

# MAGIC %md
# MAGIC ## Buenas prácticas para producción
# MAGIC
# MAGIC **Datos e ingesta**
# MAGIC - Auto Loader con checkpoint + schema evolution; `MERGE` para idempotencia.
# MAGIC - Expectativas de calidad (Lakeflow Declarative Pipelines / DLT) sobre bronze→silver.
# MAGIC - Particionado/líquid clustering por `pozo_id` + fecha para series largas.
# MAGIC
# MAGIC **Gobierno**
# MAGIC - Grupos por rol (RTOC, Ingeniería, DS) en vez de permisos individuales.
# MAGIC - Máscaras/row-filters definidos una vez; aplican a BI, SQL y IA por igual.
# MAGIC - Tags de sensibilidad + linaje para auditoría.
# MAGIC
# MAGIC **Modelos**
# MAGIC - Registrar en UC con alias (`champion`/`challenger`); promover por métrica.
# MAGIC - **Inference Tables** en el endpoint → monitoreo de drift y volumen.
# MAGIC - Lakehouse Monitoring (anomaly detection administrado) sobre tablas clave.
# MAGIC - Reentrenamiento disparado por feedback acumulado o caída de precisión.
# MAGIC
# MAGIC **Operación**
# MAGIC - Serverless + `scale_to_zero` para costo variable.
# MAGIC - Jobs orquestando ingesta→reglas→scoring (ver `pipeline_anomalias` en el DAB).
# MAGIC - Todo como código en un **DAB** → promovible dev→prod con un comando.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Esto ya corre en upstream — prueba
# MAGIC
# MAGIC - **NOV** procesa **>1.000 millones de filas/día** de **30.000+ sensores** en un lakehouse para
# MAGIC   mantenimiento predictivo de equipos de perforación (medallion + Unity Catalog + ML).
# MAGIC   *(AWS Industries blog, 2024.)*
# MAGIC - **Databricks Industry Solutions** publica un acelerador de Energía open-source
# MAGIC   (`energy-sandbox`, incl. `esp-pm` para mantenimiento predictivo de ESP) desplegable como DAB.
# MAGIC
# MAGIC > El patrón que construimos hoy es el mismo que usan operadores de primer nivel; lo adaptamos a pozos.

# COMMAND ----------

# MAGIC %md
# MAGIC ## La evolución: una Databricks App
# MAGIC
# MAGIC Hoy mostramos el motor (datos → detección → serving → feedback). El siguiente paso natural es
# MAGIC una **Databricks App** encima:
# MAGIC
# MAGIC - Vista de **flota**: estado de todos los pozos, ordenados por riesgo.
# MAGIC - **Detalle por pozo**: series, alertas de reglas/ML, comentarios enriquecidos por IA.
# MAGIC - Botón **confirmar/descartar** que escribe en `feedback_operador` (notebook 07).
# MAGIC - Chat **Genie** sobre `lecturas_serie` para preguntar en lenguaje natural.
# MAGIC
# MAGIC Referencia directa: la app `esp-pm` del acelerador `energy-sandbox` (fleet view + diagnóstico
# MAGIC por pozo + Genie Advisor). Se puede partir de ahí.
# MAGIC
# MAGIC ### Próximos pasos sugeridos
# MAGIC 1. Conectar una fuente real (SCADA/PI Historian) a `01` con Auto Loader/Lakeflow.
# MAGIC 2. Elegir 1–2 modos de falla prioritarios y afinar reglas + modelo con datos reales.
# MAGIC 3. Pilotear la App de flota sobre un grupo de pozos del RTOC.
