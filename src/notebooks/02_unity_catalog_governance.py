# Databricks notebook source
# MAGIC %md
# MAGIC 🏠 **Índice:** `00_LEEME`   ·   ⬅️ **Anterior:** `01_ingestion_incremental`   ·   **Siguiente ➡️:** `03_anomaly_rules`

# COMMAND ----------

# MAGIC %run ./_viz_helper

# COMMAND ----------

displayHTML(draw_graph(
    {
        "UC": {"col": 0, "row": 2, "label": "Unity\nCatalog"},
        "T": {"col": 1, "row": 0, "label": "Tags +\ncomentarios"},
        "G": {"col": 1, "row": 1, "label": "GRANTs\npor rol"},
        "M": {"col": 1, "row": 2, "label": "Column mask\n(ubicación)"},
        "R": {"col": 1, "row": 3, "label": "Row filter\n(campo)"},
        "L": {"col": 1, "row": 4, "label": "Linaje\nautomático"},
    },
    [("UC", "T"), ("UC", "G"), ("UC", "M"), ("UC", "R"), ("UC", "L")],
    highlight_ids=["UC"],
    caption="02 · Un plano de gobierno sobre todos los datos (y modelos)",
))

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 · Gobierno de datos con Unity Catalog
# MAGIC
# MAGIC **Capacidad:** *Gestión y gobierno de datos mediante Unity Catalog.*
# MAGIC
# MAGIC Un solo plano de gobierno sobre todos los datos del caso:
# MAGIC
# MAGIC 1. **Comentarios y tags** (clasificación de datos y descubrimiento).
# MAGIC 2. **Permisos** (GRANT) por catálogo/esquema/tabla.
# MAGIC 3. **Column masking** — enmascarar un dato sensible según el grupo del usuario.
# MAGIC 4. **Row filtering** — limitar filas por campo/activo según el usuario.
# MAGIC 5. **Linaje** — se captura automático; dónde verlo.

# COMMAND ----------

dbutils.widgets.text("catalog", "oilgas_anomaly_demo", "Catálogo")
dbutils.widgets.text("schema", "pozos", "Esquema")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
spark.sql(f"USE {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Comentarios y tags
# MAGIC
# MAGIC Documentar y clasificar es parte del gobierno. Los tags son buscables en el Catalog Explorer
# MAGIC y sirven para políticas (p. ej. todo lo marcado `sensibilidad=alta`).

# COMMAND ----------

# MAGIC %sql
# MAGIC COMMENT ON TABLE dim_pozo IS 'Dimensión de pozos. Una fila por pozo.';
# MAGIC COMMENT ON TABLE silver_lecturas IS 'Lecturas de sensores tipadas y enriquecidas (silver).';
# MAGIC
# MAGIC ALTER TABLE dim_pozo         SET TAGS ('dominio' = 'upstream', 'capa' = 'dimension');
# MAGIC ALTER TABLE silver_lecturas  SET TAGS ('dominio' = 'upstream', 'capa' = 'silver');
# MAGIC
# MAGIC -- Marcar columnas sensibles (ubicación del activo)
# MAGIC ALTER TABLE dim_pozo ALTER COLUMN latitud  SET TAGS ('sensibilidad' = 'alta');
# MAGIC ALTER TABLE dim_pozo ALTER COLUMN longitud SET TAGS ('sensibilidad' = 'alta');

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Permisos (GRANT)
# MAGIC
# MAGIC Modelo simple: lectura para todos los usuarios de la cuenta (grupo `account users`, que
# MAGIC siempre existe). En un caso real, serían grupos por rol (RTOC, Ingeniería, Data Science).

# COMMAND ----------

# MAGIC %sql
# MAGIC GRANT USE CATALOG ON CATALOG ${catalog} TO `account users`;
# MAGIC GRANT USE SCHEMA  ON SCHEMA  ${catalog}.${schema} TO `account users`;
# MAGIC GRANT SELECT      ON TABLE   ${catalog}.${schema}.silver_lecturas TO `account users`;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Column masking
# MAGIC
# MAGIC Enmascaramos la ubicación exacta del pozo salvo para un grupo autorizado. La función de
# MAGIC máscara se aplica a nivel de columna y es transparente para las consultas.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Devuelve el valor solo si el usuario pertenece al grupo; si no, NULL.
# MAGIC CREATE OR REPLACE FUNCTION mask_ubicacion(valor DOUBLE)
# MAGIC RETURN CASE WHEN is_account_group_member('rtoc_geo') THEN valor ELSE NULL END;
# MAGIC
# MAGIC ALTER TABLE dim_pozo ALTER COLUMN latitud  SET MASK mask_ubicacion;
# MAGIC ALTER TABLE dim_pozo ALTER COLUMN longitud SET MASK mask_ubicacion;
# MAGIC
# MAGIC -- Al no ser miembro de 'rtoc_geo', latitud/longitud salen en NULL:
# MAGIC SELECT pozo_id, campo, latitud, longitud FROM dim_pozo LIMIT 5;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Row filtering
# MAGIC
# MAGIC Limitar qué filas ve cada quien. Ejemplo: un usuario del "Yacimiento Sur" solo ve
# MAGIC sus pozos. Acá lo dejamos permisivo para la demo (todos ven todo si son del grupo RTOC).

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION filtro_campo(campo STRING)
# MAGIC RETURN is_account_group_member('rtoc') OR campo = 'Yacimiento Norte';
# MAGIC
# MAGIC ALTER TABLE dim_pozo SET ROW FILTER filtro_campo ON (campo);
# MAGIC
# MAGIC -- Verás solo Yacimiento Norte salvo que seas del grupo 'rtoc'
# MAGIC SELECT campo, COUNT(*) AS pozos FROM dim_pozo GROUP BY campo;

# COMMAND ----------

# MAGIC %md
# MAGIC ### Limpieza opcional de políticas (para repetir la demo)
# MAGIC Descomentar si querés dejar la tabla sin máscara/filtro.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ALTER TABLE dim_pozo ALTER COLUMN latitud  DROP MASK;
# MAGIC -- ALTER TABLE dim_pozo ALTER COLUMN longitud DROP MASK;
# MAGIC -- ALTER TABLE dim_pozo DROP ROW FILTER;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Linaje
# MAGIC
# MAGIC Unity Catalog captura el **linaje** automáticamente al correr los notebooks `01`+:
# MAGIC `landing (Volume) → bronze_lecturas → silver_lecturas → lecturas_serie → (reglas / ML / IA)`.
# MAGIC
# MAGIC Dónde verlo (en vivo): **Catalog Explorer → tabla → pestaña *Lineage*** (vista de upstream/downstream
# MAGIC + columnas). No requiere configuración.
# MAGIC
# MAGIC ### Puntos de conversación
# MAGIC - Un solo modelo de permisos para datos **y** modelos (los modelos ML también viven en UC).
# MAGIC - Las políticas (máscara/filtro) se definen **una vez** y aplican a toda consulta, BI o IA.
# MAGIC - Tags + comentarios → descubrimiento y cumplimiento (encontrar todo lo `sensibilidad=alta`).
