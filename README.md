# Well Anomaly Detection — Databricks Solution Accelerator

End-to-end reference for detecting anomalies in **oil & gas well** production and operations
data on the Databricks Lakehouse — combining **analytical rules + machine learning + generative AI**,
all governed by **Unity Catalog**. Packaged as a **Databricks Asset Bundle (DAB)**.

> Notebooks and narrative are in **Spanish**. Runs on **synthetic data** — no customer data required.

## Architecture

```
 Sources            Lakehouse · Unity Catalog                         Consume
 ┌─────────┐    ┌───────────────────────────────────────┐    ┌──────────────────┐
 │ sensors │──▶ │ 01 ingest (Auto Loader)                │    │ 06 Model Serving │
 │ (SCADA) │    │        │                                │──▶ │  REST + ai_query │
 │ op. text│─┐  │  03 rules + MATCH_RECOGNIZE             │    └────────┬─────────┘
 └─────────┘ │  │  04 ML (Isolation Forest → MLflow → UC) │             │
             └▶ │  05 AI Functions (ai_classify/extract)  │        alerts ▼
                └───────────────────┬────────────────────┘    ┌──────────────────┐
                     02 governance  │   feedback (07)          │ 07 feedback loop │
                                    └──────────────◀───────────┴──────────────────┘
```

Each notebook renders its own architecture diagram (inline SVG) and links to the next step.

## What's inside

| # | Notebook | Capability |
|---|----------|-----------|
| 00 | `00_LEEME` | Index: overview, master diagram, run order |
| 00 | `00_setup` | UC catalog/schema/Volume + synthetic 20-well dataset with injected failure modes |
| 01 | `01_ingestion_incremental` | Incremental ingestion (Auto Loader → bronze → silver) |
| 02 | `02_unity_catalog_governance` | Tags, grants, column masking, row filters, lineage |
| 03 | `03_anomaly_rules` | Thresholds + rolling z-score + `MATCH_RECOGNIZE` (sequential patterns) |
| 04 | `04_anomaly_ml` | Isolation Forest → MLflow → Unity Catalog Model Registry |
| 05 | `05_genai_enrichment` | SQL AI Functions: `ai_classify` / `ai_extract` / `ai_query` / `ai_decide` / `ai_enrich` |
| 06 | `06_model_serving` | Model Serving endpoint, consumed via Python client and `ai_query` in SQL |
| 07 | `07_feedback_loop` | User feedback → precision tracking → supervised dataset / few-shot |
| 08 | `08_reference_architecture` | Reference architecture + production best practices |

## Quickstart

Deploy and run as a bundle:

```bash
databricks bundle deploy -t dev -p <profile>
databricks bundle run generar_datos    -t dev -p <profile>   # seed synthetic data (notebook 00)
databricks bundle run run_all           -t dev -p <profile>   # run 00 → 08 in order
```

Or open `src/notebooks/00_LEEME` in the workspace and work through `00 → 08` interactively.

## Prerequisites

- Unity Catalog + serverless compute.
- Access to **Foundation Model APIs** (for the AI Functions in notebook 05/06).
- `ai_decide` and `ai_enrich` are **Beta** — enable them on the workspace **Previews** page
  (notebook 05 includes GA fallbacks with `ai_classify` if they are not available).

## Data

Synthetic and reproducible: 20 wells across two example fields, hourly sensor series with four
injected failure signatures (ESP overheating, gas lock, sand production, decline) plus Spanish
operational comments for the GenAI leg.

## Disclaimer

Provided **AS-IS**, without warranty. This is a community accelerator and **not an official
Databricks product**. Review and harden before any production use.
