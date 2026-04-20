# Paperless-ngx ML — System Implementation

Two complementary machine-learning features integrated into [Paperless-ngx](https://github.com/paperless-ngx/paperless-ngx), an open-source document management system, deployed end-to-end on Chameleon Cloud for a hypothetical 30-staff academic department.

- **Handwritten Text Recognition (HTR)** — TrOCR transcribes handwritten regions at upload time. Transcriptions merge with Tesseract OCR so handwritten content is indexed alongside typed text. Low-confidence regions surface to the user via a feedback UI; corrections flow back as labeled training data.
- **Semantic Search** — an mpnet bi-encoder (768-dim) encodes documents and queries into a shared embedding space; results from a Qdrant vector index are merged with Paperless's keyword index and ranked for the user.

---

## Repositories

All four repositories are public.

| Repo | Role | What it owns |
|---|---|---|
| [Palomarr/paperless-ml](https://github.com/Palomarr/paperless-ml) (this repo) | Serving + Integration | `ml-gateway` FastAPI + ONNX Runtime inference, `ml_hooks` Django overlay, Prometheus/Alertmanager/Grafana observability, `rollback-ctrl`, `pipeline-scheduler`, one-command Chameleon bring-up |
| [REDES01/paperless_data](https://github.com/REDES01/paperless_data) | Data — Elnath | IAM + SQuAD ingestion, postgres schema, batch training-set compilation with quality checks, synthetic traffic generator |
| [REDES01/paperless_data_integration](https://github.com/REDES01/paperless_data_integration) | Data — Elnath | HTR consumer (Kafka → region slicer → ml-gateway → postgres), live drift monitor |
| [gdtmax/paperless_training_integration](https://github.com/gdtmax/paperless_training_integration) | Training — Dongting | Training pipeline (train / eval / quality gate), ONNX export, MLflow integration |

---

## Quick start (Chameleon CHI@TACC)

```bash
# 1. Provision a `gpu_p100` bare-metal node

# 2. SSH into the node
ssh -i ~/.ssh/id_rsa_chameleon cc@<floating-ip>

# 3. One command brings up the full 18-service stack
cd ~/paperless-ml
bash scripts/chameleon_setup.sh
```

The script installs host prerequisites, clones the three peer repos as siblings, brings up the compose stack, waits for services to be healthy, extracts the Paperless admin API token, and prints service URLs.

Takes **~8–12 minutes** on a cold node.

**After the script prints its summary**, URLs to open from your laptop:

| Service | URL | Auth |
|---|---|---|
| Paperless web + Feedback UI | `http://<ip>:8000` + `/ml-ui/` | admin / admin |
| Grafana dashboards | `http://<ip>:3000` | admin / admin |
| Prometheus alerts | `http://<ip>:9090/alerts` | — |
| Alertmanager | `http://<ip>:9093` | — |
| Qdrant dashboard | `http://<ip>:6333/dashboard` | — |
| MinIO console | `http://<ip>:9001` | minioadmin / minioadmin |
| MLflow UI | `http://<ip>:5050` | — |

Full deployment + troubleshooting guide: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Architecture

Eight tiers, one compose file:

- **User-facing** — Paperless-ngx + `ml_hooks` Django overlay. The overlay is bind-mounted into the Paperless container so no fork is needed; it adds the `MlGlobalSearchView`, the `Feedback` model, the feedback UI at `/ml-ui/`, and the four Redpanda event publishers.
- **ML serving** — `ml-gateway` (FastAPI + ONNX Runtime) exposes `/htr`, `/search/encode`, `/search/query`, `/metrics`, `/health`. Models pulled from MinIO on boot.
- **Storage** — Postgres (Paperless's DB + Feedback table), Redis (Celery broker), Qdrant (768-dim vector index), MinIO (document images + model artifacts + training-data warehouse), Redpanda (Kafka-compatible event stream for uploads / corrections / queries / feedback).
- **Monitoring** — Prometheus scrapes six targets and loads eight alert rules across two groups. Grafana auto-provisions a nine-panel dashboard. Alertmanager routes alerts to `rollback-ctrl`.
- **Training** — MLflow tracking server with Postgres backend and MinIO artifact store. Dongting's pipeline logs all runs and registers models conditionally via the quality gate.
- **Automated pipeline** — `pipeline-scheduler` auto-triggers retraining when two compound conditions hold (≥500 new corrections AND ≥24h since last run). On gate pass, the scheduler promotes the new version via MLflow alias and restarts `ml-gateway`. On gate fail, per-reason counters are emitted to Prometheus so silent-failure weeks are observable.
- **Safeguarding** — `rollback-ctrl` executes automated rollback on sustained quality or drift alerts: swaps the `@production` MLflow alias back one version and restarts `ml-gateway` via the Docker API. Per-alertname cooldown handles Alertmanager retry dedup; defensive version floor refuses to swap below v1.
- **Data quality** — Elnath's three checkpoints: post-ingestion validation, training-set quality checks with rejection logs, and live drift monitor fed from the HTR consumer.

---

## Key design documents

- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — node provisioning, one-command bring-up, teardown, Path A cross-stack bring-up
- [`docs/SAFEGUARDING.md`](docs/SAFEGUARDING.md) — six-pillar safeguarding plan with concrete enforcement-mechanism citations per pillar

---

## Team

Three-person team.

| Member | Role | Primary ownership |
|---|---|---|
| Yikai Sun | Serving | `ml-gateway`, `ml_hooks` overlay, Prometheus + Grafana monitoring, rollback controller, pipeline scheduler, one-command bring-up, feedback UI |
| Dongting Gao | Training | Model training pipeline, evaluation, quality gates, MLflow integration, ONNX export, retraining runs against gate thresholds |
| Elnath Zhao | Data | Data ingestion (IAM, SQuAD), postgres schema, HTR consumer, region slicer, batch training-set compilation with quality checks, live drift monitor, synthetic data generator |

---

## External datasets

| Dataset | Purpose | License |
|---|---|---|
| [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) | HTR pretraining + fine-tuning | Non-commercial research only |
| [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) | Bi-encoder pretraining + retrieval evaluation | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |

Both datasets are ingested via Elnath's batch pipeline into our MinIO warehouse at `s3://paperless-datalake/warehouse/iam_dataset/` and `s3://paperless-datalake/warehouse/squad_dataset/`. Lineage (source URL, ingestion timestamp, row counts, validator output) is captured on every ingest.

---

## Optional: exercise the system after bring-up

```bash
# Populate realistic synthetic traffic (uploads, searches, corrections, feedback)
# against the live Paperless REST API. Token-authenticated via Elnath's generator.
bash scripts/run_data_generator.sh --rate 2.0 --duration 120

# Run the 13-checkpoint end-to-end integration test
bash scripts/verify_integration.sh

# Demo the retrain → evaluate → quality gate → promote cycle
docker compose exec pipeline-scheduler python force_tick.py

# Observe the automated rollback chain firing (alert → webhook → alias swap → restart)
bash scripts/seed_demo.sh --trigger-alert
```

---

## Reproducing from scratch

100% of project materials are in Git. To reproduce the full system on a fresh Chameleon `gpu_p100` node:

1. Clone this repository:
   ```bash
   git clone https://github.com/Palomarr/paperless-ml.git
   cd paperless-ml
   ```
2. Running `bash scripts/chameleon_setup.sh` from this repo to clone the three peer repos, install prerequisites, and bring up the full stack.
3. Following the optional exercise steps above to verify feedback capture, retraining, and rollback end-to-end

All service credentials are development defaults baked into `docker-compose.yml` with inline comments flagging them as dev-only.
