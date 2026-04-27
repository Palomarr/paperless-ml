"""
HTR retraining DAG — closed-loop from user corrections to new model version.

Runs `batch_htr` → `finetune_combined_stage1` → `notify_result` daily.

1. build_snapshot  (DockerOperator, image: htr_batch:latest)
     Reads htr_corrections rows from the ML Postgres. Applies quality filters
     (opted_in, non-empty text, user_upload source, deduplicate by region),
     splits into train/val via document-grouped sha256 hashing, writes versioned
     parquet shards + manifest.json to s3://paperless-datalake/warehouse/htr_training/v_<ts>/.
     Exits 0 even when there are no corrections — downstream just uses IAM-only.

2. finetune_combined_stage1  (DockerOperator, image: htr_trainer:latest)
     Reads IAM + the latest correction snapshot, fine-tunes TrOCR-base-stage1,
     evaluates. Registers 'htr' version in MLflow if the quality gate passes.

3. notify_result  (PythonOperator)
     Queries MLflow for the latest registered htr version and prints the
     scripts/deploy_model.sh command for promotion.

This closes the feedback loop: UI correction → ml_postgres row → snapshot
parquet → training input → registered model → manual deploy.

Schedule: daily at 02:00 UTC. Manual trigger via UI Play button or
POST /api/v1/dags/htr_retraining/dagRuns.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from urllib import request as urlrequest

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator

log = logging.getLogger(__name__)

MLFLOW_URL = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Shared env for every DockerOperator task that needs to reach the data stack.
# ml_postgres holds both htr_corrections (batch_htr reads) and everything
# trainer needs for its own lookups. MinIO holds the warehouse bucket.
_DATA_STACK_ENV = {
    "MLFLOW_TRACKING_URI":     MLFLOW_URL,
    "MINIO_ENDPOINT":          "minio:9000",
    "MINIO_ACCESS_KEY":        "admin",
    "MINIO_SECRET_KEY":        "paperless_minio",
    "MINIO_BUCKET":            "paperless-datalake",
    # batch_htr.py wants a DSN string
    "DB_DSN":                  "host=data-stack-postgres dbname=paperless user=user password=paperless_postgres",
    # trainer wants discrete fields (matches training/compose.yml)
    "PAPERLESS_ML_DBHOST":     "data-stack-postgres",
    "PAPERLESS_ML_DBPORT":     "5432",
    "PAPERLESS_ML_DBNAME":     "paperless",
    "PAPERLESS_ML_DBUSER":     "user",
    "PAPERLESS_ML_DBPASSWORD": "paperless_postgres",
}


def _check_latest_registered_version(**_ctx):
    """Query MLflow for the latest registered htr version and print a deploy hint."""
    url = f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/get-latest-versions"
    body = json.dumps({"name": "htr"}).encode()
    req = urlrequest.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlrequest.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as exc:
        log.warning("couldn't query MLflow registry: %s", exc)
        return

    versions = data.get("model_versions", [])
    if not versions:
        log.info("no htr versions registered yet — training run likely failed the quality gate")
        return

    latest = max(int(v["version"]) for v in versions)
    log.info("=" * 60)
    log.info("Latest registered htr version: htr/v%s", latest)
    log.info("=" * 60)
    log.info("")
    log.info("To deploy this version to ml_gateway, run on the VM:")
    log.info("    cd ~/paperless_data_integration && \\")
    log.info("    sg docker -c 'bash scripts/deploy_model.sh %s'", latest)
    log.info("")
    log.info("Then verify:")
    log.info("    curl -s http://localhost:8100/health | python3 -m json.tool")
    log.info("    (htr_version should read 'models:/htr/%s')", latest)


default_args = {
    "owner": "paperless-ml",
    "retries": 0,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="htr_retraining",
    default_args=default_args,
    description=(
        "Closed-loop HTR retraining: user corrections -> snapshot parquet -> "
        "fine-tune -> register. Runs daily, deployment stays manual."
    ),
    schedule="0 2 * * *",  # daily at 02:00 UTC
    start_date=datetime(2026, 4, 22),
    catchup=False,
    max_active_runs=1,
    tags=["htr", "training", "mlflow"],
    doc_md=__doc__,
) as dag:

    # ── 1. build correction snapshot ────────────────────────────────────
    # Reads htr_corrections from ML Postgres, writes parquet + manifest to
    # s3://paperless-datalake/warehouse/htr_training/v_<ts>/. Script exits 0
    # even when there are no corrections — training then falls back to
    # IAM-only, which is correct behavior for a fresh system.
    #
    # Dockerfile default CMD also runs batch_retrieval.py; we override to
    # only run batch_htr.py so a retrieval-pipeline failure can't block
    # HTR training.
    build_snapshot = DockerOperator(
        task_id="build_snapshot",
        image="htr_batch:latest",
        command=["python", "batch_htr.py"],
        network_mode="paperless_ml_net",
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
        environment=_DATA_STACK_ENV,
    )

    # ── 2. fine-tune TrOCR on IAM + the new snapshot ────────────────────
    finetune = DockerOperator(
        task_id="finetune_combined_stage1",
        image="htr_trainer:latest",
        command=["--config", "/app/configs/finetune_combined_stage1.yaml"],
        network_mode="paperless_ml_net",
        auto_remove="force",
        docker_url="unix://var/run/docker.sock",
        mount_tmp_dir=False,
        environment=_DATA_STACK_ENV,
        # The trainer returns exit 1 when the quality gate fails. That's a
        # legitimate outcome — notify_result still runs (trigger_rule=all_done
        # below) and prints whatever the latest registered version is.
        retrieve_output=False,
    )

    # ── 3. print deploy instructions ────────────────────────────────────
    notify = PythonOperator(
        task_id="notify_result",
        python_callable=_check_latest_registered_version,
        # Always run, even if upstream failed — we still want to see the
        # latest registered version in the Airflow logs after a bad run.
        trigger_rule="all_done",
    )

    build_snapshot >> finetune >> notify
