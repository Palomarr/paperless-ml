# Deployment — Paperless-ML on Chameleon CHI@TACC

This document explains how to deploy the Paperless-ML stack on a fresh
Chameleon bare-metal GPU node, end-to-end. The repository is the single
source of truth for configuration; `scripts/chameleon_setup.sh` is the
one-command bring-up.

## 1. Reserve the node

Reserve a GPU node via `python-chi` in a Jupyter notebook:

- `paperless_data/provision_chameleon.ipynb` — reserves a CHI@TACC
  `gpu_p100` bare-metal node, creates the lease, allocates and assigns a
  floating IP, waits for the instance to boot. (The same flow works for
  `gpu_rtx_6000` on CHI@UC if you prefer that hardware; edit the
  `choose_site` + flavor values accordingly.)

Expected result:
- Lease name, e.g. `proj02_serving`
- Instance with a floating IP (example: `192.5.86.123`)
- Image: `CC-Ubuntu24.04-CUDA`

## 2. SSH into the node

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@<floating-ip>
```

## 3. Clone the repo and run the setup script

```bash
git clone https://github.com/Palomarr/paperless-ml.git ~/paperless-ml
cd ~/paperless-ml
bash scripts/chameleon_setup.sh
```

The script handles host prerequisites that compose can't do itself
(Docker install, NVIDIA toolkit, peer-repo cloning), then hands off to
compose — which declares the `paperless_ml_net` shared bridge, creates
it on first boot, and brings up all services with health-gated
dependency ordering:

1. Install Docker + NVIDIA Container Toolkit (skipped if already present).
2. Clone `REDES01/paperless_data`, `REDES01/paperless_data_integration`,
   and `gdtmax/paperless_training_integration` as siblings (or pull to
   latest if already cloned).
3. `docker compose -f docker-compose.yml -f docker-compose.shared.yml up -d`
   — compose self-creates `paperless_ml_net`, brings up all 18 services
   (including `ml-gateway`, `mlflow`, `pipeline-scheduler`, `rollback-ctrl`,
   plus observability and storage) in health-gated dependency order.
4. Extract the Paperless admin API token and print it in the summary.
5. Run `scripts/verify_integration.sh` (13-checkpoint end-to-end test).

Once the stack is up, the automated pipeline and safeguarding services
are passive-ready:

- `pipeline-scheduler` polls the feedback table every 10 minutes and
  auto-triggers retraining when ≥500 new HTR corrections AND ≥24 hours
  have accumulated since the last run. On gate pass it promotes via
  `@production` alias and restarts ml-gateway.
- `rollback-ctrl` listens on the Alertmanager webhook; on any firing
  alert labeled `rollback_trigger: "true"` it executes an MLflow alias
  swap (to previous version) + ml-gateway restart, with per-alertname
  cooldown and a defensive version floor.

Expected runtime on a cold node: ~8–12 minutes (the ml-gateway image build
is the long pole — TrOCR weights + PyTorch CPU wheels).

## 4. Access URLs

Once the script finishes it prints the public URLs. With floating IP `X.X.X.X`:

| Service | URL | Auth |
|---|---|---|
| Paperless web | `http://X.X.X.X:8000` | admin / admin |
| Feedback UI | `http://X.X.X.X:8000/ml-ui/` | admin / admin |
| Grafana dashboards | `http://X.X.X.X:3000` | admin / admin |
| Prometheus alerts | `http://X.X.X.X:9090/alerts` | — |
| Alertmanager | `http://X.X.X.X:9093` | — |
| Qdrant dashboard | `http://X.X.X.X:6333/dashboard` | — |
| MinIO console | `http://X.X.X.X:9001` | minioadmin / minioadmin |
| MLflow UI | `http://X.X.X.X:5050` | — |

Internal-only services (postgres, redis, redpanda kafka, alertmanager webhook)
are not exposed on the floating IP and stay on the compose default network.

## 5. HTR Flow integration

The setup script **clones** all three peer repos as siblings but only
starts *our* stack. For a full end-to-end HTR flow (Paperless upload →
event → consumer → HTR → postgres), bring up Elnath's consumer + drift
monitor stacks separately. Training runs on demand rather than as a
long-running service — see §Training cycle below.

Peer repos are on plain `main`:

```bash
# Elnath's data stack (postgres, MinIO, redpanda — his own compose)
cd ~/paperless_data
sg docker -c 'make up'

# Elnath's HTR consumer
cd ~/paperless_data_integration/htr_consumer
sg docker -c 'docker compose up -d'

# Elnath's drift monitor (Point 3 data-quality deliverable).
# Fit-once prerequisite on the data-role machine (one-shot, writes reference
# to our MinIO at s3://paperless-datalake/warehouse/drift_reference/htr_v1/cd):
#   cd ~/paperless_data
#   MINIO_ENDPOINT=<host>:9000 MINIO_ACCESS_KEY=minioadmin MINIO_SECRET_KEY=minioadmin \
#     python scripts/build_drift_reference.py
#
# Then on the VM:
cd ~/paperless_data_integration
sg docker -c 'docker compose -f drift_monitor/compose.yml up -d --build'

# The drift monitor's /metrics is already in our Prometheus scrape config
# (ops/prometheus/prometheus.yml, job_name=drift-monitor). The HtrInputDrift
# alert (ops/prometheus/alerts.yml) fires through our existing Alertmanager
# → rollback-ctrl chain on sustained drift.
```

### Training cycle

Dongting's training pipeline is invoked on demand rather than running
as a long-running service. Two paths:

```bash
# Automated path: pipeline-scheduler auto-triggers when conditions are met
# (≥500 new corrections AND ≥24h since last run). No manual invocation needed.

# Manual path: force one pipeline tick now.
docker compose exec pipeline-scheduler python force_tick.py
```

Either path runs train → eval → quality_gate in throwaway training
containers spawned by the scheduler, and on gate pass promotes the new
version + restarts ml-gateway.

Verify with a Paperless upload — see `docs/HANDOFF.md` §1 proof-of-working
flow (document 24 reference) for expected log lines.

## 6. Teardown

```bash
cd ~/paperless-ml
docker compose -f docker-compose.yml -f docker-compose.shared.yml down
# Optional: remove peer stacks
(cd ~/paperless_data && sg docker -c 'make down') || true
(cd ~/paperless_data_integration/htr_consumer && sg docker -c 'docker compose down') || true
(cd ~/paperless_data_integration && sg docker -c 'docker compose -f drift_monitor/compose.yml down') || true
# Release the Chameleon lease from the provisioning notebook when done.
```

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` fails inside container | CC-Ubuntu24.04-CUDA driver + toolkit mismatch | `sudo systemctl restart docker`; re-run script |
| `ml-gateway` never becomes healthy | First-boot model download stalled | `docker compose logs --tail=100 ml-gateway`; wait ~2 more minutes or rebuild |
| verify checkpoint 6 fails | Redpanda auto-create races on cold start | re-run `scripts/verify_integration.sh` — it's idempotent |
| Port 8000 times out from your laptop | Chameleon security group doesn't allow ingress | Use `lease.Lease.show_security_group_rules()` in the provisioning notebook to open ports 8000/3000/9090/9093/6333/9001/5050 |
| `sg docker -c` not found | Running on a non-Debian OS | Script assumes Ubuntu; port to `dnf`/`systemctl` if you switch |
| `network paperless_ml_net exists but was not created by compose` | Stale network from pre-IaC-refactor state (pre-`9e46055`) still has no compose label | `docker compose down && docker network rm paperless_ml_net && docker compose up -d` — compose recreates with the proper label on first boot |
| Prometheus doesn't pick up a new scrape target or alert rule after `git pull` | Bind-mount inode staleness: `git pull` atomically-renames files, leaving the container pointed at the old inode | `docker compose up -d --force-recreate prometheus` — `restart` is not enough, the container must be torn down so the mount re-resolves |
| `pipeline-scheduler` logs "skipped_insufficient_corrections" forever | Production defaults require ≥500 new corrections AND ≥24h since last run; on a fresh stack neither is met | For demo/testing: `docker compose exec pipeline-scheduler python force_tick.py` — bypasses both gates and runs one pipeline cycle |
| `rollback-ctrl` returns `at_version_floor` | Only one registered version of `htr` exists — nothing to roll back to | Train + register additional versions first (scheduler creates one per gate pass), or seed a stub version for demo via `mlflow.pyfunc.log_model(..., registered_model_name="htr")` |

## 8. Flags

`scripts/chameleon_setup.sh` accepts:

- `--skip-verify` — skip the `verify_integration.sh` run at the end
- `--skip-peers` — don't clone the three peer repos
  (`REDES01/paperless_data`, `REDES01/paperless_data_integration`,
  `gdtmax/paperless_training_integration`). Useful when you're only
  smoke-testing our stack in isolation or when the peer repos are
  already present from a prior run.

## 9. MinIO bucket layout — `paperless-images` vs `paperless-datalake`

The stack creates two MinIO buckets via `minio-init`:

| Bucket | Contents | Access | Purpose |
|---|---|---|---|
| `paperless-images` | HTR region crops (`documents/<uuid>/regions/<uuid>.png`) | Anonymous download | Short-lived working storage. Elnath's region slicer writes crops here; our `ml-gateway` fetches them for TrOCR inference. |
| `paperless-datalake` | Training datasets, feedback exports, model artifacts (under `warehouse/<name>/`) | Private (requires MinIO credentials) | Archival / retraining feed. Bucket name + `warehouse/` prefix match the convention in Elnath's `paperless_data` repo (see `batch_pipeline/batch_htr.py:50` and `ingestion/ingest_iam.py:28`). |

### The `warehouse/` prefix

Objects in `paperless-datalake` are organized as:

```
paperless-datalake/
├── mlflow-artifacts/                ← MLflow run artifacts (models + metrics) — seeded placeholder by minio-init
└── warehouse/
    ├── .keep                        ← placeholder so the prefix is visible in MinIO console
    ├── feedback/                    ← future: exported HTR corrections (Parquet shards)
    ├── htr_training/                ← Elnath's batch pipeline output
    ├── iam_dataset/                 ← IAM handwriting dataset shards
    ├── retrieval_training/          ← bi-encoder training pairs
    ├── squad_dataset/               ← SQuAD retrieval eval split
    ├── models/                      ← promoted model snapshots: v1/, v2/, ... written by deploy_model.sh / pipeline-scheduler
    └── drift_reference/             ← Elnath's MMD reference set for drift_monitor (htr_v1/cd/*)
```

ml-gateway and paperless-web both receive `WAREHOUSE_BUCKET=paperless-datalake`
and `WAREHOUSE_PREFIX=warehouse` via environment variables, so any future
code (feedback-archival task, pipeline-scheduler input stage,
model-artifact writer) reads the path from env rather than hardcoding.

### Relation to CHI@TACC

Architecture line 935–936 names **CHI@TACC object storage** as the authoritative persistent store for training datasets and ingested data. Our `paperless-datalake` bucket is a stand-in for that: same bucket/prefix layout, same access semantics, ephemeral to the compose stack. Swapping to real CHI@TACC in a production deployment is an endpoint change only — point MinIO's `server` at a CHI@TACC S3 endpoint, or override `WAREHOUSE_BUCKET` and point the clients directly at `https://chi.tacc.chameleoncloud.org:7480/...`. No code changes required in ml-gateway or ml_hooks.
