# Deployment — Paperless-ML on Chameleon CHI@UC

This document explains how to deploy the Paperless-ML stack on a fresh
Chameleon bare-metal GPU node, end-to-end. The repository is the single
source of truth for configuration; `scripts/chameleon_setup.sh` is the
one-command bring-up.

## 1. Reserve the node

Reserve a GPU node via `python-chi` in a Jupyter notebook. Our team uses
Elnath's reference notebook:

- `paperless_data/provision_chameleon.ipynb` — reserves a CHI@UC node
  (RTX 6000 or P100), creates the lease, allocates and assigns a floating
  IP, waits for the instance to boot.

Expected result:
- Lease name, e.g. `proj02_serving`
- Instance with a floating IP (example: `192.5.86.123`)
- Image: `CC-Ubuntu24.04-CUDA`

## 2. SSH into the node

```bash
ssh -i ~/.ssh/id_rsa_chameleon cc@<floating-ip>
```

The `cc` user is pre-created on every CC-Ubuntu image. Our setup script
puts `cc` in the `docker` group so nothing else is needed.

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
   and `gdtmax/paperless_training_integration` as siblings (or pull to latest).
3. `docker compose -f docker-compose.yml -f docker-compose.shared.yml up -d`
   — compose self-creates `paperless_ml_net`, brings up services in
   dependency order, waits for healthchecks.
4. Extract the Paperless admin API token and print it in the summary.
5. Run `scripts/verify_integration.sh` (13-checkpoint end-to-end test).

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

## 5. Path A integration (optional, end-to-end HTR round-trip)

The setup script brings up **our** stack with the shared-network overlay
attached — but Elnath's consumer stack is not started automatically.
For a full end-to-end HTR flow (Paperless upload → event → consumer →
HTR → postgres), also bring up the peer stacks on plain `main` (D1 + D3
merged upstream on 2026-04-19, no branch checkout required):

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

Verify with a Paperless upload — see `docs/HANDOFF.md` §1 proof-of-working
flow (document 24 reference) for expected log lines.

## 6. Teardown

```bash
cd ~/paperless-ml
docker compose -f docker-compose.yml -f docker-compose.shared.yml down
# Optional: remove peer stacks
(cd ~/paperless_data && sg docker -c 'make down') || true
(cd ~/paperless_data_integration/htr_consumer && sg docker -c 'docker compose down') || true
# Release the Chameleon lease from the provisioning notebook when done.
```

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `nvidia-smi` fails inside container | CC-Ubuntu24.04-CUDA driver + toolkit mismatch | `sudo systemctl restart docker`; re-run script |
| `ml-gateway` never becomes healthy | First-boot model download stalled | `docker compose logs --tail=100 ml-gateway`; wait ~2 more minutes or rebuild |
| verify checkpoint 6 fails | Redpanda auto-create races on cold start | re-run `scripts/verify_integration.sh` — it's idempotent |
| Port 8000 times out from your laptop | Chameleon security group doesn't allow ingress | Use `lease.Lease.show_security_group_rules()` in the provisioning notebook to open ports 8000/3000/9090/9093/6333/9001 |
| `sg docker -c` not found | Running on a non-Debian OS | Script assumes Ubuntu; port to `dnf`/`systemctl` if you switch |

## 8. Flags

`scripts/chameleon_setup.sh` accepts:

- `--skip-verify` — skip the `verify_integration.sh` run at the end
- `--skip-peers` — don't clone `REDES01/paperless_data` or
  `REDES01/paperless_data_integration` (useful if they're already cloned or
  you're only smoke-testing our stack in isolation)

## 9. What's *not* in this path

- **Terraform / Ansible** — not required for 3-person teams per course
  rubric. The provisioning notebook + this setup script cover the
  reproducible-deployment requirement.
- **Multi-node orchestration** — single-VM compose stack. Kubernetes is
  out of scope.
- **HTTPS / TLS** — demo traffic only, over HTTP.
- **Secrets management** — dev credentials (`admin/admin`,
  `minioadmin/minioadmin`) are baked in. Any real deployment must
  externalise these; see `docs/SAFEGUARDING.md` §3 Transparency.

## 10. MinIO bucket layout — `paperless-images` vs `paperless-datalake`

The stack creates two MinIO buckets via `minio-init`:

| Bucket | Contents | Access | Purpose |
|---|---|---|---|
| `paperless-images` | HTR region crops (`documents/<uuid>/regions/<uuid>.png`) | Anonymous download | Short-lived working storage. Elnath's region slicer writes crops here; our `ml-gateway` fetches them for TrOCR inference. |
| `paperless-datalake` | Training datasets, feedback exports, model artifacts (under `warehouse/<name>/`) | Private (requires MinIO credentials) | Archival / retraining feed. Bucket name + `warehouse/` prefix match the convention in Elnath's `paperless_data` repo (see `batch_pipeline/batch_htr.py:50` and `ingestion/ingest_iam.py:28`). |

### The `warehouse/` prefix

Objects in `paperless-datalake` are organized as:

```
paperless-datalake/
└── warehouse/
    ├── .keep                        ← placeholder so the prefix is visible in MinIO console
    ├── feedback/                    ← future: exported HTR corrections (Parquet shards)
    ├── htr_training/                ← Elnath's batch pipeline output
    ├── iam_dataset/                 ← IAM handwriting dataset shards
    ├── retrieval_training/          ← bi-encoder training pairs
    └── squad_dataset/               ← SQuAD retrieval eval split
```

ml-gateway and paperless-web both receive `WAREHOUSE_BUCKET=paperless-datalake`
and `WAREHOUSE_PREFIX=warehouse` via environment variables, so any future
code (feedback-archival task, retraining DAG input stage, model-artifact
writer) reads the path from env rather than hardcoding.

### Relation to CHI@TACC

Architecture line 935–936 names **CHI@TACC object storage** as the
authoritative persistent store for training datasets and ingested data,
surviving VM deletion. Our `paperless-datalake` bucket is a stand-in for
that: same bucket/prefix layout, same access semantics, ephemeral to the
compose stack. Swapping to real CHI@TACC in a production deployment is
an endpoint change only — point MinIO's `server` at a CHI@TACC S3
endpoint, or override `WAREHOUSE_BUCKET` and point the clients directly
at `https://chi.tacc.chameleoncloud.org:7480/...`. No code changes
required in ml-gateway or ml_hooks.
