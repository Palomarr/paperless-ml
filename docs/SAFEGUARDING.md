# Safeguarding Plan — Paperless-ngx ML Integration

**Team**: Serving (Yikai), Data + Training-integrated (Elnath), Training-reference-scaffold (Dongting). Mid-project the production training pipeline migrated from Dongting's `gdtmax/paperless_training_integration` to Elnath's `paperless_data_integration/training/` (real TrOCR fine-tune on IAM, fired by Airflow's `htr_retraining` DAG); Dongting's repo remains as the original reference scaffold the integrated version was adapted from.
**System under scope**: Paperless-ngx with HTR (TrOCR) + semantic search (MPNet + Qdrant), event-driven feedback pipeline via Redpanda, Prometheus + Alertmanager monitoring, retraining via MLflow.

This plan covers the six principles required by the project spec: fairness, explainability, transparency, privacy, accountability, robustness. For each pillar we describe (1) how the principle applies to our specific system, (2) concrete mechanisms we have implemented and where they live in the codebase, (3) remaining gaps we acknowledge, and (4) planned mitigations for those gaps within the project window.

---

## 1. Fairness

### What it means in our system

Paperless-ngx is a document management tool. Our ML features (HTR and semantic search) process user-uploaded documents. "Fairness" for our system primarily means that the HTR model transcribes all handwriting styles with comparable quality and the semantic search ranks documents without systematically disadvantaging classes of users, query styles, or document languages. Since our intended deployment context is an academic department, fairness concerns center on (a) writer-demographic variation in HTR accuracy and (b) query-phrasing variation in search recall.

### Mechanisms implemented

- **Quality gates during training.** The integrated training pipeline at [`paperless_data_integration/training/trainer.py`](https://github.com/REDES01/paperless_data_integration/blob/main/training/trainer.py) evaluates val_cer against threshold (`val_cer < 1.0`) at the end of each fine-tune; only on PASS does the trainer call `mlflow.register_model()` to add a new `htr/vN` to the Registry. This is fired daily by [`airflow/dags/htr_training.py`](https://github.com/REDES01/paperless_data_integration/blob/main/airflow/dags/htr_training.py)'s `finetune_combined_stage1` task, enforcing the rubric clause that saved models are registered only if they pass a quality gate. The original gate semantics + threshold YAML structure were carried over from Dongting's reference scaffold at [`gdtmax/paperless_training_integration/quality_gate.py`](https://github.com/gdtmax/paperless_training_integration/blob/main/quality_gate.py).
- **Synthetic traffic generator** at [`paperless_data/data_generator/generator.py`](https://github.com/REDES01/paperless_data/blob/main/data_generator/generator.py) emulates the aggregate traffic profile of a ~30-staff academic department(Poisson arrivals, weighted action mix).
- **Low-confidence flagging**: every HTR result carries an `htr_flagged` boolean when the sequence-level confidence falls below `HTR_CONFIDENCE_THRESHOLD` (default 0.5) in [`serving/src/fastapi_app/app_ort.py:62`](../serving/src/fastapi_app/app_ort.py) (the deployed ORT runtime per E1). Flagged regions surface to the user for manual correction via Elnath's HTR Review sidebar tab in Paperless (`paperless-ngx-ml` fork), preventing silent propagation of low-quality transcriptions to downstream search indexing.
- **Correction-rate rollback trigger**: if aggregate correction rate exceeds 30% over a rolling hour (baseline ~12%), Alertmanager fires the `HTRCorrectionRateHigh` alert defined in [`ops/prometheus/alerts.yml`](../ops/prometheus/alerts.yml), which dispatches a webhook to the rollback controller. This protects users from a model that has regressed across the user population.

### Gaps acknowledged

- We do not stratify quality metrics by handwriting style, demographic group, or document language. The current HTR evaluation uses aggregate CER on IAM, which skews toward English cursive samples.
- The synthetic traffic generator covers *breadth* of user behavior but not *depth* of minority-class representation. We have not validated performance on under-represented handwriting styles (e.g., non-Latin scripts, historical hands).

### Planned mitigations within project window

- Add a per-source (`user_upload` / `synthetic` / `test`) breakdown of HTR confidence and correction rate to the Grafana dashboard. This gives us stratified quality signals during the ongoing-operation window.
- Document the known coverage limits of the IAM-only HTR training set in the data design document so users understand the scope of current support.
- Longer term, retraining with user corrections (the feedback-loop pipeline already wired) will shift the training distribution toward our actual user population, reducing bias toward IAM's original writer demographics.

---

## 2. Explainability

### What it means in our system

Our system's status should be transparent to users. Users of the system should be able to tell *why* a particular HTR transcription or search result was produced, and with *how much certainty*. They should also be able to tell when the ML layer has degraded to a keyword-only fallback, so they can calibrate trust accordingly.

### Mechanisms implemented

- **Per-request confidence surfacing.** Every HTR response includes `htr_confidence` (geometric-mean token probability) and `htr_flagged`, exposed via [`fastapi_app/app.py:276-280`](../serving/src/fastapi_app/app.py). The Prometheus `htr_confidence` histogram makes the distribution visible on the Grafana dashboard.
- **Per-result similarity scores.** Search results include `similarity_score` (cosine similarity of MPNet embeddings) per document chunk. The UI shows these scores so users can distinguish "strong match" from "weak match" results.
- **Chunk text transparency.** Every search hit exposes the specific `chunk_text` that matched, not just the document ID. Users can see *what* sentence triggered the match.
- **Fallback disclosure.** The `fallback_to_keyword` boolean in [`app.py:375-382`](../serving/src/fastapi_app/app.py) and the `MlGlobalSearchView` in [`paperless_patches/ml_hooks/views.py`](../paperless_patches/ml_hooks/views.py) explicitly flag when semantic search degraded to keyword-only (either because Qdrant was unavailable or because all similarity scores fell below the `SIMILARITY_THRESHOLD = 0.4` threshold). This tells users they are receiving a different quality of result than usual.
- **Model version provenance.** Every HTR and search response includes `model_version` so users and downstream consumers can tie a specific output back to the exact model that produced it. The values come from `HTR_MODEL_VERSION` and `RETRIEVAL_MODEL_VERSION` environment variables, set per deployment.

### Gaps acknowledged

- We do not generate natural-language explanations of *why* a particular chunk was judged similar (e.g., no attention-weight visualization, no "matched on 'invoice' and 'March'" breakdown).
- Confidence scores are exposed to technically literate users but not accompanied by guidance on how to interpret them (e.g., "confidence < 0.6 is flagged for review").

### Planned mitigations within project window

- Add a brief explainer to the feedback UI (`doc_feedback.html`) describing what the confidence score means and when to trust it.
- Publish the `SIMILARITY_THRESHOLD` and `HTR_CONFIDENCE_THRESHOLD` values in the user-facing documentation so users understand the decision boundary.

---

## 3. Transparency

### What it means in our system

The system should be legible to operators (team, future maintainers) and auditable by stakeholders. Every model output should be traceable to the exact model version that produced it, every configuration change should be recoverable from version control, and every user interaction should be loggable.

### Mechanisms implemented

- **Code + config in Git.** 100% of project artifacts live in public Git repositories. The main repository is tracked at [Palomarr/paperless-ml](https://github.com/Palomarr/paperless-ml); peer repositories for Data and Training are linked in the top-level README.
- **Model version in every response.** Already covered in pillar 2; also serves transparency by making the model version accessible to any log-analysis consumer.
- **Metrics endpoint is public within the stack.** `/metrics` on ml-gateway, Qdrant, Redpanda, and Prometheus itself is scraped by Prometheus and queryable from Grafana. Operators can inspect any counter or histogram in real time.
- **Event bus transparency.** Every user interaction (upload, correction, query, feedback) produces an event on Redpanda, retained per topic configuration. Any authorized consumer can replay these events for audit or analysis.
- **Alert rules file ([`ops/prometheus/alerts.yml`](../ops/prometheus/alerts.yml))** documents the eight monitored conditions and their thresholds with inline comments referencing the architecture document's justifications. The `rollback_trigger: "true"` label distinguishes quality-degradation alerts from infrastructure-availability alerts.
- **Migration history** in `paperless_patches/ml_hooks/migrations/` records every schema change to the Feedback table.
- **Quality-gate-failure observability.** `training_quality_gate_failures_total{model_type, reason}` counter emitted by [`ops/pipeline-scheduler/scheduler.py`](../ops/pipeline-scheduler/scheduler.py) means every gate rejection during retraining is recorded in Prometheus with a label carrying the failing metric and threshold. Silent-failure weeks become observable.
- **Pipeline scheduler decisions** are also exported: `pipeline_scheduler_decisions_total{decision}` increments with every scheduler tick outcome (`triggered`, `skipped_too_recent`, `skipped_insufficient_corrections`), so the lineage of why a retraining did or did not fire is queryable indefinitely.

### Gaps acknowledged

- We do not publish a versioned changelog of model deployments; the link from "which MLflow run is currently serving" to "what changed between this version and the last" requires reading MLflow's run comparison manually.
- Some secrets (minioadmin credentials, Paperless secret key) are plaintext in `docker-compose.yml` for the dev stack. They are clearly labeled as dev-only but are not operationally safe.

### Planned mitigations within project window

- Once Training has registered at least two models in MLflow, we will add a `docs/MODEL_CHANGELOG.md` that pairs each serving deployment with the MLflow run URL and a short human-readable summary.
- For the Chameleon deployment, secrets will move to environment variables sourced from a `.env` file that is itself gitignored. The `.env.example` will document what variables are needed without exposing values.

---

## 4. Privacy

### What it means in our system

Documents uploaded to Paperless can contain sensitive personal or organizational information (invoices, medical records, correspondence). Our ML pipeline sees document content during HTR and text-embedding. We must ensure that (a) each user sees only their own documents, (b) feedback and event data are not accessible to unauthorized consumers, (c) users can opt out of having their corrections used for retraining, and (d) inference is not leaking content to external services.

### Mechanisms implemented

- **Per-user authorization.** Paperless's built-in user model and permission system enforces per-user document isolation. Our modified search view (`MlGlobalSearchView` in [`paperless_patches/ml_hooks/views.py:97-101`](../paperless_patches/ml_hooks/views.py)) explicitly filters semantic results by `get_objects_for_user_owner_aware(request.user, "view_document", Document)` so users never see documents they do not have permission to view, even if Qdrant happens to return them from a shared vector index.
- **Feedback UI authentication.** The `/ml-ui/*` routes are gated by `_ui_login_required` in [`views.py`](../paperless_patches/ml_hooks/views.py), which accepts either session cookies or Basic Auth and rejects unauthenticated requests.
- **Opt-in flags for training data.** The Data role's schema ([`paperless_data/docker/init_sql/00_create_app_tables.sql:54-55`](../../paperless_data/docker/init_sql/00_create_app_tables.sql)) includes `opted_in` and `excluded_from_training` columns on `htr_corrections`, allowing users to decline having their corrections used for retraining. The retraining DAG respects these flags when compiling training sets.
- **In-network inference.** All ML inference runs locally within our Docker compose stack. No user document content is sent to third-party APIs. TrOCR and MPNet model weights are downloaded once from Hugging Face during image build and thereafter run offline against local inputs.
- **Soft-delete support.** The `documents.deleted_at` column in the Data schema supports soft delete; retraining compiles exclude soft-deleted rows.
- **Test-account isolation.** `query_sessions.is_test_account` and `documents.is_test_doc` flags separate emulated traffic from real user data, ensuring synthetic test data does not contaminate retraining corpora.

### Gaps acknowledged

- Redpanda event payloads currently include raw correction text and query strings. If an operator gains access to the event bus, they see user queries and correction content. We have not implemented PII redaction on the event path.
- We do not have an implemented data-retention policy. Feedback rows persist indefinitely in Postgres. Old documents in MinIO are not purged.
- Users cannot self-serve delete their feedback via the UI (the API supports it, but no UI control exists).

### Planned mitigations within project window

- Add a retention documentation section noting that the dev stack has no TTL; this is an operational gap that should be closed before a real deployment.
- If time permits in the ongoing-operation window, add an ml_hooks Celery task that purges Feedback rows older than 90 days.
- Document the opt-out flow: users who want to exclude their corrections can set `opted_in=false` via the DRF API at `/api/ml/feedback/<id>/`.

---

## 5. Accountability

### What it means in our system

Every model output, model deployment, and user interaction should be attributable to a responsible party (a user, a version, a team member) so that when things go wrong we can trace causes, assign ownership, and roll back.

### Mechanisms implemented

- **User attribution on every write.** The `Feedback` model ([`paperless_patches/ml_hooks/models.py:16-22`](../paperless_patches/ml_hooks/models.py)) carries a `user` foreign key populated from `request.user`. Every correction, click, and rating is attributed to a specific Paperless account.
- **Timestamped audit trail.** `created_at` with `auto_now_add=True` on Feedback; `uploaded_at`, `corrected_at`, `created_at` throughout the Data role's schema. Every row is time-stamped at insert.
- **Model version on every inference.** Already covered in pillar 2; also serves accountability by pinning each output to a specific artifact.
- **Event bus as tamper-evident log.** Redpanda retains every uploaded document, correction, query, and feedback event. Events include user ID, timestamp, document ID, and a `schema_version` field ([`paperless_patches/ml_hooks/events.py`](../paperless_patches/ml_hooks/events.py)). These events are the authoritative record of what happened, when, and to whom.
- **Rollback controller executes + logs every trigger.** The webhook receiver at [`ops/rollback-ctrl/app.py`](../ops/rollback-ctrl/app.py) both (a) logs the alert name, status, severity, and labels when any alert fires AND (b) for alerts carrying the `rollback_trigger: "true"` label, executes the rollback by moving the `@production` alias on MLflow's `htr` registered model back one version and restarting ml-gateway via the Docker API. Every execution is logged with `ROLLBACK: htr @production moved vN -> vN-1 (trigger=<alertname>)`, which is the authoritative record of which model version was removed from production and why. Per-alertname 5-minute cooldown dedupes Alertmanager retries so the same alert firing twice does not cause double rollback.
- **Pipeline scheduler logs every promotion.** The scheduler at [`ops/pipeline-scheduler/scheduler.py`](../ops/pipeline-scheduler/scheduler.py) logs `PROMOTE: htr @production -> v<N>` every time a gate-passed model gets promoted to production. Combined with rollback-ctrl's demote lines, the scheduler + rollback-ctrl logs together are a complete audit of which version was serving at any point in time — symmetric forward/reverse paths both produce timestamped log lines.
- **MLflow run tracking** records every training run with hyperparameters, metrics, artifacts, and the person who initiated it. The model registry maps model versions to specific runs; the `@production` alias history (queryable via `MlflowClient.search_model_version_aliases`) records every promotion and every revert.
- **Git history** provides code-level accountability: every change to the pipeline is attributable to a specific commit, author, and message.

### Gaps acknowledged

- No formal on-call rotation or incident-response runbook exists.
- The `@production` alias history gives us which version was serving when, but does not yet carry a human-readable note per promotion (e.g., why this version was promoted, or what training data shift motivated it).

### Planned mitigations within project window

- Add a short incident-response section to the top-level README covering: "how to roll back manually", "how to clear a stuck Celery queue", "how to re-scrape metrics after Prometheus downtime".
- Optionally extend `scripts/deploy_model.sh` and the pipeline scheduler to tag each promotion with a human-readable description (e.g., "promoted on YYYY-MM-DD after 500 new corrections passed quality gate at CER=0.11").

---

## 6. Robustness

### What it means in our system

The system should degrade gracefully under load, partial failure, and model regression. A single component going down should not break the user-facing experience; a degraded model should be caught and rolled back automatically.

### Mechanisms implemented

- **Automatic rollback on quality degradation.** Four rollback-triggering alerts fire via Alertmanager ([`ops/prometheus/alerts.yml`](../ops/prometheus/alerts.yml)): `HTRConfidenceLow` (rolling 1h avg confidence < 0.6), `HTRCorrectionRateHigh` (rolling 1h correction rate > 0.3), `SearchCTRLow` (rolling 2h CTR < 0.15), and `HtrInputDrift` (sustained MMD drift > 0.2/s over 2m, emitted by Elnath's drift monitor). All four dispatch a webhook to the rollback controller at [`ops/rollback-ctrl/app.py`](../ops/rollback-ctrl/app.py), which executes the rollback: queries MLflow for the current `@production` version, moves the alias to `version - 1`, and restarts ml-gateway so it picks up the reverted weights on the next boot.
- **Defensive guards on the rollback path.** The rollback controller has three guard branches visible in its log: (a) happy-path rollback (`rollback_complete` — alias swap + container restart succeeded), (b) version floor (`at_version_floor` — registry is at v1, controller refuses to swap to a version that does not exist rather than silently failing), (c) cooldown dedup (`skipped_cooldown` — second webhook for the same alertname within 5 minutes is logged and dropped, preventing Alertmanager retry amplification from rolling back twice). Each branch exercises a distinct failure mode.
- **Automated adaptation via pipeline scheduler.** When conditions warrant it, [`ops/pipeline-scheduler/scheduler.py`](../ops/pipeline-scheduler/scheduler.py) auto-triggers the retrain → evaluate → quality-gate → promote pipeline. Trigger rules are deliberately compound: (a) at least 500 new HTR corrections have accumulated since the last run (≈ one month of organic production feedback); AND (b) at least 24 hours have elapsed since the last run (floor prevents burst-retraining on a single user's batch corrections — mitigates the annotator-bias failure mode where 500 corrections all arrive in one afternoon from one power user). Both conditions must hold. On gate pass, the scheduler promotes the new version via `@production` alias and restarts ml-gateway; on gate fail, it increments a per-reason Prometheus counter so the operator can see why candidate models were rejected without reading training logs.
- **Drift monitor as early warning.** Elnath's drift monitor (fire-and-forget POST per region from `htr_consumer`) runs an online MMD detector pre-fit on 500 IAM crops. Sustained input distribution shift either causes the scheduler to retrain against the new distribution OR, if the shift is causing model quality degradation faster than retraining can catch up, triggers `HtrInputDrift` and routes through the same rollback chain as the quality alerts. Drift gets either adapted-to or reverted-from, not silently absorbed.
- **Availability alerts.** `MLGatewayDown`, `QdrantDown`, `RedpandaDown` fire within 1 minute of service unavailability. `MLGatewayHighLatency` fires if p95 request duration exceeds 5 seconds for 5 minutes. These do not carry the `rollback_trigger` label — rolling back a model does not fix an infrastructure outage, so these alerts are logged but does not trigger the rollback controller.
- **Keyword fallback.** When semantic search is unavailable, `MlGlobalSearchView` returns keyword-only results with `fallback_to_keyword=true` flagged in the response. Users see slightly different results but the search box keeps working.
- **Mock embedding fallback.** When Qdrant is unavailable but `USE_MOCK_CHUNKS` is set, `_search_mock()` in [`fastapi_app/app.py`](../serving/src/fastapi_app/app.py) serves a small in-memory corpus so developers can debug without a running vector store.
- **Non-blocking metric hooks.** `ml_client.post_fire_and_forget()` in [`paperless_patches/ml_hooks/ml_client.py`](../paperless_patches/ml_hooks/ml_client.py) swallows exceptions. If ml-gateway is down, feedback submission still succeeds at the DB layer; only the metric counter misses. This prevents a downstream failure from blocking user-facing operations.
- **Retry with backoff.** `ml_client.post()` wraps requests in a `urllib3.util.retry.Retry` with 3 attempts, 0.5× backoff factor, and status-forcelist for 502/503/504. Transient network blips do not surface as user errors.
- **Healthchecks everywhere.** Every compose service (`postgres`, `redis`, `qdrant`, `minio`, `redpanda`, `ml-gateway`, `prometheus`, `grafana`, `alertmanager`, `rollback-ctrl`) defines a healthcheck. `depends_on: condition: service_healthy` gates startup order so dependents never see an unready backend.
- **Idempotent upserts.** Qdrant point IDs are UUID5-derived from `document_id:chunk_index` ([`fastapi_app/app.py:56-75`](../serving/src/fastapi_app/app.py)), making re-encoding a document safe. Upstream database upserts use `ON CONFLICT` clauses keyed on stable identifiers.
- **End-to-end verification harness** ([`scripts/verify_integration.sh`](../scripts/verify_integration.sh)) runs 13 checkpoints against a freshly brought-up stack. Regressions surface within a single CI run.

### Gaps acknowledged

- No rate limiting exists on ml-gateway. A misbehaving client (or a DDoS attack) could overload HTR inference.
- No circuit breakers beyond the fixed retry count. If Qdrant is slow (but not down), ml-gateway waits the full timeout.
- No chaos-engineering test suite. We know individual services survive restart, but we have not tested combinations.

### Planned mitigations within project window

- Document the existing healthcheck-and-restart behavior as the primary robustness mechanism; rate limiting is a follow-on item.
- Add a short section to the operator README describing the expected graceful-degradation envelope: which services can go down without breaking the user-facing flow, and which are critical-path.

---

## Summary of principle coverage

| Principle | Primary mechanisms | Key gaps |
|---|---|---|
| Fairness | Quality gates in training, synthetic traffic diversity, low-confidence flagging, correction-rate rollback | No per-group stratified metrics |
| Explainability | Confidence + similarity scores surfaced, chunk text shown, fallback flag, model version | No natural-language explanations |
| Transparency | Architecture doc, Git-tracked code + config, public `/metrics`, event bus replay, migrations, quality-gate-failure counter, scheduler-decision counter | No versioned model deployment changelog |
| Privacy | Per-user document isolation, opt-in flags, in-network inference, test-account isolation | No PII redaction on event payloads, no retention policy |
| Accountability | User FK on every feedback row, timestamped audit trail, model version provenance, rollback-ctrl executes + logs every rollback, pipeline-scheduler logs every promotion, MLflow registry + alias history | No on-call / incident runbook |
| Robustness | Automatic rollback with cooldown + version-floor guards, auto-adaptation via pipeline-scheduler with compound trigger rule, drift monitor early warning, keyword fallback, non-blocking metric hooks, retries, healthchecks, idempotent upserts, verify harness | No rate limiting; no chaos suite |

## Sign-off

This plan was authored by the Paperless-ngx ML team and reviewed by all three role owners (Training, Serving, Data). It will be revisited at the end of the ongoing-operation window and amended with findings from the emulated-production run.
