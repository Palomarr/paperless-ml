# paperless_patches

Overlay mounted into the official `ghcr.io/paperless-ngx/paperless-ngx` image at
`/usr/src/paperless/patches`. No upstream fork.

## How it loads

1. `docker-compose.yml` bind-mounts this directory into the container.
2. `PAPERLESS_APPS=ml_hooks` is splatted into `INSTALLED_APPS` by Paperless
   (see upstream `src/paperless/settings/__init__.py` line 146).
3. `PYTHONPATH` includes `/usr/src/paperless/patches` so `ml_hooks` resolves.
4. `ml_hooks.apps.MlHooksConfig.ready()` wires the `document_consumption_finished`
   and `document_updated` signal handlers.
5. Celery autodiscovery picks up `ml_hooks.tasks` via the existing Paperless
   Celery app — no extra registration.

## Hook points (verified against paperless-ngx source)

| Event                            | Where it fires                           | What we do                         |
| -------------------------------- | ---------------------------------------- | ---------------------------------- |
| `document_consumption_finished`  | `consumer.py:648`, after OCR text stored | Enqueue HTR + embedding tasks      |
| `document_updated`               | `consumer.py:722`, `views.py` on edits   | Re-enqueue embedding on edits      |
| Feedback API                     | `/api/ml/feedback/` (DRF ViewSet)        | Writes `Feedback` rows to Postgres |

### URL mounting

`MlHooksConfig.ready()` inserts `re_path(r"^api/ml/", include("ml_hooks.urls"))`
at position 0 of `paperless.urls.urlpatterns`. This runs during `django.setup()`,
before any URL resolution, and is idempotent (guarded by a module-level marker).
No fork of `paperless/urls.py` needed.

Available endpoints:

| Method | Path                       | Purpose                             |
| ------ | -------------------------- | ----------------------------------- |
| GET    | `/api/ml/feedback/`        | List feedback rows (authenticated)  |
| POST   | `/api/ml/feedback/`        | Submit HTR correction / rating      |
| GET    | `/api/ml/feedback/{id}/`   | Retrieve a single feedback row      |

Execution order inside Paperless consumer:
`parse → _store(text) → document_consumption_finished → files copied → post_consume_script`.
We hook the signal, **not** the post-consume script, so ML tasks start before
file moves and the document is guaranteed to exist in the DB.

## Layout

```
paperless_patches/
├── ml_hooks/                 # Django app (added via PAPERLESS_APPS)
│   ├── apps.py               # AppConfig.ready() wires signals
│   ├── models.py             # Feedback table
│   ├── migrations/
│   ├── signal_handlers.py    # enqueue HTR + embed
│   ├── tasks.py              # Celery @shared_task wrappers
│   ├── ml_client.py          # FastAPI HTTP client w/ retries
│   ├── serializers.py
│   ├── views.py              # POST /api/ml/feedback/
│   ├── urls.py
│   └── admin.py
└── README.md
```
