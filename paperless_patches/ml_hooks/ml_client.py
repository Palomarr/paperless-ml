import os

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GATEWAY_URL = os.getenv("ML_GATEWAY_URL") or os.getenv(
    "ML_FASTAPI_BASE_URL", "http://ml-gateway:8000",
)
TIMEOUT_SECONDS = float(
    os.getenv("ML_GATEWAY_TIMEOUT") or os.getenv("ML_FASTAPI_TIMEOUT", "30"),
)


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset({"GET", "POST"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


_session = _build_session()


def post(path: str, payload: dict) -> dict:
    url = f"{GATEWAY_URL.rstrip('/')}/{path.lstrip('/')}"
    resp = _session.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    resp.raise_for_status()
    return resp.json()


def post_fire_and_forget(path: str, payload: dict | None = None) -> None:
    """Best-effort POST for metric hooks. Never raises.

    Used by ml_hooks to bump counters on ml-gateway when a user submits a
    correction or clicks a search result. Metric recording must never block
    or fail a user-facing request, so exceptions are swallowed.
    """
    url = f"{GATEWAY_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        _session.post(url, json=payload or {}, timeout=2)
    except Exception:
        pass
