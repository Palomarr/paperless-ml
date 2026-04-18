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
