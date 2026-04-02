"""
Lightweight integration test for HTR and Search serving endpoints.
Validates responses against the contract JSON schemas in contracts/.
"""

import argparse
import base64
import io
import json
import os
import struct
import sys
import zlib

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONTRACTS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "contracts"))

# ── Type mapping used by the validator ──────────────────────────────────────

TYPE_MAP = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
    list: "list",
}

def infer_type(value):
    """Return the expected Python type for a sample value."""
    # JSON numbers without a decimal part decode as int; treat both int and
    # float as acceptable for numeric fields.
    return type(value)


# ── Tiny PNG generator (10x10 white, no Pillow needed) ──────────────────────

def make_tiny_png(width=10, height=10):
    """Return bytes for a minimal white PNG image."""
    def _chunk(chunk_type, data):
        c = chunk_type + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # raw image data: each row is a filter byte (0) + RGB pixels
    raw = b""
    for _ in range(height):
        raw += b"\x00" + b"\xff\xff\xff" * width
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    return header + ihdr + idat + iend


# ── Helpers ─────────────────────────────────────────────────────────────────

def load_contract(name):
    path = os.path.join(CONTRACTS_DIR, name)
    with open(path) as f:
        return json.load(f)


def check_fields(response_json, expected_sample, label):
    """Compare response fields against the sample contract.
    Returns a list of error strings (empty = pass).
    """
    errors = []
    for key, sample_val in expected_sample.items():
        if key not in response_json:
            errors.append(f"  missing field: {key}")
            continue
        actual = response_json[key]
        expected_t = infer_type(sample_val)
        # Allow int where float is expected and vice-versa for numeric fields
        if expected_t in (int, float) and isinstance(actual, (int, float)):
            continue
        if not isinstance(actual, expected_t):
            errors.append(
                f"  field '{key}': expected {TYPE_MAP.get(expected_t, str(expected_t))}, "
                f"got {TYPE_MAP.get(type(actual), str(type(actual)))}"
            )
    return errors


def test_endpoint(base_url, path, payload, expected_output, label):
    """POST payload to endpoint and validate against expected output contract."""
    url = f"{base_url}{path}"
    print(f"\n{'='*60}")
    print(f"TEST: {label}")
    print(f"  POST {url}")

    try:
        resp = requests.post(url, json=payload, timeout=30)
    except requests.ConnectionError:
        print(f"  FAIL - connection refused")
        return False
    except requests.Timeout:
        print(f"  FAIL - request timed out (30s)")
        return False

    # Check status
    if resp.status_code != 200:
        print(f"  FAIL - HTTP {resp.status_code}")
        try:
            print(f"  Body: {resp.text[:300]}")
        except Exception:
            pass
        return False

    print(f"  HTTP 200 OK")

    # Parse JSON
    try:
        body = resp.json()
    except ValueError:
        print(f"  FAIL - response is not valid JSON")
        return False

    # Validate fields
    errors = check_fields(body, expected_output, label)
    if errors:
        print(f"  FAIL - schema mismatch:")
        for e in errors:
            print(e)
        return False

    print(f"  PASS - all fields present with correct types")
    return True


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Contract integration tests for serving endpoints")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", default=8000, type=int, help="Server port (default: 8000)")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Load contracts
    htr_input = load_contract("htr_input.json")
    htr_output = load_contract("htr_output.json")
    search_input = load_contract("search_input.json")
    search_output = load_contract("search_output.json")

    # Build HTR payload with a base64 test image so it works without MinIO
    png_bytes = make_tiny_png()
    htr_payload = dict(htr_input)
    htr_payload["image_base64"] = base64.b64encode(png_bytes).decode("ascii")

    results = []

    # Test HTR endpoint
    ok = test_endpoint(base_url, "/predict/htr", htr_payload, htr_output, "HTR predict")
    results.append(("HTR predict", ok))

    # Test Search endpoint
    ok = test_endpoint(base_url, "/predict/search", search_input, search_output, "Search predict")
    results.append(("Search predict", ok))

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("All tests passed.")
    else:
        print("Some tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
