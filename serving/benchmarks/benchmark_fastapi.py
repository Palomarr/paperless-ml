"""Benchmark FastAPI endpoints at various concurrency levels."""

import argparse
import asyncio
import base64
import io
import json
import statistics
import time

import aiohttp
from PIL import Image


def make_test_image_b64() -> str:
    img = Image.new("RGB", (224, 224), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


async def send_request(session, url, payload):
    start = time.perf_counter()
    try:
        async with session.post(url, json=payload) as resp:
            await resp.json()
            latency = time.perf_counter() - start
            return {"latency": latency, "status": resp.status}
    except Exception as e:
        latency = time.perf_counter() - start
        return {"latency": latency, "status": 0, "error": str(e)}


async def benchmark_endpoint(url, payload, concurrency, num_requests):
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        sem = asyncio.Semaphore(concurrency)

        async def bounded_request():
            async with sem:
                return await send_request(session, url, payload)

        tasks = [bounded_request() for _ in range(num_requests)]
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_time = time.perf_counter() - wall_start

    latencies = [r["latency"] for r in results]
    errors = sum(1 for r in results if r.get("status") != 200)
    latencies.sort()

    return {
        "concurrency": concurrency,
        "num_requests": num_requests,
        "p50_ms": round(statistics.median(latencies) * 1000, 1),
        "p95_ms": round(latencies[int(len(latencies) * 0.95)] * 1000, 1),
        "throughput_rps": round(num_requests / wall_time, 2),
        "error_rate": round(errors / num_requests, 4),
        "wall_time_s": round(wall_time, 2),
    }


async def main(base_url, concurrency_levels, num_requests):
    test_image = make_test_image_b64()
    htr_payload = {
        "image": test_image,
        "document_id": 1,
        "page_number": 1,
        "region_id": 1,
    }
    search_payload = {"query": "handwritten text recognition", "top_k": 5}

    all_results = {}

    for endpoint, payload in [
        ("/predict/htr", htr_payload),
        ("/predict/search", search_payload),
    ]:
        url = f"{base_url}{endpoint}"
        print(f"\n{'='*60}")
        print(f"Benchmarking {endpoint}")
        print(f"{'='*60}")
        endpoint_results = []
        for c in concurrency_levels:
            print(
                f"  concurrency={c}, requests={num_requests} ...",
                end=" ",
                flush=True,
            )
            result = await benchmark_endpoint(url, payload, c, num_requests)
            print(
                f"p50={result['p50_ms']}ms  p95={result['p95_ms']}ms  "
                f"rps={result['throughput_rps']}  errors={result['error_rate']}"
            )
            endpoint_results.append(result)
        all_results[endpoint] = endpoint_results

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=str, default="1,4,8,16")
    parser.add_argument("--output", type=str, help="Save JSON results to file")
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    results = asyncio.run(main(args.url, levels, args.requests))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
