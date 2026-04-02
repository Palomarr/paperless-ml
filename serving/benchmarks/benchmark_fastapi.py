"""Benchmark FastAPI endpoints at various concurrency levels."""

import argparse
import asyncio
import json
import statistics
import time

import aiohttp


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


async def main(base_url, concurrency_levels, num_requests, crop_s3_url):
    htr_payload = {
        "document_id": "a3f7c2e1-9b4d-4e8a-b5c6-1234567890ab",
        "page_id": "b8e2d4f6-7a3c-4b1e-9d5f-abcdef012345",
        "region_id": "c9d3e5a7-6b2f-4c0d-8e4a-fedcba987654",
        "crop_s3_url": crop_s3_url,
    }
    search_payload = {
        "session_id": "d4e5f6a7-8b9c-4d0e-1f2a-3b4c5d6e7f8a",
        "query_text": "handwritten text recognition",
        "user_id": "e5f6a7b8-9c0d-4e1f-2a3b-4c5d6e7f8a9b",
        "top_k": 5,
    }

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
    parser.add_argument(
        "--crop-s3-url",
        type=str,
        default="s3://paperless-images/documents/a3f7c2e1/regions/test.png",
        help="S3/MinIO URL for the test HTR image",
    )
    args = parser.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    results = asyncio.run(main(args.url, levels, args.requests, args.crop_s3_url))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
