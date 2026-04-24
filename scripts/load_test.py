#!/usr/bin/env python3
"""Load test for the RAG chat endpoint.

Examples:
    # Basic smoke test (5 concurrent, 20 total)
    python scripts/load_test.py

    # Heavier load
    python scripts/load_test.py --concurrency 20 --total 100

    # Custom query
    python scripts/load_test.py --query "결핵 치료 원칙"

    # Verify session isolation (3 users, 3 turns each, concurrent)
    python scripts/load_test.py --isolation

Watches:
    - Latency percentiles (p50/p95/p99)
    - Status breakdown (ok / quota_fallback / generic_fallback / no_docs / http_error / timeout)
    - Throughput (req/sec)
"""
import argparse
import asyncio
import statistics
import time
import uuid
from collections import Counter

import httpx


async def fire_one(client: httpx.AsyncClient, url: str, query: str, session_id: str) -> dict:
    t0 = time.perf_counter()
    try:
        r = await client.post(
            url,
            json={"query": query, "session_id": session_id, "stream": False},
            timeout=120.0,
        )
        dt = time.perf_counter() - t0
        if r.status_code != 200:
            return {"status": "http_error", "code": r.status_code, "dt": dt}
        body = r.json()
        answer = body.get("answer", "") or ""
        if "관련 정보를 찾을 수 없습니다" in answer:
            kind = "no_docs"
        elif "요청이 몰려" in answer:
            kind = "quota_fallback"
        elif "오류가 발생했습니다" in answer:
            kind = "generic_fallback"
        elif len(answer) < 10:
            kind = "too_short"
        else:
            kind = "ok"
        return {
            "status": kind,
            "dt": dt,
            "len": len(answer),
            "session_id": body.get("session_id"),
        }
    except httpx.TimeoutException:
        return {"status": "timeout", "dt": time.perf_counter() - t0}
    except Exception as e:
        return {"status": "error", "error": str(e), "dt": time.perf_counter() - t0}


async def concurrent_test(url: str, query: str, concurrency: int, total: int) -> tuple[list[dict], float]:
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(concurrency)

        async def fire_with_sem(_):
            async with sem:
                sid = f"loadtest-{uuid.uuid4()}"
                return await fire_one(client, url, query, sid)

        print(f"Firing {total} requests (concurrency={concurrency}) at {url}")
        t_start = time.perf_counter()
        results = await asyncio.gather(*[fire_with_sem(i) for i in range(total)])
        t_total = time.perf_counter() - t_start

    return results, t_total


def print_results(results: list[dict], t_total: float) -> None:
    counts = Counter(r["status"] for r in results)
    ok_latencies = [r["dt"] for r in results if r["status"] == "ok"]
    all_latencies = [r["dt"] for r in results]

    print(f"\nTotal wall time : {t_total:.2f}s")
    print(f"Throughput      : {len(results) / t_total:.2f} req/s")

    print("\nStatus breakdown:")
    for status, count in counts.most_common():
        pct = 100 * count / len(results)
        print(f"  {status:20s} {count:4d} ({pct:.1f}%)")

    def pct(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        s = sorted(vals)
        idx = min(int(len(s) * p), len(s) - 1)
        return s[idx]

    if ok_latencies:
        print("\nLatency (ok only):")
        print(f"  p50: {pct(ok_latencies, 0.50):.2f}s")
        print(f"  p95: {pct(ok_latencies, 0.95):.2f}s")
        print(f"  p99: {pct(ok_latencies, 0.99):.2f}s")
        print(f"  avg: {statistics.mean(ok_latencies):.2f}s")
        print(f"  max: {max(ok_latencies):.2f}s")
    print("\nLatency (all):")
    print(f"  p50: {pct(all_latencies, 0.50):.2f}s")
    print(f"  p95: {pct(all_latencies, 0.95):.2f}s")
    print(f"  avg: {statistics.mean(all_latencies):.2f}s")


async def isolation_test(url: str) -> None:
    """3 users × 3 turns concurrently. Verifies sessions don't cross-contaminate."""
    print("Session isolation test: 3 users × 3 turns concurrent")

    users = [
        (f"A-{uuid.uuid4()}", [
            "결핵에 대해 알고 싶어요",
            "치료 기간은 얼마나 걸리나요?",
            "부작용이 있나요?",
        ]),
        (f"B-{uuid.uuid4()}", [
            "2025년 학술대회 하나 알려주세요",
            "그 행사 장소는요?",
            "연수평점은?",
        ]),
        (f"C-{uuid.uuid4()}", [
            "예방접종 안내",
            "대상자는 누구인가요?",
            "신청은 어떻게 하나요?",
        ]),
    ]

    async with httpx.AsyncClient() as client:
        async def user_flow(sid: str, queries: list[str]) -> list[dict]:
            out = []
            for q in queries:
                r = await fire_one(client, url, q, sid)
                r["query"] = q
                out.append(r)
            return out

        t0 = time.perf_counter()
        all_results = await asyncio.gather(*[user_flow(sid, qs) for sid, qs in users])
        t_total = time.perf_counter() - t0

    for (sid, _), results in zip(users, all_results):
        print(f"\nUser {sid[:10]}")
        for r in results:
            status = r["status"]
            dt = r.get("dt", 0)
            n = r.get("len") or 0
            q = r.get("query", "")[:40]
            print(f"  [{status:17s} {dt:5.2f}s {n:4d}chars] {q}")

    print(f"\nTotal: {t_total:.2f}s for 9 requests across 3 sessions")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8000/chat")
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--total", type=int, default=20)
    p.add_argument("--query", default="2025년 행사 하나만 알려주세요")
    p.add_argument("--isolation", action="store_true", help="Run session isolation test instead")
    args = p.parse_args()

    if args.isolation:
        asyncio.run(isolation_test(args.url))
    else:
        results, t_total = asyncio.run(
            concurrent_test(args.url, args.query, args.concurrency, args.total)
        )
        print_results(results, t_total)


if __name__ == "__main__":
    main()
