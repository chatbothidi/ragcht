"""Concurrent load test for the /chat SSE endpoint.

Usage:
    python scripts/loadtest_chat.py [--url http://localhost:8000] [--query "..."] \\
        [--levels 1,2,3,5,8,12,20] [--cooldown 30]

For each concurrency level N, fires N requests in parallel, fully drains each SSE
stream, and records per-request metrics (success, first-token latency, total
duration, token count, error type). Reports a summary table and dumps raw
results to data/loadtest_results_<timestamp>.json.

Note: this exercises real Vertex AI LLM calls and consumes quota. Run sparingly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

import httpx

DEFAULT_URL = "http://localhost:8000"
DEFAULT_QUERY = "올해 진행하는 학술대회 알려줘"
DEFAULT_LEVELS = [1, 2, 3, 5, 8, 12, 20]
DEFAULT_COOLDOWN = 30
PER_REQUEST_TIMEOUT = 90.0  # 충분히 길게 잡아 retry 백오프도 포착

FALLBACK_PHRASES = (
    "관련 정보를 찾을 수 없습니다",
    "제공된 문서에서 관련 정보를 찾을 수 없습니다",
    "현재 요청이 몰려",
    "답변 생성 중 오류가 발생했습니다",
)


async def one_request(
    client: httpx.AsyncClient, url: str, query: str, idx: int
) -> dict:
    payload = {"query": query, "stream": True}
    started = time.monotonic()
    first_token_at: float | None = None
    text_chunks: list[str] = []
    status = None
    error: str | None = None

    try:
        async with client.stream(
            "POST",
            f"{url}/chat",
            json=payload,
            headers={"Accept": "text/event-stream"},
            timeout=PER_REQUEST_TIMEOUT,
        ) as resp:
            status = resp.status_code
            if resp.status_code != 200:
                error = f"http_{resp.status_code}"
                return _make_record(idx, status, started, first_token_at, text_chunks, error)

            buffer = ""
            async for chunk in resp.aiter_text():
                buffer += chunk
                while "\n" in buffer:
                    line, _, buffer = buffer.partition("\n")
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:]
                    if not raw:
                        continue
                    try:
                        ev = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if ev.get("type") == "token":
                        if first_token_at is None:
                            first_token_at = time.monotonic()
                        text_chunks.append(ev.get("content", ""))
                    elif ev.get("type") == "done":
                        pass
    except httpx.TimeoutException:
        error = "client_timeout"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    return _make_record(idx, status, started, first_token_at, text_chunks, error)


def _make_record(
    idx: int,
    status: int | None,
    started: float,
    first_token_at: float | None,
    text_chunks: list[str],
    error: str | None,
) -> dict:
    ended = time.monotonic()
    full_text = "".join(text_chunks)
    is_fallback = any(p in full_text for p in FALLBACK_PHRASES)
    return {
        "idx": idx,
        "status": status,
        "first_token_s": round(first_token_at - started, 3) if first_token_at else None,
        "total_s": round(ended - started, 3),
        "chars": len(full_text),
        "is_fallback": is_fallback,
        "error": error,
    }


async def run_level(
    client: httpx.AsyncClient, url: str, query: str, n: int
) -> list[dict]:
    print(f"\n=== Concurrency N={n} ===  starting {n} requests")
    tasks = [one_request(client, url, query, i) for i in range(n)]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return results


def summarize(level: int, records: list[dict]) -> dict:
    successes = [r for r in records if r["status"] == 200 and not r["error"] and not r["is_fallback"] and r["chars"] > 0]
    fallbacks = [r for r in records if r["is_fallback"]]
    errors = [r for r in records if r["error"]]
    total_s = [r["total_s"] for r in records if r["total_s"] is not None]
    first_s = [r["first_token_s"] for r in records if r["first_token_s"] is not None]

    def stat(xs: list[float], q: float) -> float | None:
        if not xs:
            return None
        xs_sorted = sorted(xs)
        k = max(0, min(len(xs_sorted) - 1, int(round(q * (len(xs_sorted) - 1)))))
        return round(xs_sorted[k], 2)

    return {
        "n": level,
        "success": f"{len(successes)}/{len(records)}",
        "fallback": len(fallbacks),
        "error": len(errors),
        "total_p50": stat(total_s, 0.5),
        "total_p95": stat(total_s, 0.95),
        "first_p50": stat(first_s, 0.5),
        "first_p95": stat(first_s, 0.95),
        "mean_total": round(statistics.mean(total_s), 2) if total_s else None,
    }


def print_summary_row(s: dict) -> None:
    print(
        f"N={s['n']:>3}  success={s['success']:>6}  fallback={s['fallback']:>2}  "
        f"error={s['error']:>2}  total p50={s['total_p50']}s p95={s['total_p95']}s  "
        f"first p50={s['first_p50']}s p95={s['first_p95']}s"
    )


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--levels", default=",".join(str(x) for x in DEFAULT_LEVELS))
    parser.add_argument("--cooldown", type=float, default=DEFAULT_COOLDOWN)
    parser.add_argument("--warmup", action="store_true", default=True)
    args = parser.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x.strip()]
    print(f"Target: {args.url}")
    print(f"Query: {args.query!r}")
    print(f"Levels: {levels}")
    print(f"Per-level cooldown: {args.cooldown}s")

    all_summaries: list[dict] = []
    all_records: dict[str, list[dict]] = {}

    async with httpx.AsyncClient() as client:
        if args.warmup:
            print("\n--- Warmup (1 request, untimed) ---")
            try:
                await one_request(client, args.url, args.query, idx=-1)
            except Exception as exc:
                print(f"warmup error: {exc}")

        for i, n in enumerate(levels):
            records = await run_level(client, args.url, args.query, n)
            all_records[f"N={n}"] = records
            summary = summarize(n, records)
            all_summaries.append(summary)
            print_summary_row(summary)
            for r in records:
                tag = "OK" if (r["status"] == 200 and not r["error"] and not r["is_fallback"] and r["chars"] > 0) else (
                    "FALLBACK" if r["is_fallback"] else "ERROR" if r["error"] else "EMPTY"
                )
                print(f"  [{tag}] req={r['idx']} status={r['status']} total={r['total_s']}s "
                      f"first={r['first_token_s']}s chars={r['chars']} error={r['error']}")

            if i < len(levels) - 1:
                print(f"  ... cooldown {args.cooldown}s")
                await asyncio.sleep(args.cooldown)

    print("\n=== Final summary ===")
    for s in all_summaries:
        print_summary_row(s)

    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"loadtest_results_{ts}.json"
    out_path.write_text(
        json.dumps(
            {
                "config": {
                    "url": args.url,
                    "query": args.query,
                    "levels": levels,
                    "cooldown": args.cooldown,
                },
                "summary": all_summaries,
                "records": all_records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
