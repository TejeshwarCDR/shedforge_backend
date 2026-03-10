"""Lightweight performance smoke checks for API latency budgets.

Writes a JSON report consumable by CI artifact uploads.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx


@dataclass
class EndpointStats:
    endpoint: str
    samples_ms: list[float]
    avg_ms: float
    p95_ms: float
    max_ms: float


@dataclass
class PerfReport:
    base_url: str
    sample_count: int
    budget_avg_ms: float
    budget_p95_ms: float
    endpoints: list[EndpointStats]


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100) * (len(ordered) - 1)))))
    return ordered[idx]


def collect_endpoint(client: httpx.Client, endpoint: str, sample_count: int) -> EndpointStats:
    latencies: list[float] = []
    for _ in range(sample_count):
        start = time.perf_counter()
        response = client.get(endpoint)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if response.status_code >= 400:
            raise RuntimeError(f"{endpoint} returned {response.status_code}: {response.text[:200]}")
        latencies.append(elapsed_ms)
    return EndpointStats(
        endpoint=endpoint,
        samples_ms=[round(item, 2) for item in latencies],
        avg_ms=round(statistics.fmean(latencies), 2),
        p95_ms=round(percentile(latencies, 95), 2),
        max_ms=round(max(latencies), 2),
    )


def main() -> int:
    base_url = os.getenv("PERF_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    sample_count = int(os.getenv("PERF_SAMPLE_COUNT", "8"))
    budget_avg_ms = float(os.getenv("PERF_BUDGET_AVG_MS", "800"))
    budget_p95_ms = float(os.getenv("PERF_BUDGET_P95_MS", "1500"))
    report_path = Path(os.getenv("PERF_REPORT_PATH", "backend/perf-results/api-latency-smoke.json"))

    endpoints = [
        "/api/health/live",
        "/api/health/ready",
        "/api/system/analytics",
    ]

    stats: list[EndpointStats] = []
    with httpx.Client(base_url=base_url, timeout=20.0) as client:
        for endpoint in endpoints:
            stats.append(collect_endpoint(client, endpoint, sample_count))

    report = PerfReport(
        base_url=base_url,
        sample_count=sample_count,
        budget_avg_ms=budget_avg_ms,
        budget_p95_ms=budget_p95_ms,
        endpoints=stats,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "base_url": report.base_url,
                "sample_count": report.sample_count,
                "budget_avg_ms": report.budget_avg_ms,
                "budget_p95_ms": report.budget_p95_ms,
                "endpoints": [asdict(item) for item in report.endpoints],
            },
            indent=2,
        )
    )

    failures: list[str] = []
    for item in stats:
        if item.avg_ms > budget_avg_ms:
            failures.append(f"{item.endpoint} avg {item.avg_ms}ms > budget {budget_avg_ms}ms")
        if item.p95_ms > budget_p95_ms:
            failures.append(f"{item.endpoint} p95 {item.p95_ms}ms > budget {budget_p95_ms}ms")

    if failures:
        print("Performance smoke failed:")
        for message in failures:
            print(f"- {message}")
        return 1

    print(f"Performance smoke passed. Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
