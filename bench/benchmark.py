"""CPU load test: the stock inference server vs the same server with request batching.

For each server mode it starts uvicorn on 127.0.0.1, fires single-series univariate
forecast requests from N concurrent clients (closed loop), then stops the server.
Writes bench/results.json, bench/results.md and bench/throughput_latency.png.

Run from the build folder:
    HF_HOME=$PWD/.hf .venv/bin/python -I bench/benchmark.py
    .venv/bin/python -I bench/benchmark.py --report-only   # re-render table/chart from results.json
"""

import argparse
import asyncio
import json
import os
import platform
import io
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "bench"
UPSTREAM = ROOT / "upstream"
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"
CONTEXT_LEN, HORIZON = 512, 32
ENDPOINT = "/univariate/forecast/mean"

MODES = {
    # stock = upstream HEAD, untouched (no batching, one model call per request on the thread pool)
    "stock": {"env": {}},
    "batched-10ms": {"env": {"BATCHING_ENABLED": "1", "BATCH_MAX_WAIT_MS": "10", "BATCH_MAX_SIZE": "256"}},
    "batched-0ms": {"env": {"BATCHING_ENABLED": "1", "BATCH_MAX_WAIT_MS": "0", "BATCH_MAX_SIZE": "256"}},
}


def series(i: int) -> list[float]:
    """Deterministic daily-seasonal series with trend and noise; request i always sends the same data."""
    rng = np.random.default_rng(i)
    t = np.arange(CONTEXT_LEN)
    y = 100 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 6.28)) + rng.normal(0, 2, CONTEXT_LEN)
    return [round(float(v), 4) for v in y]


def stock_app_dir() -> Path:
    """Extract upstream HEAD's inference/app (no local changes) into bench/.stock/inference/app."""
    target = BENCH / ".stock"
    archive = subprocess.run(
        ["git", "-C", str(UPSTREAM), "archive", "HEAD", "inference/app"], check=True, capture_output=True
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(target, filter="data")
    return target / "inference"


def start_server(mode: str) -> subprocess.Popen:
    app_dir = stock_app_dir() if mode == "stock" else UPSTREAM / "inference"
    env = {**os.environ, "HF_HUB_OFFLINE": "1", **MODES[mode]["env"]}
    log = open(BENCH / f"server-{mode}.log", "w")
    cmd = [sys.executable, "-I", "-m", "uvicorn", "app.http_server:app", "--app-dir", str(app_dir),
           "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning",
           "--timeout-keep-alive", "120"]  # same for every mode; 5 s default closed idle pooled connections mid-run (httpx.ReadError)  # fmt: skip
    proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}, see {log.name}")
        try:
            if httpx.get(f"{BASE}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    proc.kill()
    raise TimeoutError("server did not become healthy")


def stop_server(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def scrape_batch_counts() -> tuple[float, float] | None:
    # Own connection: a pooled one may have idled past uvicorn's 5 s keep-alive and been closed (httpx.ReadError).
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.get(f"{BASE}/metrics")
    if response.status_code != 200:  # the stock server has no /metrics
        return None
    values = dict(line.split() for line in response.text.splitlines() if line.startswith("tirex_batch_size_"))
    return float(values["tirex_batch_size_sum"]), float(values["tirex_batch_size_count"])


async def run_level(clients: int, n_requests: int, offset: int = 0):
    """Closed loop: `clients` workers each send one request, wait for it, send the next."""
    payloads = [{"context": [series(offset + i)], "prediction_length": HORIZON} for i in range(n_requests)]
    latencies, outputs = [0.0] * n_requests, [None] * n_requests
    next_index = iter(range(n_requests))
    limits = httpx.Limits(max_connections=clients, max_keepalive_connections=clients)
    async with httpx.AsyncClient(limits=limits, timeout=600) as client:

        async def worker():
            for i in next_index:
                t0 = time.perf_counter()
                response = await client.post(BASE + ENDPOINT, json=payloads[i])
                latencies[i] = time.perf_counter() - t0
                response.raise_for_status()
                outputs[i] = response.json()[0]

        before = await scrape_batch_counts()
        start = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(clients)))
        wall = time.perf_counter() - start
        after = await scrape_batch_counts()

    calls = (after[1] - before[1]) if before else None
    row = {
        "clients": clients,
        "requests": n_requests,
        "wall_s": wall,
        "series_per_s": n_requests / wall,
        "p50_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
        "model_calls": calls,
        "series_per_call": (after[0] - before[0]) / calls if calls else None,
    }
    return row, outputs


def hardware() -> dict:
    import torch

    def sysctl(key):
        return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True).stdout.strip()

    return {
        "cpu": sysctl("machdep.cpu.brand_string") or platform.processor(),
        "logical_cpus": os.cpu_count(),
        "ram_gb": round(int(sysctl("hw.memsize") or 0) / 2**30),
        "os": f"macOS {platform.mac_ver()[0]}" if platform.mac_ver()[0] else platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": "cpu",
        "power": subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0:1],
        "low_power_mode": "lowpowermode         1" in subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout,
        "upstream_commit": subprocess.run(
            ["git", "-C", str(UPSTREAM), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }


def benchmark(levels: list[int], modes: list[str], repeats: int) -> dict:
    results = {"hardware": hardware(), "repeats": repeats, "workload": {
        "endpoint": ENDPOINT, "series_per_request": 1, "context_length": CONTEXT_LEN, "prediction_length": HORIZON,
    }, "runs": []}  # fmt: skip
    outputs = {}
    for repeat in range(repeats):
        order = modes[repeat % len(modes) :] + modes[: repeat % len(modes)]  # rotate so no mode always goes first
        for mode in order:
            proc = start_server(mode)
            try:
                for clients in levels:
                    warmup = max(16, clients)
                    asyncio.run(run_level(clients, warmup, offset=10**6))  # not measured
                    row, outs = asyncio.run(run_level(clients, max(64, 4 * clients)))
                    row.update(mode=mode, repeat=repeat)
                    results["runs"].append(row)
                    outputs[(mode, clients, repeat)] = outs
                    print(json.dumps(row), flush=True)
            finally:
                stop_server(proc)

    results["rows"] = []
    for mode in modes:
        for clients in levels:
            runs = [r for r in results["runs"] if r["mode"] == mode and r["clients"] == clients]
            row = {"mode": mode, "clients": clients, "requests": runs[0]["requests"], "runs": len(runs)}
            for key in ("series_per_s", "p50_ms", "p95_ms", "series_per_call"):
                values = [r[key] for r in runs if r[key] is not None]
                row[key] = float(np.median(values)) if values else None
            row["series_per_s_min"] = min(r["series_per_s"] for r in runs)
            row["series_per_s_max"] = max(r["series_per_s"] for r in runs)
            results["rows"].append(row)

    # Same inputs went to every mode: how far do batched forecasts drift from the stock ones?
    results["max_abs_diff_vs_stock"] = {
        mode: max(
            float(np.max(np.abs(np.array(outputs[(mode, c, r)]) - np.array(outputs[("stock", c, r)]))))
            for c in levels
            for r in range(repeats)
        )
        for mode in modes
        if mode != "stock" and "stock" in modes
    }
    return results


def write_report(results: dict) -> None:
    hw = results["hardware"]
    lines = [
        f"Hardware: {hw['cpu']}, {hw['logical_cpus']} logical CPUs, {hw['ram_gb']} GB RAM, {hw['os']}, "
        f"Python {hw['python']}, torch {hw['torch']} ({hw['torch_threads']} intra-op threads), CPU only. "
        f"Upstream commit {hw['upstream_commit']}.",
        "",
        f"Workload: closed-loop clients, each request = 1 univariate series of {results['workload']['context_length']}"
        f" points, prediction_length {results['workload']['prediction_length']}, `{results['workload']['endpoint']}`.",
        "",
        "Each cell is the median over its runs (mode order rotated between repeats); "
        "the Series/s range is min-max over those runs.",
        "",
        "| Server | Clients | Runs | Requests per run | Series/s | Series/s range | p50 latency | p95 latency "
        "| Series per model call |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results["rows"]:
        per_call = f"{r['series_per_call']:.1f}" if r["series_per_call"] else "1 (by design)"
        lines.append(
            f"| {r['mode']} | {r['clients']} | {r['runs']} | {r['requests']} | {r['series_per_s']:.1f} | "
            f"{r['series_per_s_min']:.1f}-{r['series_per_s_max']:.1f} | "
            f"{r['p50_ms']:.0f} ms | {r['p95_ms']:.0f} ms | {per_call} |"
        )
    diffs = results.get("max_abs_diff_vs_stock", {})
    if diffs:
        lines += ["", "Largest absolute difference between batched and stock forecasts for the same inputs "
                  "(series values are around 100-140): " + ", ".join(f"{m}: {d:.2e}" for m, d in diffs.items()) + "."]
    (BENCH / "results.md").write_text("\n".join(lines) + "\n")
    plot(results)


def plot(results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a"]  # validated categorical slots 1-3 (dataviz reference palette)
    markers = ["o", "s", "^"]
    modes = list(dict.fromkeys(r["mode"] for r in results["rows"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    panels = [("series_per_s", "Throughput (series/s, higher is better)", False),
              ("p95_ms", "p95 latency (ms, lower is better, log scale)", True)]  # fmt: skip
    for ax, (key, title, log_y) in zip(axes, panels):
        for mode, color, marker in zip(modes, colors, markers):
            rows = [r for r in results["rows"] if r["mode"] == mode]
            xs, ys = [r["clients"] for r in rows], [r[key] for r in rows]
            ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=7, mec="#fcfcfb", mew=2, label=mode,
                    solid_capstyle="round", solid_joinstyle="round")  # fmt: skip
            if mode in modes[:2]:  # label stock and the default batched mode only, so end labels never collide
                ax.annotate(f"{ys[-1]:,.0f}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                            va="center", fontsize=9, color="#0b0b0b")  # fmt: skip
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.minorticks_off()
        if log_y:
            ax.set_yscale("log")
        else:
            ax.set_ylim(bottom=0)
        ax.set_xlabel("Concurrent clients", color="#52514e")
        ax.set_title(title, loc="left", fontsize=11, color="#0b0b0b")
        ax.grid(True, color="#e6e5e0", lw=1)
        ax.set_axisbelow(True)
        ax.set_facecolor("#fcfcfb")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#d0cfca")
        ax.tick_params(colors="#52514e")
        ax.set_xlim(right=xs[-1] * 1.9)
    axes[0].legend(frameon=False, loc="upper left", fontsize=9)
    hw = results["hardware"]
    power = ", battery + Low Power Mode" if hw.get("low_power_mode") else ""
    fig.suptitle(f"TiRex-2 HTTP server on CPU, single-series requests ({hw['cpu']}, torch {hw['torch']}{power})",
                 x=0.01, ha="left", fontsize=12, color="#0b0b0b")  # fmt: skip
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(BENCH / "throughput_latency.png", facecolor=fig.get_facecolor())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clients", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=list(MODES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.report_only:
        results = json.loads((BENCH / "results.json").read_text())
    else:
        results = benchmark(args.clients, args.modes, args.repeats)
        (BENCH / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    write_report(results)
    print((BENCH / "results.md").read_text())
