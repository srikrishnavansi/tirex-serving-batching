Hardware: Apple M5, 10 logical CPUs, 16 GB RAM, macOS 26.5.2, Python 3.13.12, torch 2.9.1 (4 intra-op threads), CPU only. Upstream commit dbcae91.

Workload: closed-loop clients, each request = 1 univariate series of 512 points, prediction_length 32, `/univariate/forecast/mean`.

Each cell is the median over its runs (mode order rotated between repeats); the Series/s range is min-max over those runs.

| Server | Clients | Runs | Requests per run | Series/s | Series/s range | p50 latency | p95 latency | Series per model call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stock | 1 | 1 | 64 | 14.0 | 14.0-14.0 | 70 ms | 79 ms | 1 (by design) |
| stock | 16 | 1 | 64 | 3.9 | 3.9-3.9 | 4119 ms | 4461 ms | 1 (by design) |
| stock | 64 | 1 | 256 | 3.2 | 3.2-3.2 | 23941 ms | 25102 ms | 1 (by design) |
| stock | 256 | 1 | 1024 | 3.2 | 3.2-3.2 | 76526 ms | 123968 ms | 1 (by design) |
| batched-10ms | 1 | 2 | 64 | 12.4 | 12.3-12.5 | 80 ms | 82 ms | 1.0 |
| batched-10ms | 16 | 2 | 64 | 40.6 | 40.6-40.6 | 407 ms | 433 ms | 9.1 |
| batched-10ms | 64 | 2 | 256 | 53.9 | 53.2-54.6 | 1124 ms | 1725 ms | 25.9 |
| batched-10ms | 256 | 1 | 1024 | 32.7 | 32.7-32.7 | 6855 ms | 11702 ms | 9.8 |
| batched-0ms | 1 | 1 | 64 | 12.7 | 12.7-12.7 | 72 ms | 86 ms | 1.0 |
| batched-0ms | 16 | 1 | 64 | 43.5 | 43.5-43.5 | 364 ms | 378 ms | 8.0 |
| batched-0ms | 64 | 1 | 256 | 51.7 | 51.7-51.7 | 1121 ms | 2021 ms | 18.3 |
| batched-0ms | 256 | 1 | 1024 | 36.1 | 36.1-36.1 | 5449 ms | 10367 ms | 10.9 |
