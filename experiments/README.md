# Experiments

Rendered by `scripts/results_table.py`; do not edit by hand.


## m1-baselines

Results at commit `2be9bfeeb0ac`, 2026-10-02T03:06:05+00:00.
Sets: V1=4f22b6469bbd, V2=574867bf0d4d, V3=07db0ab68315 (built from seed 1 at commit 2d3885bd31a2)

| set | system | gaps | macro-F1 | classes | break-F1 | JOIN F1 | PARA F1 | wrong-join /1k | damage | str≠raw | str≠norm | para match |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | identity | 123610 | 0.418 | JOIN,SPACE,NL,PARA | 0.346 | 0.000 | 0.424 | 0.00 | 0.0000 | 0.603 | 0.000 | 0.161 |
| V1 | rules | 123610 | 0.635 | JOIN,SPACE,NL,PARA | 0.513 | 0.682 | 0.410 | 0.00 | 0.0186 | 0.863 | 0.846 | 0.145 |
| V2 | identity | 2401 | 0.753 | SPACE,NL,PARA | 0.714 | 0.000 | 0.864 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.555 |
| V2 | rules | 2401 | 0.806 | SPACE,NL,PARA | 0.869 | 0.000 | 0.776 | 0.00 | 0.0604 | 0.931 | 0.931 | 0.526 |
| V3 | identity | 9187 | 1.000 | SPACE,NL,PARA | 1.000 | 0.000 | 1.000 | 0.00 | 0.0000 | 0.000 | 0.000 | 1.000 |
| V3 | rules | 9187 | 0.940 | SPACE,NL,PARA | 0.988 | 0.000 | 0.933 | 0.00 | 0.0026 | 0.170 | 0.170 | 0.921 |


## m2-scratch

Results at commit `2e48caa41060`, 2026-10-02T13:13:32+00:00.
Sets: V1=4f22b6469bbd, V2=574867bf0d4d, V3=07db0ab68315 (built from seed 1 at commit 2d3885bd31a2)

| set | system | gaps | macro-F1 | classes | break-F1 | JOIN F1 | PARA F1 | wrong-join /1k | damage | str≠raw | str≠norm | para match |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | identity | 123610 | 0.418 | JOIN,SPACE,NL,PARA | 0.346 | 0.000 | 0.424 | 0.00 | 0.0000 | 0.603 | 0.000 | 0.161 |
| V1 | rules | 123610 | 0.635 | JOIN,SPACE,NL,PARA | 0.513 | 0.682 | 0.410 | 0.00 | 0.0186 | 0.863 | 0.846 | 0.145 |
| V1 | scratch | 123610 | 0.922 | JOIN,SPACE,NL,PARA | 0.885 | 0.974 | 0.861 | 0.11 | 0.0365 | 0.903 | 0.903 | 0.656 |
| V2 | identity | 2401 | 0.753 | SPACE,NL,PARA | 0.714 | 0.000 | 0.864 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.555 |
| V2 | rules | 2401 | 0.806 | SPACE,NL,PARA | 0.869 | 0.000 | 0.776 | 0.00 | 0.0604 | 0.931 | 0.931 | 0.526 |
| V2 | scratch | 2401 | 0.733 | SPACE,NL,PARA | 0.737 | 0.000 | 0.785 | 9.16 | 0.0804 | 1.000 | 1.000 | 0.453 |
| V3 | identity | 9187 | 1.000 | SPACE,NL,PARA | 1.000 | 0.000 | 1.000 | 0.00 | 0.0000 | 0.000 | 0.000 | 1.000 |
| V3 | rules | 9187 | 0.940 | SPACE,NL,PARA | 0.988 | 0.000 | 0.933 | 0.00 | 0.0026 | 0.170 | 0.170 | 0.921 |
| V3 | scratch | 9187 | 0.975 | SPACE,NL,PARA | 0.971 | 0.000 | 0.988 | 0.00 | 0.0016 | 0.080 | 0.080 | 0.959 |


## Training runs

| run | commit | class weights | best epoch / run | V1 macro-F1 | V2 macro-F1 | V3 damage | params | device | minutes |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| scratch-v1-inverse | `dd2a39b955d0` | inverse | 3 / 5 | 0.741 |  | 0.0447 | 5,551,692 | cuda | 5.2 |
| scratch-v1 | `dd2a39b955d0` | none | 8 / 8 | 0.922 |  | 0.0016 | 5,551,692 | cuda | 8.6 |


## Service benchmark

Latency is one request at a time; throughput is eight concurrent requests of 2,000 characters. In-process rows are measured on the named machine; http rows go through a running server and carry no size or memory figures.

| label | system | mode | device | disk MB | RSS MB | p50 / p95 ms @500 | @2,000 | @10,000 | chars/s (batch 8) | commit |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| m1-mac-cpu | identity | in-process | cpu | 0.0 | 24 | 0.1 / 0.1 | 0.3 / 0.3 | 1.5 / 1.5 | 7,204,882 | `c181dae5d6cb` |
| m1-mac-cpu | rules | in-process | cpu | 0.2 | 32 | 0.1 / 0.1 | 0.5 / 0.5 | 2.7 / 2.7 | 4,412,221 | `c181dae5d6cb` |
| m1-mac-cpu | scratch | in-process | cpu | 22.6 | 274 | 7.5 / 7.6 | 40.6 / 41.9 | 268.1 / 274.1 | 71,266 | `c181dae5d6cb` |
