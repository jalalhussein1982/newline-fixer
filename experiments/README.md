# Experiments

Rendered by `scripts/results_table.py`; do not edit by hand.


## m1-baselines

Results at commit `a3e5a6b6e626`, 2026-10-01T23:12:48+00:00.

| set | system | gaps | macro-F1 | classes | break-F1 | JOIN F1 | PARA F1 | wrong-join /1k | damage | str≠raw | str≠norm | para match |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | identity | 75829 | 0.435 | JOIN,SPACE,NL,PARA | 0.374 | 0.000 | 0.439 | 0.00 | 0.0000 | 0.548 | 0.000 | 0.158 |
| V1 | rules | 75829 | 0.577 | JOIN,SPACE,NL,PARA | 0.438 | 0.604 | 0.408 | 0.00 | 0.0144 | 0.788 | 0.780 | 0.142 |
| V3 | identity | 9215 | 1.000 | SPACE,NL,PARA | 1.000 | 0.000 | 1.000 | 0.00 | 0.0000 | 0.000 | 0.000 | 1.000 |
| V3 | rules | 9215 | 0.805 | SPACE,NL,PARA | 0.957 | 0.000 | 0.746 | 0.00 | 0.0110 | 0.350 | 0.350 | 0.772 |
