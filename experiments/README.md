# Experiments

Rendered by `scripts/results_table.py`; do not edit by hand.


## m1-baselines

Results at commit `650802baa8da`, 2026-10-02T01:49:51+00:00.
Sets: V1=4f22b6469bbd, V3=07db0ab68315 (built from seed 1 at commit 2d3885bd31a2)

| set | system | gaps | macro-F1 | classes | break-F1 | JOIN F1 | PARA F1 | wrong-join /1k | damage | str≠raw | str≠norm | para match |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | identity | 123610 | 0.418 | JOIN,SPACE,NL,PARA | 0.346 | 0.000 | 0.424 | 0.00 | 0.0000 | 0.603 | 0.000 | 0.161 |
| V1 | rules | 123610 | 0.635 | JOIN,SPACE,NL,PARA | 0.513 | 0.682 | 0.410 | 0.00 | 0.0186 | 0.863 | 0.846 | 0.145 |
| V3 | identity | 9187 | 1.000 | SPACE,NL,PARA | 1.000 | 0.000 | 1.000 | 0.00 | 0.0000 | 0.000 | 0.000 | 1.000 |
| V3 | rules | 9187 | 0.940 | SPACE,NL,PARA | 0.988 | 0.000 | 0.933 | 0.00 | 0.0026 | 0.170 | 0.170 | 0.921 |
