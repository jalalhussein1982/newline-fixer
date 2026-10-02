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


## m5-candidates

Scratch p50 per 256-token window: 22.5 ms; limit (3x): 67.4 ms.

| run | pretrained | params | V1 macro-F1 | p50 ms / window | p95 ms | within limit |
|---|---|---:|---:|---:|---:|---|
| ft-deberta-select | microsoft/deberta-v3-xsmall | 70,646,404 | 0.937 | 65.5 | 76.0 | yes |
| ft-distilbert-select | distilbert-base-cased | 65,195,524 | 0.901 | 51.2 | 53.6 | yes |


## m5-finetuned

Results at commit `20eb57a61d0a`, 2026-10-02T18:12:49+00:00.
Sets: V1=4f22b6469bbd, V2=574867bf0d4d, V3=07db0ab68315 (built from seed 1 at commit 2d3885bd31a2)

| set | system | gaps | macro-F1 | classes | break-F1 | JOIN F1 | PARA F1 | wrong-join /1k | damage | str≠raw | str≠norm | para match |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| V1 | identity | 123610 | 0.418 | JOIN,SPACE,NL,PARA | 0.346 | 0.000 | 0.424 | 0.00 | 0.0000 | 0.603 | 0.000 | 0.161 |
| V1 | rules | 123610 | 0.635 | JOIN,SPACE,NL,PARA | 0.513 | 0.682 | 0.410 | 0.00 | 0.0186 | 0.863 | 0.846 | 0.145 |
| V1 | scratch | 123610 | 0.922 | JOIN,SPACE,NL,PARA | 0.885 | 0.974 | 0.861 | 0.11 | 0.0365 | 0.903 | 0.903 | 0.656 |
| V1 | finetuned | 123610 | 0.954 | JOIN,SPACE,NL,PARA | 0.940 | 0.995 | 0.899 | 0.02 | 0.0384 | 0.891 | 0.891 | 0.747 |
| V1 | finetuned-ablation | 123610 | 0.839 | JOIN,SPACE,NL,PARA | 0.763 | 0.912 | 0.749 | 0.68 | 0.0337 | 0.949 | 0.949 | 0.428 |
| V2 | identity | 2401 | 0.753 | SPACE,NL,PARA | 0.714 | 0.000 | 0.864 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.555 |
| V2 | rules | 2401 | 0.806 | SPACE,NL,PARA | 0.869 | 0.000 | 0.776 | 0.00 | 0.0604 | 0.931 | 0.931 | 0.526 |
| V2 | scratch | 2401 | 0.733 | SPACE,NL,PARA | 0.737 | 0.000 | 0.785 | 9.16 | 0.0804 | 1.000 | 1.000 | 0.453 |
| V2 | finetuned | 2401 | 0.898 | SPACE,NL,PARA | 0.915 | 0.000 | 0.886 | 4.58 | 0.0641 | 0.966 | 0.966 | 0.708 |
| V2 | finetuned-ablation | 2401 | 0.614 | SPACE,NL,PARA | 0.547 | 0.000 | 0.548 | 6.66 | 0.0979 | 1.000 | 1.000 | 0.226 |
| V3 | identity | 9187 | 1.000 | SPACE,NL,PARA | 1.000 | 0.000 | 1.000 | 0.00 | 0.0000 | 0.000 | 0.000 | 1.000 |
| V3 | rules | 9187 | 0.940 | SPACE,NL,PARA | 0.988 | 0.000 | 0.933 | 0.00 | 0.0026 | 0.170 | 0.170 | 0.921 |
| V3 | scratch | 9187 | 0.975 | SPACE,NL,PARA | 0.971 | 0.000 | 0.988 | 0.00 | 0.0016 | 0.080 | 0.080 | 0.959 |
| V3 | finetuned | 9187 | 0.991 | SPACE,NL,PARA | 0.986 | 0.000 | 0.985 | 0.00 | 0.0008 | 0.070 | 0.070 | 0.974 |
| V3 | finetuned-ablation | 9187 | 0.926 | SPACE,NL,PARA | 0.900 | 0.000 | 0.909 | 0.44 | 0.0057 | 0.330 | 0.330 | 0.794 |


## test-sets

Results at commit `4955e06adaa2` (dirty tree), 2026-10-02T14:16:08+00:00.
Sets: T0=95d8fe63481b, T1=a3d16ebe012c, T2=11ba1ea6f18c, T3=aaceae2a74c8 (built from seed 1 at commit 2d3885bd31a2)

| set | system | gaps | macro-F1 | classes | break-F1 | JOIN F1 | PARA F1 | wrong-join /1k | damage | str≠raw | str≠norm | para match |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | identity | 29 | 0.231 | JOIN,SPACE,NL,PARA | 0.000 | 0.000 | 0.000 | 0.00 | 0.0000 | 1.000 | 0.000 | 0.000 |
| T0 | rules | 29 | 1.000 | JOIN,SPACE,NL,PARA | 1.000 | 1.000 | 1.000 | 0.00 | 0.1724 | 1.000 | 1.000 | 1.000 |
| T0 | scratch | 29 | 0.620 | JOIN,SPACE,NL,PARA | 0.800 | 1.000 | 0.500 | 0.00 | 0.2069 | 1.000 | 1.000 | 0.000 |
| T1 | identity | 120009 | 0.426 | JOIN,SPACE,NL,PARA | 0.356 | 0.000 | 0.455 | 0.00 | 0.0000 | 0.594 | 0.000 | 0.187 |
| T1 | rules | 120009 | 0.627 | JOIN,SPACE,NL,PARA | 0.508 | 0.684 | 0.436 | 0.00 | 0.0188 | 0.846 | 0.823 | 0.174 |
| T1 | scratch | 120009 | 0.921 | JOIN,SPACE,NL,PARA | 0.884 | 0.968 | 0.861 | 0.13 | 0.0382 | 0.869 | 0.869 | 0.667 |
| T2 | identity | 3645 | 0.487 | JOIN,SPACE,NL,PARA | 0.587 | 0.000 | 0.708 | 0.00 | 0.0000 | 0.000 | 0.000 | 0.368 |
| T2 | rules | 3645 | 0.498 | JOIN,SPACE,NL,PARA | 0.760 | 0.000 | 0.667 | 0.00 | 0.0601 | 0.975 | 0.975 | 0.382 |
| T2 | scratch | 3645 | 0.536 | JOIN,SPACE,NL,PARA | 0.669 | 0.077 | 0.651 | 6.58 | 0.0829 | 1.000 | 1.000 | 0.375 |
| T3 | identity | 18392 | 1.000 | SPACE,NL,PARA | 1.000 | 0.000 | 1.000 | 0.00 | 0.0000 | 0.000 | 0.000 | 1.000 |
| T3 | rules | 18392 | 0.811 | SPACE,NL,PARA | 0.978 | 0.000 | 0.759 | 0.00 | 0.0102 | 0.190 | 0.190 | 0.893 |
| T3 | scratch | 18392 | 0.977 | SPACE,NL,PARA | 0.968 | 0.000 | 0.982 | 0.11 | 0.0023 | 0.140 | 0.140 | 0.929 |


## Training runs

| run | commit | class weights | best epoch / run | V1 macro-F1 | V2 macro-F1 | V3 damage | params | device | minutes |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| finetuned-ablation | `6093ac8f9125` |  | 3 / 3 | 0.839 |  | 0.0057 | 70,646,404 | cuda | 15.4 |
| finetuned | `6093ac8f9125` |  | 3 / 3 | 0.954 |  | 0.0008 | 70,646,404 | cuda | 15.0 |
| ft-deberta-select | `6093ac8f9125` |  | 1 / 1 | 0.937 |  | 0.0023 | 70,646,404 | cuda | 3.8 |
| ft-distilbert-select | `6093ac8f9125` |  | 1 / 1 | 0.901 |  | 0.0024 | 65,195,524 | cuda | 2.4 |
| scratch-v1-inverse | `dd2a39b955d0` | inverse | 3 / 5 | 0.741 |  | 0.0447 | 5,551,692 | cuda | 5.2 |
| scratch-v1 | `dd2a39b955d0` | none | 8 / 8 | 0.922 |  | 0.0016 | 5,551,692 | cuda | 8.6 |


## Service benchmark

Latency is one request at a time; throughput is eight concurrent requests of 2,000 characters. In-process rows are measured on the named machine; http rows go through a running server and carry no size or memory figures. Rows labelled `container-*` were measured through HTTP against the image running in Docker Desktop's Linux VM on the same M1 Mac; the design's latency rule is evaluated on the host CPU rows.

| label | system | mode | device | disk MB | RSS MB | p50 / p95 ms @500 | @2,000 | @10,000 | chars/s (batch 8) | commit |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| container-rules | rules | http | cpu | - | - | 1.4 / 4.3 | 1.9 / 3.1 | 4.2 / 5.0 | 1,502,660 | `3ef9677a888b` |
| container-scratch | scratch | http | cpu | - | - | 54.7 / 61.8 | 304.1 / 318.5 | 1977.4 / 2056.9 | 23,563 | `3ef9677a888b` |
| m1-mac-cpu | identity | in-process | cpu | 0.0 | 24 | 0.1 / 0.1 | 0.3 / 0.3 | 1.5 / 1.6 | 6,976,161 | `610ca964767e` |
| m1-mac-cpu | rules | in-process | cpu | 0.2 | 32 | 0.1 / 0.1 | 0.5 / 0.5 | 2.7 / 2.7 | 4,354,047 | `610ca964767e` |
| m1-mac-cpu | scratch | in-process | cpu | 22.6 | 297 | 7.3 / 7.5 | 40.2 / 42.0 | 267.8 / 283.9 | 79,701 | `610ca964767e` |
| m1-mac-cpu | finetuned | in-process | cpu | 290.9 | 662 | 28.8 / 34.5 | 92.8 / 96.4 | 807.9 / 855.7 | 24,898 | `610ca964767e` |
