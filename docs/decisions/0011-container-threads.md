# 0011. Container thread count: torch's default stays

Date: 2026-10-02. Status: accepted.

## Context

Decision 0010 served `finetuned` on a host figure of 92.8 ms p50 at 2,000 characters (M1 CPU,
in-process), and its postscript recorded 342.9 ms p50 (p95 585.1) for the same model through
HTTP inside Docker Desktop's Linux VM, which has 8 vCPUs here. The 3.7 times gap was not
explained. The untested hypothesis was thread oversubscription: torch sizes its intra-op pool
from the vCPU count it sees, and in the VM those vCPUs are shared with the host. The Hugging
Face Space planned in M6 has 2 vCPUs, so a setting that helps here may matter more there.

`NF_TORCH_THREADS` (positive integer, unset by default) was added to the service settings and
applied with `torch.set_num_threads` before any model is built, so the hypothesis could be
tested in the image itself.

## Options

1. Leave torch's default thread count (what the image did until now).
2. Set a fixed thread count in the image (`ENV NF_TORCH_THREADS=<n>`).
3. Leave the image alone and make the count a Space variable only.

## Decision

Option 1: the image does not set `NF_TORCH_THREADS`. The rule fixed before measuring was to
set the value with the lowest p50 at 2,000 characters if it beat the default by more than
10%. It did not. Same image rebuilt at commit 3ea4171, same idle M1 host, one container per
setting, `scripts/bench.py --url`, records in `experiments/bench/`:

| threads | p50 ms @2,000 | p95 ms @2,000 | chars/s (batch 8) | record |
|---|---|---|---|---|
| default (torch, 8 vCPUs) | 342.9 | 585.1 | 9,458 | `container-finetuned.json` |
| 1 | 460.6 | 493.2 | 11,429 | `container-finetuned-threads-1.json` |
| 2 | 388.1 | 457.4 | 7,983 | `container-finetuned-threads-2.json` |
| 4 | 326.7 | 400.9 | 8,180 | `container-finetuned-threads-4.json` |

The best p50 is 4 threads at 326.7 ms, 4.7% under the default's 342.9 ms, inside the 10%
margin and inside the run-to-run spread seen earlier for this model (the M1 host p50 moved
from 97.1 to 92.8 ms between two runs of the same code). The default row was measured with
the M5 image on a different day, so even that 4.7% is not a controlled comparison, and a slower same-day default run would raise the 4.7% margin; that case was not excluded. The threads-2 and threads-4 records are flagged dirty because the bench script saw the previous record unstaged; the code under test was the same commit. The
oversubscription hypothesis did not hold as the main cause: shrinking the pool to one thread
costs 34% at 2,000 characters, so the model really uses the cores, and the 3.7 times gap to
the host is the VM (virtualised CPU, no direct Apple-silicon performance cores), which points to the VM rather than wasted threads, though this was not isolated. Two weaker signals, not used for the decision: the p95 is tighter at 2 and 4
threads (457 and 401 ms against 585), and one thread gives the best throughput under eight
concurrent requests (11,429 against 9,458 chars/s), as expected when requests, not threads,
share the cores.

## Consequences

- The image is unchanged (no `ENV NF_TORCH_THREADS`). The setting stays available and
  documented, so the Space can override it with a variable, which is option 3 as a fallback
  rather than as the policy.
- The Space has 2 vCPUs, a different regime from this 8-vCPU VM; its p50 is measured in Task
  3 of M6 and the thread count is revisited there only if that figure is poor (2 threads at
  388 ms here is a hint, not evidence for the Space).
- ONNX export stays a non-goal unless the Space p50 at 2,000 characters exceeds 1,000 ms.
- The thread sweep is single-run per setting; a 10% effect would need repeats to resolve,
  which is why the decision rule demanded a margin.

## Postscript (2026-10-02, M6 Task 3)

On the Space (free `cpu-basic`, 2 vCPUs, same image, default threads) the measured p50 at 2,000 characters is 342.2 ms (p95 379.0 ms), over HTTP from the author's Mac (`experiments/bench/space.json`, commit 4819a84). That is far under the 1,000 ms ONNX threshold, so ONNX stays out of scope and the thread count is not revisited.
