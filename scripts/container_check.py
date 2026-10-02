"""Build the image, start it, wait for health, post the challenge example, check, stop (design 7).
Usage: uv run python scripts/container_check.py [--image newline-fixer:local] [--port 8000] [--model rules] [--no-build] [--expect-mismatch]
       uv run python scripts/container_check.py --url https://host [--expect-mismatch]
With --url the check runs against an already running service (for example the Hugging Face
Space): no Docker command runs, --image/--port/--model/--no-build are ignored, and the health
wait is 180 s because free Spaces wake from sleep.
Exits 0 only when the example round-trips through the container.
With --expect-mismatch (for --model finetuned, decision 0010) the example is expected NOT to
round-trip exactly: the check requires health 200 and that the content (non-whitespace
characters) is preserved, and passes when the output differs from the example; an unexpected
exact match is reported but is not a failure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

from newline_fixer.example import EXAMPLE_INPUT, EXAMPLE_OUTPUT
from newline_fixer.text import content, normalize


def sh(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def wait_healthy(base: str, cid: str | None, seconds: float) -> None:
    """Poll /healthz; with a container id, fail fast when the container has exited."""
    deadline = time.monotonic() + seconds
    last = ""
    while time.monotonic() < deadline:
        if cid is not None and sh("docker", "inspect", "-f", "{{.State.Running}}", cid) != "true":
            logs = subprocess.run(
                ["docker", "logs", "--tail", "20", cid], capture_output=True, text=True
            )
            raise SystemExit(f"container exited: {logs.stdout}{logs.stderr}")
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=2) as r:
                if r.status == 200:
                    return
        except urllib.error.HTTPError as e:
            last = f"{e.code} {e.read().decode(errors='replace')}"
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last = str(e)
        time.sleep(1)
    raise SystemExit(f"service never became healthy; last answer: {last}")


def check_example(base: str, label: str, expect_mismatch: bool) -> None:
    """Post the challenge example to `base` and compare (shared by the container and URL modes)."""
    req = urllib.request.Request(
        f"{base}/v1/fix",
        data=json.dumps({"text": EXAMPLE_INPUT}).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        body = json.loads(r.read())
    expected = normalize(EXAMPLE_OUTPUT)
    if expect_mismatch:
        if content(body["text"]) != content(EXAMPLE_INPUT):
            print("CONTENT CHANGED\n--- got ---\n" + body["text"])
            sys.exit(1)
        model = body["stats"]["model"]
        if body["text"] == expected:
            print(
                f"NOTE: {label} ({model}) unexpectedly reproduces the example exactly; "
                "a match is not a failure"
            )
        else:
            print(
                f"PASS (expected mismatch): {label} ({model}) serves and preserves content; "
                "output differs from the example as decision 0010 records"
            )
        return
    if body["text"] != expected:
        print("MISMATCH\n--- got ---\n" + body["text"] + "\n--- expected ---\n" + expected)
        sys.exit(1)
    print(
        f"PASS: {label} ({body['stats']['model']}) reproduces the example in {body['stats']['latency_ms']} ms"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image", default="newline-fixer:local")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="rules")
    p.add_argument("--no-build", action="store_true")
    p.add_argument(
        "--url",
        help="check a running service at this base URL; no Docker command runs and "
        "--image/--port/--model/--no-build are ignored",
    )
    p.add_argument(
        "--expect-mismatch",
        action="store_true",
        help="expect the known one-gap difference on the example (decision 0010)",
    )
    a = p.parse_args()
    if a.url:
        base = a.url.rstrip("/")
        wait_healthy(base, None, 180)
        check_example(base, base, a.expect_mismatch)
        return
    if not a.no_build:
        subprocess.run(["docker", "build", "-t", a.image, "."], check=True)
    cid = sh("docker", "run", "-d", "-p", f"{a.port}:8000", "-e", f"NF_MODEL={a.model}", a.image)
    try:
        base = f"http://127.0.0.1:{a.port}"
        wait_healthy(base, cid, 120)
        check_example(base, a.image, a.expect_mismatch)
    finally:
        logs = subprocess.run(
            ["docker", "logs", "--tail", "5", cid], capture_output=True, text=True
        )
        print(logs.stdout + logs.stderr)
        subprocess.run(["docker", "rm", "-f", cid], check=False, capture_output=True)


if __name__ == "__main__":
    main()
