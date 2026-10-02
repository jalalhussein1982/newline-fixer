"""Build the image, start it, wait for health, post the challenge example, check, stop (design 7).
Usage: uv run python scripts/container_check.py [--image newline-fixer:local] [--port 8000] [--model rules] [--no-build]
Exits 0 only when the example round-trips through the container.
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
from newline_fixer.text import normalize


def sh(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def wait_healthy(base: str, seconds: float) -> None:
    deadline = time.time() + seconds
    last = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/healthz", timeout=2) as r:
                if r.status == 200:
                    return
        except urllib.error.HTTPError as e:
            last = f"{e.code} {e.read().decode(errors='replace')}"
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            last = str(e)
        time.sleep(1)
    raise SystemExit(f"container never became healthy; last answer: {last}")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--image", default="newline-fixer:local")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--model", default="rules")
    p.add_argument("--no-build", action="store_true")
    a = p.parse_args()
    if not a.no_build:
        subprocess.run(["docker", "build", "-t", a.image, "."], check=True)
    cid = sh("docker", "run", "-d", "-p", f"{a.port}:8000", "-e", f"NF_MODEL={a.model}", a.image)
    base = f"http://127.0.0.1:{a.port}"
    try:
        wait_healthy(base, 120)
        req = urllib.request.Request(
            f"{base}/v1/fix",
            data=json.dumps({"text": EXAMPLE_INPUT}).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read())
        expected = normalize(EXAMPLE_OUTPUT)
        if body["text"] != expected:
            print("MISMATCH\n--- got ---\n" + body["text"] + "\n--- expected ---\n" + expected)
            sys.exit(1)
        print(
            f"PASS: {a.image} ({body['stats']['model']}) reproduces the example in {body['stats']['latency_ms']} ms"
        )
    finally:
        logs = subprocess.run(
            ["docker", "logs", "--tail", "5", cid], capture_output=True, text=True
        )
        print(logs.stdout + logs.stderr)
        subprocess.run(["docker", "rm", "-f", cid], check=False, capture_output=True)


if __name__ == "__main__":
    main()
