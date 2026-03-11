from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def _check_imports() -> None:
    import numpy  # noqa: F401
    import pandas  # noqa: F401
    import sklearn  # noqa: F401
    import torch  # noqa: F401


def _check_paths(root: Path) -> list[str]:
    required = [
        root / "scripts" / "infer_stream_pi.py",
        root / "configs" / "base.yaml",
        root / "src" / "telemetry_ad" / "orchestration.py",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return missing


def _check_tailscale_status() -> str:
    try:
        out = subprocess.check_output(["tailscale", "status"], text=True, stderr=subprocess.STDOUT, timeout=8)
        first_line = out.strip().splitlines()[0] if out.strip() else "tailscale status returned no output"
        return f"ok: {first_line}"
    except (FileNotFoundError, subprocess.SubprocessError):
        return "not_available_or_not_running"


def _check_url(url: str) -> tuple[bool, str]:
    try:
        with urlopen(url, timeout=8) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
        return True, payload[:200]
    except URLError as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Raspberry Pi preflight for telemetry-ad deployment")
    parser.add_argument("--api-base-url", default=None, help="Example: http://100.x.y.z:8000")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    print(f"[preflight] repo_root={root}")
    print(f"[preflight] python={sys.executable}")
    print(f"[preflight] version={sys.version.split()[0]}")

    try:
        _check_imports()
        print("[preflight] imports=ok")
    except Exception as exc:  # pragma: no cover
        print(f"[preflight][error] imports failed: {exc}")
        return 1

    missing = _check_paths(root)
    if missing:
        print(f"[preflight][error] missing_files={json.dumps(missing, indent=2)}")
        return 1
    print("[preflight] required_files=ok")

    print(f"[preflight] tailscale={_check_tailscale_status()}")

    if args.api_base_url:
        base = args.api_base_url.rstrip("/")
        for endpoint in ("/health", "/stream/next?batch_size=1"):
            ok, msg = _check_url(base + endpoint)
            tag = "ok" if ok else "error"
            print(f"[preflight][{tag}] GET {base + endpoint} -> {msg}")
            if not ok:
                return 2

    print("[preflight] completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
