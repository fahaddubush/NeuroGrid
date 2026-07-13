#!/usr/bin/env python3
"""Fail when a source release contains generated, secret, or transient files."""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".venv", "artifacts", "scratch"}
FORBIDDEN_SUFFIXES = {".pyc", ".pyo", ".log", ".tmp"}
FORBIDDEN_PREFIXES = (
    ".claude/",
    "vendor/",
    "data/raw/",
    "data/processed",
    "artifacts/",
    "scratch/",
)
SENSITIVE_SUFFIXES = {".key", ".pem", ".db", ".sqlite", ".pth", ".pkl", ".parquet"}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return [Path(line) for line in result.stdout.splitlines() if line]


def main() -> int:
    failures: list[str] = []
    for relative in tracked_files():
        if any(part in FORBIDDEN_PARTS for part in relative.parts):
            failures.append(f"tracked generated path: {relative}")
        if relative.suffix.lower() in FORBIDDEN_SUFFIXES:
            failures.append(f"tracked transient file: {relative}")
        normalized = relative.as_posix()
        if normalized.startswith(FORBIDDEN_PREFIXES):
            failures.append(f"tracked generated/private prefix: {relative}")
        if relative.suffix.lower() in SENSITIVE_SUFFIXES:
            failures.append(f"tracked sensitive/generated artifact: {relative}")
        if relative.name == ".env":
            failures.append("tracked secret-bearing .env file")
    for path in [*ROOT.joinpath("src").rglob("*.py"), *ROOT.joinpath("tests").rglob("*.py")]:
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            failures.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")
    if failures:
        print("Release verification failed:")
        print("\n".join(f"- {failure}" for failure in failures))
        return 1
    print("Release verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
