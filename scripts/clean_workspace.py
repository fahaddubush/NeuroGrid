#!/usr/bin/env python3
"""
clean_workspace.py

Removes transient simulation logs, artifacts, and caching directories
to keep the repository hygienic.
"""

import argparse
import os
import shutil
from pathlib import Path

def clean(dry_run: bool = False):
    root = Path(__file__).resolve().parent.parent

    # Directories to clear (removes the directory and its contents)
    dirs_to_remove = [
        "data/district_logs",
        "artifacts",
        "logs",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    ]

    # Directories to empty (keeps the directory but removes contents)
    dirs_to_empty = [
        "docs/system_memory/episodic",
        "scratch",
    ]

    print("Cleaning workspace...")

    def remove_readonly(func, path, _):
        os.chmod(path, 0o777)
        func(path)

    for d in dirs_to_remove:
        path = root / d
        if path.exists() and path.is_dir():
            print(f"{'Would remove' if dry_run else 'Removed'} directory: {d}")
            if not dry_run:
                shutil.rmtree(path, onerror=remove_readonly)

    for d in dirs_to_empty:
        path = root / d
        if path.exists() and path.is_dir():
            for item in list(path.iterdir()):
                if dry_run:
                    print(f"Would remove: {item.relative_to(root)}")
                    continue
                if item.is_dir():
                    shutil.rmtree(item, onerror=remove_readonly)
                else:
                    try:
                        item.unlink()
                    except PermissionError:
                        os.chmod(item, 0o777)
                        item.unlink()
            print(f"{'Would empty' if dry_run else 'Emptied'} directory: {d}")

    # Bytecode caches occur at every package depth, not only src/ and tests/.
    for path in root.rglob("__pycache__"):
        if any(part in {".git", ".claude", ".venv"} for part in path.parts):
            continue
        print(f"{'Would remove' if dry_run else 'Removed'} cache: {path.relative_to(root)}")
        if not dry_run:
            shutil.rmtree(path, onerror=remove_readonly)

    print("Workspace clean complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    clean(dry_run=parser.parse_args().dry_run)
