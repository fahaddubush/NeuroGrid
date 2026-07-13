"""Tests for portable path resolution (Item 6)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from src.utils.paths import ensure_dir, resolve_path, workspace_root


class TestWorkspaceRoot(unittest.TestCase):
    def test_default_is_repo_root(self):
        # repo_root/src/utils/paths.py → repo_root has src/, tests/, README.md.
        root = workspace_root()
        self.assertTrue((root / "src").is_dir())
        self.assertTrue((root / "tests").is_dir())

    def test_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("WORKSPACE_ROOT")
            os.environ["WORKSPACE_ROOT"] = tmp
            try:
                self.assertEqual(workspace_root().resolve(), Path(tmp).resolve())
            finally:
                if old is None:
                    os.environ.pop("WORKSPACE_ROOT", None)
                else:
                    os.environ["WORKSPACE_ROOT"] = old


class TestResolvePath(unittest.TestCase):
    def test_absolute_pass_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            abs_path = Path(tmp).resolve()
            self.assertEqual(resolve_path(str(abs_path)), abs_path)

    def test_relative_joins_workspace(self):
        rel = "data/district_logs"
        out = resolve_path(rel)
        self.assertTrue(out.is_absolute())
        self.assertTrue(str(out).endswith(os.path.normpath("data/district_logs")))

    def test_none_returns_none(self):
        self.assertIsNone(resolve_path(None))


class TestEnsureDir(unittest.TestCase):
    def test_creates_under_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("WORKSPACE_ROOT")
            os.environ["WORKSPACE_ROOT"] = tmp
            try:
                p = ensure_dir("artifacts/test_subdir")
                self.assertTrue(p.is_dir())
                self.assertTrue(str(p).startswith(str(Path(tmp).resolve())))
            finally:
                if old is None:
                    os.environ.pop("WORKSPACE_ROOT", None)
                else:
                    os.environ["WORKSPACE_ROOT"] = old


if __name__ == "__main__":
    unittest.main()
