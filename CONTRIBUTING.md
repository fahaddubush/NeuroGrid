# Contributing

1. Create a focused branch.
2. Keep generated data, trained models, logs, and local environments out of Git.
3. Add tests for every correctness or protocol change.
4. Run `python scripts/verify_release.py`, `python -m ruff check --select E9,F63,F7,F82 src tests scripts`, and
   `python -m pytest -q -p no:cacheprovider` before opening a pull request.
5. Update architecture and security documentation when protocol guarantees change.

Never commit `.env`, household-level raw data, TLS private keys, SQLite state, or model run
directories.
