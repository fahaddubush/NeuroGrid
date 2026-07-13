# Security Policy

NeuroGrid processes household energy telemetry and federated model updates. Treat both as
sensitive data.

## Supported configuration

- Insecure gRPC is restricted to loopback addresses by default.
- Non-loopback deployments must configure mutual TLS using the variables documented in
  `.env.example`.
- Certificate common names or subject alternative names must match the declared agent ID.
- Remote LLM endpoints are disabled by default. If explicitly enabled, HTTPS is mandatory.
- Federated payloads are size-limited, schema-hashed, versioned, and loaded with PyTorch's
  restricted `weights_only` loader.
- Legacy pickled scaler artifacts are rejected unless a trusted operator explicitly opts in.

## Reporting a vulnerability

Do not open a public issue containing household data, credentials, or an exploitable proof of
concept. Contact the repository owner privately with reproduction steps and affected versions.

## Scope limitations

The project implements authenticated transport and robust aggregation primitives, but it does
not claim cryptographic secure aggregation: District servers can inspect individual updates.
Round-level Gaussian output perturbation is also not equivalent to per-example DP-SGD.

## Dependency-audit exceptions

Continuous integration treats dependency-audit findings as blocking except for
`PYSEC-2026-139` and `GHSA-rrmf-rvhw-rf47`. Both affect PyTorch and currently have no
published fixed release in the audit database. These exceptions are explicit and narrowly
scoped; they should be removed as soon as a compatible fixed PyTorch release is available.
