"""Shared secure-by-default gRPC transport configuration."""
from __future__ import annotations

import ipaddress
import os
from pathlib import Path


def is_loopback(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def allow_insecure(host: str) -> bool:
    return is_loopback(host) or os.getenv("NEUROGRID_ALLOW_INSECURE", "0") == "1"


def _read_required(env_name: str) -> bytes:
    value = os.getenv(env_name)
    if not value:
        raise RuntimeError(f"{env_name} is required when NEUROGRID_TLS=1")
    path = Path(value).expanduser().resolve()
    return path.read_bytes()


def client_credentials(grpc_module):
    if os.getenv("NEUROGRID_TLS", "0") != "1":
        return None
    root = _read_required("NEUROGRID_CA_CERT")
    cert = _read_required("NEUROGRID_CLIENT_CERT")
    key = _read_required("NEUROGRID_CLIENT_KEY")
    return grpc_module.ssl_channel_credentials(
        root_certificates=root, private_key=key, certificate_chain=cert
    )


def server_credentials(grpc_module):
    if os.getenv("NEUROGRID_TLS", "0") != "1":
        return None
    root = _read_required("NEUROGRID_CA_CERT")
    cert = _read_required("NEUROGRID_SERVER_CERT")
    key = _read_required("NEUROGRID_SERVER_KEY")
    return grpc_module.ssl_server_credentials(
        [(key, cert)], root_certificates=root, require_client_auth=True
    )


def authorize_agent(agent_id: str, context) -> bool:
    allowed_raw = os.getenv("NEUROGRID_ALLOWED_AGENTS", "").strip()
    if allowed_raw:
        allowed = {item.strip() for item in allowed_raw.split(",") if item.strip()}
        if agent_id not in allowed:
            return False
    if os.getenv("NEUROGRID_TLS", "0") != "1":
        return True
    if context is None:
        return False
    auth = context.auth_context() or {}
    identities = auth.get("x509_common_name", []) + auth.get("x509_subject_alternative_name", [])
    decoded = {
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in identities
    }
    return agent_id in decoded
