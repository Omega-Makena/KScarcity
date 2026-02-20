"""Trust policy checks for federated connectors and channels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


_ALLOWED_CHANNELS = {"mtls", "private_link", "vpn", "internal_mesh"}
_ALLOWED_ATTESTATION = {"verified", "pending", "failed", "unknown"}


@dataclass
class TrustPolicy:
    """Runtime trust constraints for decentralized coordination."""

    require_verified_attestation_for_secret: bool = True
    allowed_channels: List[str] = field(default_factory=lambda: sorted(_ALLOWED_CHANNELS))


def _looks_like_inline_secret(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in ["password", "secret", "token", "apikey", "api_key"]) and ":" in value


def validate_connector_trust(connector_payload: Dict[str, Any], policy: TrustPolicy | None = None) -> List[str]:
    policy = policy or TrustPolicy()
    issues: List[str] = []

    options = dict(connector_payload.get("options", {}))
    source_type = str(connector_payload.get("source_type", "")).lower()
    channel = str(options.get("channel_security", "mtls")).lower()
    attestation = str(options.get("attestation_status", "unknown")).lower()
    classification = str(options.get("max_classification", "INTERNAL")).upper()

    if channel not in {c.lower() for c in policy.allowed_channels}:
        issues.append(f"unsupported_channel_security:{channel}")

    if attestation not in _ALLOWED_ATTESTATION:
        issues.append(f"invalid_attestation_status:{attestation}")

    if classification == "SECRET" and policy.require_verified_attestation_for_secret:
        if attestation != "verified":
            issues.append("secret_requires_verified_attestation")

    for key, value in options.items():
        if key.lower() in {"password", "secret", "token", "api_key", "apikey"}:
            issues.append(f"inline_secret_not_allowed:{key}")
        if isinstance(value, str) and _looks_like_inline_secret(value):
            issues.append(f"inline_secret_pattern_detected:{key}")

    if source_type in {"postgres", "mysql", "oracle", "sqlserver"}:
        if not options.get("credential_ref"):
            issues.append("missing_credential_ref")

    return issues
