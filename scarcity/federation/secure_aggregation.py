"""
Cryptographic secure aggregation (pairwise mask protocol).

Implements a Bonawitz-style pairwise masking scheme with:
- Long-term identity signing keys (Ed25519)
- Per-round ephemeral DH keys (X25519)
- Pairwise masks derived from shared secrets
- Dropout handling via mask reveal for dropped peers

Note: This implementation is intended for cross-silo settings and assumes a
trusted coordinator for key distribution. It is designed to be transport-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING
import base64
import hashlib

import numpy as np

CRYPTO_AVAILABLE = True
try:  # pragma: no cover - optional dependency
    from cryptography.hazmat.primitives.asymmetric import ed25519, x25519
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes, serialization
except Exception:  # pragma: no cover
    CRYPTO_AVAILABLE = False
    if TYPE_CHECKING:  # pragma: no cover
        from cryptography.hazmat.primitives.asymmetric import ed25519, x25519  # type: ignore
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF  # type: ignore
        from cryptography.hazmat.primitives import hashes, serialization  # type: ignore


def _require_crypto() -> None:
    if not CRYPTO_AVAILABLE:
        raise RuntimeError(
            "cryptography is required for secure aggregation. "
            "Install with `pip install cryptography` or the federation extra."
        )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _b64d(data: str) -> bytes:
    return base64.b64decode(data.encode("ascii"))


def _pair_id(a: str, b: str) -> str:
    return f"{a}:{b}" if a < b else f"{b}:{a}"


def _mask_sign(self_id: str, peer_id: str) -> float:
    return 1.0 if self_id < peer_id else -1.0


def _derive_seed(shared_secret: bytes, round_id: str, pair_key: str) -> int:
    _require_crypto()
    salt = hashlib.sha256(f"{round_id}:{pair_key}".encode("utf-8")).digest()
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=b"scarcity-secagg",
    )
    seed_bytes = hkdf.derive(shared_secret)
    return int.from_bytes(seed_bytes, "big", signed=False)


def _mask_vector(seed: int, shape: Tuple[int, ...]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 1.0, size=shape).astype(np.float32)


@dataclass
class IdentityKeyPair:
    private_key: Any

    @classmethod
    def generate(cls) -> "IdentityKeyPair":
        _require_crypto()
        return cls(ed25519.Ed25519PrivateKey.generate())

    def sign(self, message: bytes) -> bytes:
        _require_crypto()
        return self.private_key.sign(message)

    def public_bytes(self) -> bytes:
        _require_crypto()
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @staticmethod
    def verify(public_key: bytes, message: bytes, signature: bytes) -> bool:
        _require_crypto()
        try:
            key = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
            key.verify(signature, message)
            return True
        except Exception:
            return False


@dataclass
class EphemeralKeyPair:
    private_key: Any

    @classmethod
    def generate(cls) -> "EphemeralKeyPair":
        _require_crypto()
        return cls(x25519.X25519PrivateKey.generate())

    def public_bytes(self) -> bytes:
        _require_crypto()
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def shared_secret(self, peer_public: bytes) -> bytes:
        _require_crypto()
        peer_key = x25519.X25519PublicKey.from_public_bytes(peer_public)
        return self.private_key.exchange(peer_key)


@dataclass
class EphemeralKeyRecord:
    peer_id: str
    public_key_b64: str
    signature_b64: str

    @property
    def public_bytes(self) -> bytes:
        return _b64d(self.public_key_b64)

    @property
    def signature_bytes(self) -> bytes:
        return _b64d(self.signature_b64)


class SecureAggClient:
    """
    Participant in secure aggregation.

    Holds an identity keypair and generates per-round ephemeral keys to
    derive pairwise masks. Maintains per-peer mask seeds for dropout recovery.
    """

    def __init__(self, peer_id: str, identity: IdentityKeyPair):
        self.peer_id = peer_id
        self.identity = identity
        self._ephemeral: Optional[EphemeralKeyPair] = None
        self._pair_seeds: Dict[str, bytes] = {}

    def start_round(self, round_id: str) -> EphemeralKeyRecord:
        self._ephemeral = EphemeralKeyPair.generate()
        pub = self._ephemeral.public_bytes()
        message = round_id.encode("utf-8") + pub
        signature = self.identity.sign(message)
        return EphemeralKeyRecord(
            peer_id=self.peer_id,
            public_key_b64=_b64(pub),
            signature_b64=_b64(signature),
        )

    def build_masked_update(
        self,
        update: np.ndarray,
        round_id: str,
        peers: Sequence[EphemeralKeyRecord],
        identity_registry: Mapping[str, bytes],
    ) -> np.ndarray:
        if self._ephemeral is None:
            raise RuntimeError("Ephemeral key not initialized for round.")

        shape = update.shape
        masked = update.astype(np.float32, copy=True)

        for record in peers:
            if record.peer_id == self.peer_id:
                continue
            peer_public = identity_registry.get(record.peer_id)
            if not peer_public:
                raise RuntimeError(f"Missing identity for peer {record.peer_id}")
            message = round_id.encode("utf-8") + record.public_bytes
            if not IdentityKeyPair.verify(peer_public, message, record.signature_bytes):
                raise RuntimeError(f"Invalid signature for peer {record.peer_id}")

            shared = self._ephemeral.shared_secret(record.public_bytes)
            pair_key = _pair_id(self.peer_id, record.peer_id)
            seed = _derive_seed(shared, round_id, pair_key)
            self._pair_seeds[record.peer_id] = seed.to_bytes(32, "big", signed=False)

            mask = _mask_vector(seed, shape)
            masked += _mask_sign(self.peer_id, record.peer_id) * mask

        return masked

    def reveal_mask_seeds(self, dropped_peers: Iterable[str]) -> Dict[str, str]:
        reveals: Dict[str, str] = {}
        for peer_id in dropped_peers:
            seed = self._pair_seeds.get(peer_id)
            if seed is not None:
                reveals[peer_id] = _b64(seed)
        return reveals


class SecureAggCoordinator:
    """
    Coordinator for secure aggregation rounds.

    Tracks identity keys, collects signed ephemeral keys, and performs dropout
    unmasking using revealed pairwise mask seeds.
    """

    def __init__(self, identity_registry: Mapping[str, bytes]):
        self.identity_registry = dict(identity_registry)

    def verify_record(self, record: EphemeralKeyRecord, round_id: str) -> None:
        public = self.identity_registry.get(record.peer_id)
        if not public:
            raise RuntimeError(f"Missing identity for peer {record.peer_id}")
        message = round_id.encode("utf-8") + record.public_bytes
        if not IdentityKeyPair.verify(public, message, record.signature_bytes):
            raise RuntimeError(f"Invalid signature for peer {record.peer_id}")

    def unmask_for_dropouts(
        self,
        aggregate_sum: np.ndarray,
        round_id: str,
        dropped_peers: Sequence[str],
        reveal_map: Mapping[str, Mapping[str, str]],
    ) -> np.ndarray:
        if not dropped_peers:
            return aggregate_sum
        adjusted = aggregate_sum.astype(np.float32, copy=True)
        for survivor_id, reveals in reveal_map.items():
            for dropped_id in dropped_peers:
                seed_b64 = reveals.get(dropped_id)
                if not seed_b64:
                    continue
                seed = int.from_bytes(_b64d(seed_b64), "big", signed=False)
                mask = _mask_vector(seed, adjusted.shape)
                # The survivor contributed sign(self, dropped) * mask
                adjusted -= _mask_sign(survivor_id, dropped_id) * mask
        return adjusted


__all__ = [
    "IdentityKeyPair",
    "EphemeralKeyPair",
    "EphemeralKeyRecord",
    "SecureAggClient",
    "SecureAggCoordinator",
]
