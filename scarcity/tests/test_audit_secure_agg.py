import numpy as np
import pytest

from scarcity.federation.secure_aggregation import IdentityKeyPair, SecureAggClient, SecureAggCoordinator


@pytest.mark.slow
def test_secure_aggregation_dropout_unmask():
    pytest.importorskip("cryptography")
    ids = {pid: IdentityKeyPair.generate() for pid in ["a", "b", "c"]}
    registry = {pid: key.public_bytes() for pid, key in ids.items()}
    clients = {pid: SecureAggClient(pid, key) for pid, key in ids.items()}
    round_id = "r1"
    records = [c.start_round(round_id) for c in clients.values()]
    updates = {pid: np.array([i + 1.0, i + 2.0], dtype=np.float32) for i, pid in enumerate(clients)}
    masked = {pid: clients[pid].build_masked_update(updates[pid], round_id, records, registry) for pid in ["a", "b"]}
    agg = masked["a"] + masked["b"]
    reveals = {pid: clients[pid].reveal_mask_seeds(["c"]) for pid in ["a", "b"]}
    unmasked = SecureAggCoordinator(registry).unmask_for_dropouts(agg, round_id, ["c"], reveals)
    assert np.allclose(unmasked, updates["a"] + updates["b"])
