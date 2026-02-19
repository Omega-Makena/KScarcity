import numpy as np

from scarcity.federation.privacy_guard import PrivacyConfig, PrivacyGuard


def test_privacy_guard_noise_applied():
    np.random.seed(0)
    guard = PrivacyGuard(PrivacyConfig(dp_noise_sigma=0.5))
    values = [[1.0, 2.0], [3.0, 4.0]]
    noised = guard.apply_noise(values)
    assert not np.allclose(noised, values)
