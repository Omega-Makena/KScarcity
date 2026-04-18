from __future__ import annotations

from scarcity.simulation.parameters import AllParams
from scarcity.simulation.production import (
    compute_gross_output,
    compute_intermediate_demand,
    compute_marginal_products,
)
from scarcity.simulation.types import SECTORS, Sector, ShockVector


def test_ces_output_positive_for_positive_inputs() -> None:
    params = AllParams.default_kenya()
    shocks = ShockVector.neutral()

    k = {s: 100.0 for s in SECTORS}
    n = {s: 10.0 for s in SECTORS}
    y_gross = compute_gross_output(k, n, params.production, shocks, None)

    for s in SECTORS:
        assert y_gross[s] > 0.0


def test_cobb_douglas_limit_near_sigma_one() -> None:
    params = AllParams.default_kenya()
    shocks = ShockVector.neutral()
    sector = Sector.SERVICES

    params.production.sigma[sector] = 0.999
    k = {s: 100.0 for s in SECTORS}
    n = {s: 8.0 for s in SECTORS}
    y_ces = compute_gross_output(k, n, params.production, shocks, None)[sector]

    a = params.production.A[sector]
    alpha = params.production.alpha[sector]
    h = params.production.h[sector]
    y_cd = a * (k[sector] ** alpha) * ((h * n[sector]) ** (1.0 - alpha))

    rel_err = abs(y_ces - y_cd) / max(abs(y_cd), 1e-12)
    assert rel_err < 1e-4


def test_constant_returns_to_scale() -> None:
    params = AllParams.default_kenya()
    shocks = ShockVector.neutral()
    k = {s: 120.0 for s in SECTORS}
    n = {s: 12.0 for s in SECTORS}
    y1 = compute_gross_output(k, n, params.production, shocks, None)

    k2 = {s: 2.0 * k[s] for s in SECTORS}
    n2 = {s: 2.0 * n[s] for s in SECTORS}
    y2 = compute_gross_output(k2, n2, params.production, shocks, None)

    for s in SECTORS:
        ratio = y2[s] / max(y1[s], 1e-12)
        assert abs(ratio - 2.0) < 1e-6


def test_marginal_products_positive_and_declining() -> None:
    params = AllParams.default_kenya()
    shocks = ShockVector.neutral()

    k = {s: 100.0 for s in SECTORS}
    n = {s: 10.0 for s in SECTORS}
    y = compute_gross_output(k, n, params.production, shocks, None)
    mpk_1, mpl_1 = compute_marginal_products(y, k, n, params.production, None)

    k_hi = {s: 150.0 for s in SECTORS}
    y_k = compute_gross_output(k_hi, n, params.production, shocks, None)
    mpk_2, _ = compute_marginal_products(y_k, k_hi, n, params.production, None)

    n_hi = {s: 15.0 for s in SECTORS}
    y_n = compute_gross_output(k, n_hi, params.production, shocks, None)
    _, mpl_2 = compute_marginal_products(y_n, k, n_hi, params.production, None)

    for s in SECTORS:
        assert mpk_1[s] > 0.0
        assert mpl_1[s] > 0.0
        assert mpk_2[s] < mpk_1[s]
        assert mpl_2[s] < mpl_1[s]


def test_intermediate_demand_matrix_dimensions_and_non_negative() -> None:
    params = AllParams.default_kenya()
    shocks = ShockVector.neutral()
    k = {s: 90.0 for s in SECTORS}
    n = {s: 9.0 for s in SECTORS}
    y = compute_gross_output(k, n, params.production, shocks, None)
    intermediate = compute_intermediate_demand(y, params.io)

    assert set(intermediate.keys()) == set(SECTORS)
    for using in SECTORS:
        assert set(intermediate[using].keys()) == set(SECTORS)
        for supplying in SECTORS:
            assert intermediate[using][supplying] >= 0.0
