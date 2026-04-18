from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from scarcity.simulation.parameters import ExternalParams
from scarcity.simulation.parameters import InputOutputParams
from scarcity.simulation.types import SECTORS, Sector


@dataclass(frozen=True)
class ForeignComputation:
    exports: dict[Sector, float]
    imports: dict[Sector, float]
    imports_intermediate: dict[Sector, float]
    remittances: float
    aid: float
    CA: float
    KA: float
    BOP: float
    delta_RES_fx: float
    E_nom_new: float
    E_real_new: float


def compute_foreign_block(
    Y: Dict[Sector, float],
    Y_base: Dict[Sector, float],
    Y_gross: Dict[Sector, float],
    P: Dict[Sector, float],
    E_nom_prev: float,
    P_world: float,
    Y_world: float,
    world_demand_shock: float,
    remittance_shock: float,
    aid_shock: float,
    demand_shift: Dict[Sector, float],
    trade_disruption: Dict[Sector, float],
    fx_pressure_from_coupling: float,
    i_external: float,
    B_foreign_prev: float,
    delta_B_foreign: float,
    delta_RES_fx_intervention: float,
    params: ExternalParams,
    io_params: InputOutputParams,
    Y_gdp_nominal: float,
    scale_factor: float,
) -> ForeignComputation:
    """Compute exports/imports, CA/KA, BOP, managed-float FX dynamics, and reserves."""
    EX_base_total = float(params.export_gdp_ratio) * float(Y_gdp_nominal)
    IM_base_total = float(params.import_gdp_ratio) * float(Y_gdp_nominal)

    exports: Dict[Sector, float] = {}
    imports: Dict[Sector, float] = {}
    imports_int: Dict[Sector, float] = {}

    epsilon_export = 1.0
    for s in SECTORS:
        EX_base_s = EX_base_total * float(params.export_composition[s])
        price_term_ex = (max(float(P[s]), 1e-12) / max(float(E_nom_prev) * float(P_world), 1e-12)) ** (-float(params.eta_export[s]))
        exports[s] = (
            EX_base_s
            * price_term_ex
            * (max(float(Y_world), 1e-12) ** epsilon_export)
            * float(world_demand_shock)
            * float(trade_disruption[s])
        )

        IM_base_s = IM_base_total * float(io_params.import_content[s])
        price_term_im = (max(float(E_nom_prev) * float(P_world), 1e-12) / max(float(P[s]), 1e-12)) ** (-float(params.eta_import[s]))
        income_term_im = (max(float(Y[s]), 1e-12) / max(float(Y_base[s]), 1e-12)) ** float(params.epsilon_import[s])
        imports[s] = IM_base_s * price_term_im * income_term_im * float(demand_shift[s])

        IM_int_s = (
            float(io_params.import_content[s])
            * sum(float(io_params.io_matrix[s][j]) * float(Y_gross[s]) for j in SECTORS)
            * float(E_nom_prev)
            * float(P_world)
            / max(float(P[s]), 1e-12)
        )
        imports_int[s] = max(IM_int_s, 0.0)

    REM = float(params.remittances_gdp_ratio) * float(Y_gdp_nominal) * float(remittance_shock)
    AID = float(params.aid_gdp_ratio) * float(Y_gdp_nominal) * float(aid_shock)

    CA = (
        sum(exports[s] - imports[s] - imports_int[s] for s in SECTORS)
        + REM
        + AID
        - float(i_external) * float(B_foreign_prev) * float(E_nom_prev)
    )

    capital_flight = float(fx_pressure_from_coupling) * 1.0
    KA = float(delta_B_foreign) * float(E_nom_prev) - capital_flight
    BOP = CA + KA
    delta_RES_fx = BOP / max(float(E_nom_prev), 1e-12) + float(delta_RES_fx_intervention)

    E_pressure = -BOP / max(float(scale_factor), 1e-12)
    E_intervention = float(delta_RES_fx_intervention) * float(E_nom_prev) / max(float(scale_factor), 1e-12)
    delta_E_nom = E_pressure - E_intervention + float(fx_pressure_from_coupling)
    delta_E_nom = min(max(delta_E_nom, -0.10), 0.10)
    E_nom_new = max(float(E_nom_prev) * (1.0 + delta_E_nom), 1e-9)

    # In absence of explicit foreign CPI path, P_world serves as external deflator proxy.
    E_real_new = E_nom_new * max(float(P_world), 1e-12)

    return ForeignComputation(
        exports=exports,
        imports=imports,
        imports_intermediate=imports_int,
        remittances=float(REM),
        aid=float(AID),
        CA=float(CA),
        KA=float(KA),
        BOP=float(BOP),
        delta_RES_fx=float(delta_RES_fx),
        E_nom_new=float(E_nom_new),
        E_real_new=float(E_real_new),
    )
