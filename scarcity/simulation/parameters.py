from __future__ import annotations

from dataclasses import dataclass, field

from scarcity.simulation.types import SECTORS, Sector


def _assert_share_sum(name: str, values: dict[Sector, float], tol: float = 1e-6) -> None:
    total = float(sum(values.values()))
    if abs(total - 1.0) > tol:
        raise ValueError(f"{name} must sum to 1.0, got {total}")


@dataclass
class NationalAccountsParams:
    """2019-2023 averages from KNBS/World Bank sources."""

    gdp_share: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.218,
            Sector.MANUFACTURING: 0.164,
            Sector.SERVICES: 0.473,
            Sector.INFORMAL: 0.145,
        }
    )
    employment_share: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.338,
            Sector.MANUFACTURING: 0.120,
            Sector.SERVICES: 0.262,
            Sector.INFORMAL: 0.280,
        }
    )
    gdp_real_2023: float = 10_980.0
    labor_force_2023: float = 23.5
    employment_2023: float = 21.4
    unemployment_rate_2023: float = 0.054
    population_growth_rate: float = 0.022
    labor_force_growth_rate: float = 0.028
    tfp_growth_trend: float = 0.008

    def __post_init__(self) -> None:
        _assert_share_sum("gdp_share", self.gdp_share)
        _assert_share_sum("employment_share", self.employment_share)


@dataclass
class ProductionParams:
    """CES production by sector."""

    A: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 1.0,
            Sector.MANUFACTURING: 1.0,
            Sector.SERVICES: 1.0,
            Sector.INFORMAL: 1.0,
        }
    )
    alpha: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.25,
            Sector.MANUFACTURING: 0.45,
            Sector.SERVICES: 0.35,
            Sector.INFORMAL: 0.15,
        }
    )
    sigma: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.65,
            Sector.MANUFACTURING: 0.80,
            Sector.SERVICES: 0.90,
            Sector.INFORMAL: 0.50,
        }
    )
    h: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.60,
            Sector.MANUFACTURING: 0.80,
            Sector.SERVICES: 1.00,
            Sector.INFORMAL: 0.50,
        }
    )
    delta: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.015,
            Sector.MANUFACTURING: 0.025,
            Sector.SERVICES: 0.020,
            Sector.INFORMAL: 0.030,
        }
    )
    capital_output_ratio: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 1.8,
            Sector.MANUFACTURING: 3.2,
            Sector.SERVICES: 2.5,
            Sector.INFORMAL: 0.8,
        }
    )

    def __post_init__(self) -> None:
        for mapping_name in ("A", "alpha", "sigma", "h", "delta", "capital_output_ratio"):
            mapping = getattr(self, mapping_name)
            for sector in SECTORS:
                if sector not in mapping:
                    raise ValueError(f"{mapping_name} missing {sector}")


@dataclass
class InputOutputParams:
    """4×4 intermediate-use matrix reconciled with the KNBS 9-sector IO structure.

    The 3×3 block for (AGRICULTURE, MANUFACTURING, SERVICES) is derived from the
    9-sector Kenya 2017 SUT (via io_structure.aggregate_io_to_sfc_sectors) using
    the standard IO aggregation formula:

        A_agg[I,J] = Σ_{i∈I} Σ_{j∈J} A[i,j] · x_j / X_J

    KNBS concordance used for aggregation:
      AGRICULTURE  ← agriculture (0.228 GDP share)
      MANUFACTURING ← manufacturing (0.076) + mining (0.005)
                      + construction (0.071) + water (0.009)  → 0.161 total
      SERVICES     ← services (0.490) + health (0.024)
                      + transport (0.058) + security (0.037)  → 0.609 total

    The INFORMAL row/column is not in the KNBS SUT and retains field estimates.

    import_content is weighted by GDP share and FX pass-through sensitivity
    (higher FX sensitivity ≈ higher import dependency; anchor: manufacturing ~0.31).
    """

    io_matrix: dict[Sector, dict[Sector, float]] = field(
        default_factory=lambda: {
            # Row = sector consuming inputs; Col = sector supplying them.
            # A[i,j] = fraction of sector j's output used per unit of sector j output.
            # Column-sum check: all < 1.0 (Hawkins-Simon satisfied).
            Sector.AGRICULTURE: {
                Sector.AGRICULTURE:   0.12,   # agri self-use (seed, fodder)
                Sector.MANUFACTURING: 0.03,   # KNBS-derived: agri row × MFG block
                Sector.SERVICES:      0.04,   # KNBS-derived: agri row × SRV block
                Sector.INFORMAL:      0.02,   # field estimate (unchanged)
            },
            Sector.MANUFACTURING: {
                Sector.AGRICULTURE:   0.17,   # KNBS-derived: MFG rows × agri col
                Sector.MANUFACTURING: 0.22,   # KNBS-derived: MFG×MFG block
                Sector.SERVICES:      0.15,   # KNBS-derived: MFG×SRV block
                Sector.INFORMAL:      0.03,   # field estimate (unchanged)
            },
            Sector.SERVICES: {
                Sector.AGRICULTURE:   0.13,   # KNBS-derived: SRV rows × agri col
                Sector.MANUFACTURING: 0.21,   # KNBS-derived: SRV×MFG block
                Sector.SERVICES:      0.30,   # KNBS-derived: SRV×SRV block
                Sector.INFORMAL:      0.04,   # field estimate (unchanged)
            },
            Sector.INFORMAL: {
                Sector.AGRICULTURE:   0.10,   # field estimate (unchanged)
                Sector.MANUFACTURING: 0.08,   # field estimate (unchanged)
                Sector.SERVICES:      0.05,   # field estimate (unchanged)
                Sector.INFORMAL:      0.06,   # field estimate (unchanged)
            },
        }
    )
    import_content: dict[Sector, float] = field(
        default_factory=lambda: {
            # Fraction of intermediate inputs sourced from imports.
            # Agriculture: pure agri, low import dependency.
            # Manufacturing: weighted avg (mfg 35%, mining 45%, const 28%, water 12%)
            #   → 0.31 (down from prior 0.35; construction/water pull average down).
            # Services: weighted avg (serv 10%, health 15%, transport 20%, security 8%)
            #   → 0.11 (slightly up; transport fuel imports raise the mean).
            # Informal: field estimate (unchanged).
            Sector.AGRICULTURE:   0.150,
            Sector.MANUFACTURING: 0.310,
            Sector.SERVICES:      0.110,
            Sector.INFORMAL:      0.080,
        }
    )

    def __post_init__(self) -> None:
        for s in SECTORS:
            if s not in self.io_matrix:
                raise ValueError(f"io_matrix missing row {s}")
            for j in SECTORS:
                if j not in self.io_matrix[s]:
                    raise ValueError(f"io_matrix row {s} missing col {j}")
            row_sum = float(sum(self.io_matrix[s].values()))
            if row_sum >= 1.0:
                raise ValueError(f"io_matrix row {s} must sum to < 1.0, got {row_sum}")


@dataclass
class HouseholdParams:
    c_1: float = 0.82
    c_2: float = 0.04
    consumption_shares: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.36,
            Sector.MANUFACTURING: 0.22,
            Sector.SERVICES: 0.32,
            Sector.INFORMAL: 0.10,
        }
    )
    import_share_consumption: float = 0.18
    quintile_income_shares: list[float] = field(
        default_factory=lambda: [0.047, 0.081, 0.121, 0.190, 0.561]
    )
    food_share_by_quintile: list[float] = field(
        default_factory=lambda: [0.62, 0.52, 0.42, 0.33, 0.18]
    )
    poverty_line: float = 5_995.0

    def __post_init__(self) -> None:
        _assert_share_sum("consumption_shares", self.consumption_shares)
        if len(self.quintile_income_shares) != 5:
            raise ValueError("quintile_income_shares must have 5 elements")
        if len(self.food_share_by_quintile) != 5:
            raise ValueError("food_share_by_quintile must have 5 elements")
        if abs(sum(self.quintile_income_shares) - 1.0) > 1e-6:
            raise ValueError("quintile_income_shares must sum to 1")


@dataclass
class GovernmentParams:
    vat_rate: float = 0.16
    income_tax_effective: float = 0.12
    corporate_tax_effective: float = 0.08
    trade_tax_rate: float = 0.08

    wage_bill_share: float = 0.35
    transfers_share: float = 0.12
    interest_share: float = 0.22
    investment_share: float = 0.15
    other_recurrent_share: float = 0.16

    debt_gdp_ratio_2023: float = 0.68
    domestic_share_of_debt: float = 0.52
    avg_domestic_maturity_quarters: int = 8
    avg_external_maturity_quarters: int = 40

    deficit_target_gdp: float = -0.05
    debt_ceiling_gdp: float = 0.75
    fiscal_consolidation_speed: float = 0.1

    def __post_init__(self) -> None:
        composition_sum = (
            self.wage_bill_share
            + self.transfers_share
            + self.interest_share
            + self.investment_share
            + self.other_recurrent_share
        )
        if abs(composition_sum - 1.0) > 1e-6:
            raise ValueError("Government expenditure shares must sum to 1.0")


@dataclass
class MonetaryParams:
    i_neutral: float = 0.025
    pi_target: float = 0.0125
    phi_pi: float = 1.5
    phi_y: float = 0.5
    i_floor: float = 0.0125
    i_ceiling: float = 0.05
    smoothing: float = 0.7

    spread_loan: float = 0.015
    spread_deposit: float = -0.010
    spread_govt: float = 0.005

    fx_reserve_target_months: float = 4.5
    fx_intervention_speed: float = 0.1
    reserve_requirement: float = 0.045


@dataclass
class ExternalParams:
    eta_export: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.8,
            Sector.MANUFACTURING: 1.2,
            Sector.SERVICES: 0.6,
            Sector.INFORMAL: 0.1,
        }
    )
    eta_import: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.5,
            Sector.MANUFACTURING: 1.0,
            Sector.SERVICES: 0.7,
            Sector.INFORMAL: 0.3,
        }
    )
    epsilon_import: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.8,
            Sector.MANUFACTURING: 1.5,
            Sector.SERVICES: 1.2,
            Sector.INFORMAL: 0.5,
        }
    )

    export_gdp_ratio: float = 0.12
    import_gdp_ratio: float = 0.22
    export_composition: dict[Sector, float] = field(
        default_factory=lambda: {
            Sector.AGRICULTURE: 0.45,
            Sector.MANUFACTURING: 0.25,
            Sector.SERVICES: 0.28,
            Sector.INFORMAL: 0.02,
        }
    )
    remittances_gdp_ratio: float = 0.034
    aid_gdp_ratio: float = 0.015

    world_gdp_growth: float = 0.0075
    world_inflation: float = 0.005
    us_interest_rate: float = 0.0125

    sovereign_spread_base: float = 0.015
    E_nom_2023: float = 140.0

    def __post_init__(self) -> None:
        _assert_share_sum("export_composition", self.export_composition)


@dataclass
class BankingParams:
    loan_to_deposit_ratio: float = 0.78
    govt_securities_to_assets: float = 0.25
    capital_adequacy_ratio: float = 0.172
    npl_ratio_2023: float = 0.149

    credit_growth_sensitivity_to_output_gap: float = 1.5
    npl_sensitivity_to_unemployment: float = 2.0
    provision_rate: float = 0.5
    min_capital_adequacy: float = 0.145
    dividend_payout_ratio: float = 0.30

    max_leverage_ratio: float = 10.0
    credit_rationing_threshold: float = 0.12


@dataclass
class AllParams:
    national_accounts: NationalAccountsParams = field(default_factory=NationalAccountsParams)
    production: ProductionParams = field(default_factory=ProductionParams)
    io: InputOutputParams = field(default_factory=InputOutputParams)
    households: HouseholdParams = field(default_factory=HouseholdParams)
    government: GovernmentParams = field(default_factory=GovernmentParams)
    monetary: MonetaryParams = field(default_factory=MonetaryParams)
    external: ExternalParams = field(default_factory=ExternalParams)
    banking: BankingParams = field(default_factory=BankingParams)

    @staticmethod
    def default_kenya() -> AllParams:
        return AllParams()
