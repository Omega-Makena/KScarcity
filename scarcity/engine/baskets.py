"""
Basket Registry — domain-scoped variable groupings for isolated FL.

A "basket" is a named sector/domain. Each basket owns a subset of variables.
Variables may belong to multiple baskets; discovery within each basket is
fully isolated — hypotheses never leak across basket boundaries.

Basket isolation is the contract that keeps the general engine general:
  - The engine code has no sector knowledge.
  - Bias is introduced only by which variables a basket engine sees.
  - Cross-sector relationships are discovered by cross-basket bridges (future).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Basket definitions
# ---------------------------------------------------------------------------

_BASKET_SPECS: Dict[str, Dict] = {
    "macro": {
        "description": "Core macroeconomic aggregates — output, prices, trade, fiscal",
        "variables": [
            "gdp_growth", "inflation", "unemployment",
            "exports_gdp", "imports_gdp", "current_account",
            "govt_consumption", "tax_revenue", "govt_debt",
        ],
        # structural = estimated from data; identity = accounting constraint;
        # instrument = policy-controlled; derived = computed from other vars
        "variable_types": {
            "gdp_growth":       "structural",
            "inflation":        "structural",
            "unemployment":     "structural",
            "exports_gdp":      "structural",
            "imports_gdp":      "structural",
            "current_account":  "identity",    # = exports - imports (± transfers)
            "govt_consumption": "instrument",
            "tax_revenue":      "identity",    # = f(gdp, tax_rate) by accounting
            "govt_debt":        "instrument",
        },
    },
    "financial": {
        "description": "Financial sector depth — credit, money, interest rates",
        "variables": [
            "real_interest_rate", "broad_money", "private_credit",
            "govt_debt", "inflation", "gdp_growth",
        ],
        "variable_types": {
            "real_interest_rate": "instrument",
            "broad_money":        "identity",   # M2 = currency + deposits (money multiplier)
            "private_credit":     "structural",
            "govt_debt":          "instrument",
            "inflation":          "structural",
            "gdp_growth":         "structural",
        },
    },
    "infrastructure": {
        "description": "Physical and digital infrastructure",
        "variables": [
            "electricity_access", "internet_users", "mobile_subscriptions",
            "urban_population", "gdp_growth",
        ],
        "variable_types": {
            "electricity_access":  "structural",
            "internet_users":      "structural",
            "mobile_subscriptions":"structural",
            "urban_population":    "structural",
            "gdp_growth":          "structural",
        },
    },
    "human_capital": {
        "description": "Education, health, demographics",
        "variables": [
            "school_enrollment", "life_expectancy", "urban_population",
            "unemployment", "gdp_growth",
        ],
        "variable_types": {
            "school_enrollment": "structural",
            "life_expectancy":   "structural",
            "urban_population":  "structural",
            "unemployment":      "structural",
            "gdp_growth":        "structural",
        },
    },
}


# ---------------------------------------------------------------------------
# Basket dataclass
# ---------------------------------------------------------------------------

@dataclass
class Basket:
    """
    An isolated domain context for engine instances.

    The `variables` set defines what this basket sees. Only rows filtered to
    these variables are fed into the basket's engine instance. Hypotheses
    discovered are therefore scoped to this domain.

    `variable_types` maps each variable to one of:
      "structural"  — empirically estimated from data
      "identity"    — defined by an accounting constraint (current_account,
                      tax_revenue, broad_money)
      "instrument"  — policy-controlled (real_interest_rate, govt_debt,
                      govt_consumption)
      "derived"     — computed from other variables in the basket
    """
    basket_id: str
    description: str
    variables: FrozenSet[str]
    variable_types: Dict[str, str] = field(default_factory=dict)

    @property
    def schema(self) -> Dict:
        """Engine-compatible schema dict."""
        return {"fields": [{"name": v, "type": "float"} for v in sorted(self.variables)]}

    def filter_row(self, row: Dict[str, float]) -> Dict[str, float]:
        """Return only the variables that belong to this basket."""
        return {k: v for k, v in row.items() if k in self.variables}

    def has_variables(self, row: Dict[str, float], min_present: int = 2) -> bool:
        """True if the row contains at least min_present basket variables."""
        return sum(1 for k in row if k in self.variables) >= min_present

    def get_variable_type(self, variable: str) -> str:
        """Return the semantic type of a variable; defaults to 'structural'."""
        return self.variable_types.get(variable, "structural")

    def identity_variables(self) -> FrozenSet[str]:
        """Variables tagged as accounting identities."""
        return frozenset(v for v in self.variables if self.variable_types.get(v) == "identity")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class BasketRegistry:
    """
    Central registry of all domain baskets.

    Provides lookup by basket_id and reverse lookup (variable → baskets).
    """

    def __init__(self, specs: Dict[str, Dict] = _BASKET_SPECS):
        self._baskets: Dict[str, Basket] = {}
        for bid, spec in specs.items():
            self._baskets[bid] = Basket(
                basket_id=bid,
                description=spec["description"],
                variables=frozenset(spec["variables"]),
                variable_types=spec.get("variable_types", {}),
            )
        # reverse index: variable → set of basket_ids
        self._var_to_baskets: Dict[str, List[str]] = {}
        for bid, basket in self._baskets.items():
            for var in basket.variables:
                self._var_to_baskets.setdefault(var, []).append(bid)

    def get(self, basket_id: str) -> Basket:
        if basket_id not in self._baskets:
            raise KeyError(f"Unknown basket: {basket_id!r}. Available: {list(self._baskets)}")
        return self._baskets[basket_id]

    def all_ids(self) -> List[str]:
        return list(self._baskets.keys())

    def baskets_for_variable(self, variable: str) -> List[str]:
        """Which baskets contain this variable?"""
        return self._var_to_baskets.get(variable, [])

    def route_row(self, row: Dict[str, float]) -> Dict[str, Dict[str, float]]:
        """
        Split a raw observation into per-basket filtered sub-rows.
        Returns {basket_id: filtered_row} for baskets with >=2 variables present.
        """
        result: Dict[str, Dict[str, float]] = {}
        for bid, basket in self._baskets.items():
            filtered = basket.filter_row(row)
            if len(filtered) >= 2:
                result[bid] = filtered
        return result

    def variable_type(self, variable: str) -> str:
        """Global type lookup: first basket that knows the variable wins."""
        for basket in self._baskets.values():
            t = basket.variable_types.get(variable)
            if t:
                return t
        return "structural"

    def __contains__(self, basket_id: str) -> bool:
        return basket_id in self._baskets

    def __len__(self) -> int:
        return len(self._baskets)


# Module-level singleton
REGISTRY = BasketRegistry()
