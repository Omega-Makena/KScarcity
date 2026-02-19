"""
KShield - National Threat Detection & Economic Governance System

Exposes core components for the unified dashboard.
Components are imported lazily to avoid 3+ second import overhead
when only UI modules are needed.
"""


def __getattr__(name):
    """Lazy import of heavy submodules — only loaded when actually accessed."""
    if name in ("EconomicGovernor", "EconomicGovernorConfig"):
        from .core.governance import EconomicGovernor, EconomicGovernorConfig
        globals()["EconomicGovernor"] = EconomicGovernor
        globals()["EconomicGovernorConfig"] = EconomicGovernorConfig
        return globals()[name]
    if name in ("default_economic_policies", "EconomicPolicy"):
        from .core.policies import default_economic_policies, EconomicPolicy
        globals()["default_economic_policies"] = default_economic_policies
        globals()["EconomicPolicy"] = EconomicPolicy
        return globals()[name]
    if name in ("Shock", "ShockManager", "ShockType"):
        from .core.shocks import Shock, ShockManager, ShockType
        globals()["Shock"] = Shock
        globals()["ShockManager"] = ShockManager
        globals()["ShockType"] = ShockType
        return globals()[name]
    if name in ("KShieldHub", "get_hub"):
        try:
            from .hub import KShieldHub, get_hub
            globals()["KShieldHub"] = KShieldHub
            globals()["get_hub"] = get_hub
        except ImportError:
            globals()["KShieldHub"] = None
            globals()["get_hub"] = None
        return globals()[name]
    if name in ("PulseSensor", "compute_threat_report"):
        try:
            from .pulse import PulseSensor, compute_threat_report
            globals()["PulseSensor"] = PulseSensor
            globals()["compute_threat_report"] = compute_threat_report
        except ImportError:
            globals()["PulseSensor"] = None
            globals()["compute_threat_report"] = None
        return globals()[name]
    raise AttributeError(f"module 'kshiked' has no attribute {name!r}")


__all__ = [
    "EconomicGovernor",
    "EconomicGovernorConfig",
    "default_economic_policies",
    "EconomicPolicy",
    "Shock",
    "ShockManager",
    "ShockType",
    "KShieldHub",
    "get_hub",
    "PulseSensor",
    "compute_threat_report"
]

