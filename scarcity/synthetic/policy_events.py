"""
Policy Event Injector — Realistic Kenyan Policy Events for Synthetic Tweet Generation

Models the lifecycle of government policy announcements and their social media impact.
Policy events follow a 7-phase lifecycle that maps to distinct tweet patterns:

    LEAK → ANNOUNCE → REACT → MOBILIZE → IMPLEMENT → IMPACT → SETTLE

Each phase shifts the distribution of tweet intents, stances, and emotional intensity
across different account types, risk bands, and counties. This enables downstream
tracing of tweet volume/sentiment back to specific policy decisions.

Designed around real 2025-2026 Kenyan political events:
- Finance Bill cycles (taxes, levies)
- SHIF/NHIF health insurance transition
- Housing Levy rollout
- Fuel subsidy changes
- University funding model overhaul
- Digital services tax
- County revenue allocation
- Security operations
- Agricultural import policy
- Constitutional reform attempts
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Policy Phase Lifecycle
# =============================================================================

class PolicyPhase(str, Enum):
    """
    7-stage lifecycle of a policy event on social media.
    Durations are defaults — each PolicyEvent can override.
    """
    LEAK       = "leak"         # Rumor / media leak before official announcement
    ANNOUNCE   = "announce"     # Official government announcement
    REACT      = "react"        # Immediate public reaction (24-72h)
    MOBILIZE   = "mobilize"     # Organized opposition / support campaigns
    IMPLEMENT  = "implement"    # Policy takes effect
    IMPACT     = "impact"       # Visible economic / social consequences
    SETTLE     = "settle"       # Public attention fades, normalization

    @property
    def tweet_intensity(self) -> float:
        """Baseline tweet volume multiplier per phase."""
        return {
            PolicyPhase.LEAK: 0.3,
            PolicyPhase.ANNOUNCE: 1.0,
            PolicyPhase.REACT: 1.8,       # Peak social media activity
            PolicyPhase.MOBILIZE: 1.5,
            PolicyPhase.IMPLEMENT: 0.8,
            PolicyPhase.IMPACT: 1.2,
            PolicyPhase.SETTLE: 0.2,
        }[self]

    @property
    def dominant_intents(self) -> List[str]:
        """Which tweet intents dominate during this phase."""
        return {
            PolicyPhase.LEAK: ["rumor_mill", "casual"],
            PolicyPhase.ANNOUNCE: ["frustration", "satire_mockery", "casual"],
            PolicyPhase.REACT: ["frustration", "escalation", "satire_mockery"],
            PolicyPhase.MOBILIZE: ["mobilization", "escalation", "coordination"],
            PolicyPhase.IMPLEMENT: ["frustration", "infrastructure_stress", "casual"],
            PolicyPhase.IMPACT: ["frustration", "escalation", "infrastructure_stress"],
            PolicyPhase.SETTLE: ["casual", "satire_mockery"],
        }[self]

    @property
    def stance_distribution(self) -> Dict[str, float]:
        """Probability distribution of pro / anti / neutral stance per phase."""
        return {
            PolicyPhase.LEAK: {"anti": 0.35, "neutral": 0.45, "pro": 0.20},
            PolicyPhase.ANNOUNCE: {"anti": 0.45, "neutral": 0.30, "pro": 0.25},
            PolicyPhase.REACT: {"anti": 0.55, "neutral": 0.25, "pro": 0.20},
            PolicyPhase.MOBILIZE: {"anti": 0.60, "neutral": 0.15, "pro": 0.25},
            PolicyPhase.IMPLEMENT: {"anti": 0.45, "neutral": 0.35, "pro": 0.20},
            PolicyPhase.IMPACT: {"anti": 0.50, "neutral": 0.30, "pro": 0.20},
            PolicyPhase.SETTLE: {"anti": 0.25, "neutral": 0.50, "pro": 0.25},
        }[self]


# =============================================================================
# Policy Sector Tags
# =============================================================================

class PolicySector(str, Enum):
    """Sectors affected by policy events."""
    TAXATION    = "taxation"
    HEALTH      = "health"
    HOUSING     = "housing"
    FUEL_ENERGY = "fuel_energy"
    EDUCATION   = "education"
    DIGITAL     = "digital"
    SECURITY    = "security"
    AGRICULTURE = "agriculture"
    DEVOLUTION  = "devolution"
    CONSTITUTIONAL = "constitutional"
    TRANSPORT   = "transport"
    EMPLOYMENT  = "employment"


# =============================================================================
# Policy Event Definition
# =============================================================================

@dataclass
class PolicyEvent:
    """
    A single policy event with its lifecycle timeline and targeting rules.

    Attributes:
        event_id:           Unique identifier (e.g. "finance_bill_2026")
        name:               Human-readable name
        sector:             PolicySector tag
        description:        One-liner for tweet generation context
        leak_date:          When rumors start
        announce_date:      Official announcement
        implementation_date: When policy takes effect
        affected_counties:  Empty = nationwide
        severity:           0.0-1.0 public impact magnitude
        hashtags:           Organic hashtags for this event
        keywords_sw:        Swahili/Sheng keywords
        keywords_en:        English keywords
        phase_durations:    Override default phase durations (days)
    """
    event_id: str
    name: str
    sector: PolicySector
    description: str
    leak_date: datetime
    announce_date: datetime
    implementation_date: datetime
    affected_counties: List[str] = field(default_factory=list)  # empty = all
    severity: float = 0.5
    hashtags: List[str] = field(default_factory=list)
    keywords_sw: List[str] = field(default_factory=list)
    keywords_en: List[str] = field(default_factory=list)
    phase_durations: Dict[str, int] = field(default_factory=dict)

    def get_phase(self, current_date: datetime) -> Optional[PolicyPhase]:
        """
        Determine which lifecycle phase this event is in on current_date.
        Returns None if event hasn't started or has fully settled.
        """
        # Default durations (days from trigger)
        durations = {
            "leak": self.phase_durations.get("leak", 3),
            "announce_gap": 0,  # announce_date is fixed
            "react": self.phase_durations.get("react", 3),
            "mobilize": self.phase_durations.get("mobilize", 5),
            "implement_gap": 0,  # implementation_date is fixed
            "impact": self.phase_durations.get("impact", 7),
            "settle": self.phase_durations.get("settle", 5),
        }

        # LEAK: from leak_date to announce_date
        if self.leak_date <= current_date < self.announce_date:
            return PolicyPhase.LEAK

        # ANNOUNCE: announce_date itself (1 day)
        react_start = self.announce_date + timedelta(days=1)
        if self.announce_date <= current_date < react_start:
            return PolicyPhase.ANNOUNCE

        # REACT: 1-3 days after announcement
        react_end = react_start + timedelta(days=durations["react"])
        if react_start <= current_date < react_end:
            return PolicyPhase.REACT

        # MOBILIZE: until implementation
        if react_end <= current_date < self.implementation_date:
            return PolicyPhase.MOBILIZE

        # IMPLEMENT: implementation day + 2 days
        impact_start = self.implementation_date + timedelta(days=2)
        if self.implementation_date <= current_date < impact_start:
            return PolicyPhase.IMPLEMENT

        # IMPACT: 7 days of visible consequences
        impact_end = impact_start + timedelta(days=durations["impact"])
        if impact_start <= current_date < impact_end:
            return PolicyPhase.IMPACT

        # SETTLE: tail-off period
        settle_end = impact_end + timedelta(days=durations["settle"])
        if impact_end <= current_date < settle_end:
            return PolicyPhase.SETTLE

        return None

    def targets_county(self, county: str) -> bool:
        """Check if this event affects a specific county."""
        if not self.affected_counties:
            return True  # nationwide
        return county in self.affected_counties

    def get_reaction_strength(self, risk_band: str) -> float:
        """
        How strongly an account in this risk_band reacts to this event.
        High-risk accounts react more intensely.
        """
        band_mult = {
            "Low": 0.3,
            "Medium": 0.6,
            "High": 0.9,
            "Critical": 1.0,
        }
        return self.severity * band_mult.get(risk_band, 0.5)


# =============================================================================
# Kenyan Policy Event Catalog (2026 Timeline)
# =============================================================================

def build_kenya_2026_events(start_date: datetime) -> List[PolicyEvent]:
    """
    Build the catalog of realistic Kenyan policy events for simulation.

    These are calibrated against:
    - Actual 2024-2025 Kenyan policy patterns (Finance Bill protests, SHIF rollout)
    - Typical government policy cycle timings
    - Regional impact patterns

    The start_date anchors Day 0 of the simulation.
    """
    d = start_date  # shorthand

    events = [
        # ── FINANCE & TAXATION ──────────────────────────────────────────
        PolicyEvent(
            event_id="finance_bill_2026",
            name="Finance Bill 2026",
            sector=PolicySector.TAXATION,
            description="New Finance Bill proposing VAT on bread and digital transactions",
            leak_date=d + timedelta(days=1),
            announce_date=d + timedelta(days=4),
            implementation_date=d + timedelta(days=18),
            severity=0.95,
            hashtags=[
                "#RejectFinanceBill2026", "#FinanceBill2026", "#TaxUsMore",
                "#GenZNotAfraid", "#OccupyParliament", "#SisiNiNumbers"
            ],
            keywords_sw=[
                "ushuru", "kodi", "bei", "Finance Bill", "Bunge",
                "mfuko", "serikali inataka", "wananyonya damu"
            ],
            keywords_en=[
                "tax", "VAT", "Finance Bill", "parliament", "revenue",
                "afford", "cost of living", "bread tax"
            ],
            phase_durations={"react": 4, "mobilize": 8, "impact": 10},
        ),

        PolicyEvent(
            event_id="fuel_levy_hike",
            name="Fuel Levy Increase",
            sector=PolicySector.FUEL_ENERGY,
            description="KRA raises fuel levy by KES 5 per litre citing road maintenance",
            leak_date=d + timedelta(days=2),
            announce_date=d + timedelta(days=5),
            implementation_date=d + timedelta(days=12),
            affected_counties=[],  # nationwide
            severity=0.85,
            hashtags=[
                "#FuelPrice", "#PetrolWatch", "#MatutuFareUp",
                "#RutoMustGo", "#CostOfLiving"
            ],
            keywords_sw=[
                "mafuta", "petroli", "dizeli", "bei ya mafuta",
                "matatu fare", "nauli", "KES", "shilingi"
            ],
            keywords_en=[
                "fuel", "petrol", "diesel", "pump price",
                "transport cost", "matatu fare", "levy"
            ],
        ),

        # ── HEALTH ──────────────────────────────────────────────────────
        PolicyEvent(
            event_id="shif_rollout_phase2",
            name="SHIF Phase 2 Mandatory Enrollment",
            sector=PolicySector.HEALTH,
            description="Social Health Insurance Fund mandatory for informal sector workers",
            leak_date=d + timedelta(days=6),
            announce_date=d + timedelta(days=9),
            implementation_date=d + timedelta(days=22),
            severity=0.70,
            hashtags=[
                "#SHIF", "#NHIFIsDead", "#HealthcareForAll",
                "#SHIFScam", "#ReturnNHIF"
            ],
            keywords_sw=[
                "SHIF", "NHIF", "matibabu", "hospitali",
                "bima ya afya", "deduction", "jua kali"
            ],
            keywords_en=[
                "SHIF", "NHIF", "health insurance", "hospital",
                "deduction", "informal sector", "coverage"
            ],
        ),

        # ── HOUSING ─────────────────────────────────────────────────────
        PolicyEvent(
            event_id="housing_levy_increase",
            name="Housing Levy Rate Doubled",
            sector=PolicySector.HOUSING,
            description="Housing levy increased from 1.5% to 3% of gross salary",
            leak_date=d + timedelta(days=8),
            announce_date=d + timedelta(days=11),
            implementation_date=d + timedelta(days=25),
            severity=0.80,
            hashtags=[
                "#HousingLevy", "#AffordableHousing", "#PaycheckShrinking",
                "#RutoHousing", "#RejectHousingLevy"
            ],
            keywords_sw=[
                "housing levy", "kodi ya nyumba", "mshahara",
                "deduction", "affordable housing", "nyumba"
            ],
            keywords_en=[
                "housing levy", "salary deduction", "affordable housing",
                "paycheck", "3 percent", "housing fund"
            ],
        ),

        # ── EDUCATION ───────────────────────────────────────────────────
        PolicyEvent(
            event_id="university_funding_model",
            name="New University Funding Model",
            sector=PolicySector.EDUCATION,
            description="Government replaces HELB with means-tested university funding model",
            leak_date=d + timedelta(days=3),
            announce_date=d + timedelta(days=7),
            implementation_date=d + timedelta(days=20),
            severity=0.65,
            hashtags=[
                "#FundingModel", "#HELBRefund", "#UniversityFees",
                "#StudentsUnite", "#EducationKE"
            ],
            keywords_sw=[
                "HELB", "ada", "fees", "chuo kikuu", "university",
                "scholarship", "bursary", "elimu"
            ],
            keywords_en=[
                "HELB", "university fees", "funding model",
                "scholarship", "student loan", "means testing"
            ],
        ),

        # ── DIGITAL ECONOMY ─────────────────────────────────────────────
        PolicyEvent(
            event_id="digital_tax_expansion",
            name="Digital Services Tax Expansion",
            sector=PolicySector.DIGITAL,
            description="DST expanded to cover M-Pesa transactions above KES 500",
            leak_date=d + timedelta(days=5),
            announce_date=d + timedelta(days=8),
            implementation_date=d + timedelta(days=15),
            severity=0.75,
            hashtags=[
                "#DigitalTax", "#MPesaTax", "#TaxEverything",
                "#DigitalEconomy", "#MPesaIsBroken"
            ],
            keywords_sw=[
                "M-Pesa", "kodi ya simu", "digital tax", "transaction",
                "Safaricom", "paybill", "till number"
            ],
            keywords_en=[
                "digital tax", "M-Pesa", "transaction tax",
                "mobile money", "Safaricom", "DST"
            ],
        ),

        # ── TRANSPORT ───────────────────────────────────────────────────
        PolicyEvent(
            event_id="sgr_fare_restructure",
            name="SGR Fare Restructuring",
            sector=PolicySector.TRANSPORT,
            description="SGR economy fare increased 40%, Mombasa-Nairobi",
            leak_date=d + timedelta(days=10),
            announce_date=d + timedelta(days=13),
            implementation_date=d + timedelta(days=19),
            affected_counties=["Nairobi", "Mombasa", "Machakos", "Kilifi"],
            severity=0.55,
            hashtags=[
                "#SGRFare", "#MahindiFareToo", "#MombasaNairobi",
                "#TrainExpensive"
            ],
            keywords_sw=[
                "SGR", "treni", "nauli ya SGR", "Mombasa",
                "economy", "first class", "bei"
            ],
            keywords_en=[
                "SGR", "train fare", "Mombasa", "economy class",
                "40 percent", "transport"
            ],
        ),

        # ── AGRICULTURE ─────────────────────────────────────────────────
        PolicyEvent(
            event_id="sugar_import_ban_lift",
            name="Sugar Import Ban Lifted",
            sector=PolicySector.AGRICULTURE,
            description="Government lifts sugar import ban, flooding market with cheap imports",
            leak_date=d + timedelta(days=7),
            announce_date=d + timedelta(days=10),
            implementation_date=d + timedelta(days=14),
            affected_counties=[
                "Kakamega", "Bungoma", "Kisumu", "Kisii",
                "Narok", "Kericho", "Bomet"
            ],
            severity=0.70,
            hashtags=[
                "#SaveOurSugar", "#SugarImports", "#ProtectFarmers",
                "#WesternKenya", "#SugarBelt"
            ],
            keywords_sw=[
                "sukari", "import", "mkulima", "farmers",
                "bei ya sukari", "factory", "kiwanda"
            ],
            keywords_en=[
                "sugar", "imports", "farmers", "sugar belt",
                "cheap sugar", "factory", "cane growers"
            ],
        ),

        # ── SECURITY ────────────────────────────────────────────────────
        PolicyEvent(
            event_id="security_operation_northeast",
            name="Security Operation in Northern Kenya",
            sector=PolicySector.SECURITY,
            description="KDF launches large-scale security operation in Garissa and Turkana",
            leak_date=d + timedelta(days=4),
            announce_date=d + timedelta(days=6),
            implementation_date=d + timedelta(days=8),
            affected_counties=["Garissa", "Turkana"],
            severity=0.60,
            hashtags=[
                "#KDFOperation", "#SecurityKE", "#NorthernKenya",
                "#PeaceNow"
            ],
            keywords_sw=[
                "KDF", "jeshi", "operesheni", "usalama",
                "security", "Garissa", "Turkana"
            ],
            keywords_en=[
                "KDF", "military", "operation", "security",
                "Garissa", "Turkana", "deployment"
            ],
        ),

        # ── DEVOLUTION ──────────────────────────────────────────────────
        PolicyEvent(
            event_id="county_budget_cuts",
            name="County Revenue Allocation Slashed",
            sector=PolicySector.DEVOLUTION,
            description="National Treasury cuts county allocations by 15%, governors protest",
            leak_date=d + timedelta(days=9),
            announce_date=d + timedelta(days=12),
            implementation_date=d + timedelta(days=24),
            severity=0.65,
            hashtags=[
                "#CountyBudget", "#DevolutionUnderAttack",
                "#GovernorsFight", "#47Counties"
            ],
            keywords_sw=[
                "kaunti", "bajeti", "governor", "pesa",
                "county budget", "allocation", "treasury"
            ],
            keywords_en=[
                "county", "budget", "allocation", "treasury",
                "devolution", "governors", "15 percent"
            ],
        ),

        # ── CONSTITUTIONAL ──────────────────────────────────────────────
        PolicyEvent(
            event_id="referendum_push_2026",
            name="Constitutional Amendment Push",
            sector=PolicySector.CONSTITUTIONAL,
            description="Ruling coalition pushes for referendum to extend term limits",
            leak_date=d + timedelta(days=12),
            announce_date=d + timedelta(days=16),
            implementation_date=d + timedelta(days=28),
            severity=0.90,
            hashtags=[
                "#NoReferendum", "#TermLimits", "#ProtectConstitution",
                "#BBI2", "#WanjikuSays", "#PunguzaMizigo"
            ],
            keywords_sw=[
                "katiba", "referendum", "term limit", "BBI",
                "marekebisho", "kura ya maoni", "amendment"
            ],
            keywords_en=[
                "referendum", "constitution", "amendment",
                "term limits", "BBI", "recall"
            ],
        ),

        # ── EMPLOYMENT ──────────────────────────────────────────────────
        PolicyEvent(
            event_id="minimum_wage_freeze",
            name="Minimum Wage Freeze Announced",
            sector=PolicySector.EMPLOYMENT,
            description="Labour CS announces minimum wage freeze for 2026 despite inflation",
            leak_date=d + timedelta(days=6),
            announce_date=d + timedelta(days=9),
            implementation_date=d + timedelta(days=16),
            severity=0.60,
            hashtags=[
                "#MinimumWage", "#WorkersRights", "#COTUSpeak",
                "#LivingWage", "#PayWaLabour"
            ],
            keywords_sw=[
                "mshahara", "minimum wage", "COTU", "kazi",
                "wafanyikazi", "inflation", "mfumuko wa bei"
            ],
            keywords_en=[
                "minimum wage", "freeze", "workers", "COTU",
                "inflation", "cost of living", "salary"
            ],
        ),

        # ── ELECTRICITY ─────────────────────────────────────────────────
        PolicyEvent(
            event_id="electricity_tariff_hike",
            name="KPLC Electricity Tariff Increase",
            sector=PolicySector.FUEL_ENERGY,
            description="EPRA approves 15% electricity tariff increase effective March",
            leak_date=d + timedelta(days=11),
            announce_date=d + timedelta(days=14),
            implementation_date=d + timedelta(days=21),
            severity=0.72,
            hashtags=[
                "#KPLCHike", "#ElectricityBill", "#EPRAMustExplain",
                "#StimaNiGhali", "#KPLC"
            ],
            keywords_sw=[
                "stima", "KPLC", "umeme", "bill", "bei ya stima",
                "EPRA", "token", "prepaid"
            ],
            keywords_en=[
                "electricity", "KPLC", "tariff", "power bill",
                "EPRA", "token", "15 percent"
            ],
        ),
    ]

    return events


# =============================================================================
# Policy Event Injector
# =============================================================================

class PolicyEventInjector:
    """
    Injects policy events into the synthetic tweet generation pipeline.

    Usage:
        injector = PolicyEventInjector(start_date=datetime(2026, 2, 9))

        # For each simulated timestamp:
        active = injector.get_active_policy_events(current_date)
        for event, phase in active:
            intent = injector.select_intent(phase, account_risk_band)
            stance = injector.select_stance(phase, account_type)
            hashtag = injector.pick_hashtag(event)
            keyword = injector.pick_keyword(event, language="sw")
    """

    def __init__(self, start_date: datetime, seed: int = 42):
        self.start_date = start_date
        self.events = build_kenya_2026_events(start_date)
        self._rng = random.Random(seed)

    @property
    def event_ids(self) -> List[str]:
        """All registered event IDs."""
        return [e.event_id for e in self.events]

    def get_active_policy_events(
        self, current_date: datetime
    ) -> List[Tuple[PolicyEvent, PolicyPhase]]:
        """
        Return all policy events that are active on current_date,
        along with their current phase.
        """
        active: List[Tuple[PolicyEvent, PolicyPhase]] = []
        for event in self.events:
            phase = event.get_phase(current_date)
            if phase is not None:
                active.append((event, phase))
        return active

    def select_intent(
        self,
        phase: PolicyPhase,
        risk_band: str,
        account_type: str = "Individual",
    ) -> Optional[str]:
        """
        Pick a tweet intent appropriate for the policy phase.
        Higher risk bands and certain account types are more likely to
        pick aggressive intents.
        """
        intents = phase.dominant_intents

        # Government accounts mostly post pro-government casual/official
        if account_type == "Government":
            return self._rng.choice(["casual", "casual", "frustration"])

        # Bots amplify the most extreme intent available
        if account_type == "Bot":
            extreme = [i for i in intents if i in (
                "escalation", "mobilization", "hate_incitement"
            )]
            return self._rng.choice(extreme) if extreme else self._rng.choice(intents)

        # Risk-band weighting: higher risk → more extreme intents
        if risk_band in ("High", "Critical"):
            # Weight towards latter (more extreme) intents
            weights = [1.0 + i * 0.5 for i in range(len(intents))]
        else:
            # Uniform
            weights = [1.0] * len(intents)

        return self._rng.choices(intents, weights=weights, k=1)[0]

    def select_stance(
        self,
        phase: PolicyPhase,
        account_type: str = "Individual",
    ) -> str:
        """
        Pick a stance (anti / neutral / pro) for the account.
        Government accounts are always pro. Bots are always anti.
        """
        if account_type == "Government":
            return "pro"
        if account_type == "Bot":
            return "anti"

        dist = phase.stance_distribution
        return self._rng.choices(
            list(dist.keys()),
            weights=list(dist.values()),
            k=1,
        )[0]

    def pick_hashtag(self, event: PolicyEvent) -> str:
        """Pick a random hashtag from the event's hashtag list."""
        if event.hashtags:
            return self._rng.choice(event.hashtags)
        return ""

    def pick_keyword(self, event: PolicyEvent, language: str = "sw") -> str:
        """Pick a keyword (Swahili or English) from the event."""
        pool = event.keywords_sw if language == "sw" else event.keywords_en
        if pool:
            return self._rng.choice(pool)
        return ""

    def get_topic_cluster(self, event: PolicyEvent) -> str:
        """Map event sector to a topic cluster label for the output CSV."""
        return event.sector.value

    def get_event_by_id(self, event_id: str) -> Optional[PolicyEvent]:
        """Lookup a specific event by ID."""
        for e in self.events:
            if e.event_id == event_id:
                return e
        return None

    def should_account_react(
        self,
        event: PolicyEvent,
        phase: PolicyPhase,
        account: dict,
    ) -> bool:
        """
        Probabilistic check: should this account tweet about this event?
        Considers: severity, risk_band, county targeting, phase intensity.

        The probability is capped to avoid any single high-severity event
        from dominating the entire dataset.
        """
        # County check
        home_county = account.get("home_county", "Nairobi")
        if not event.targets_county(home_county):
            # Small chance of nationwide spillover
            if self._rng.random() > 0.15:
                return False

        # Reaction probability = severity × phase_intensity × risk_modifier
        risk_band = account.get("risk_band", "Low")
        risk_mult = {"Low": 0.15, "Medium": 0.30, "High": 0.45, "Critical": 0.55}
        prob = event.severity * phase.tweet_intensity * risk_mult.get(risk_band, 0.2)

        # Account type modifier
        acc_type = account.get("account_type", "Individual")
        if acc_type == "Bot":
            prob *= 1.3  # bots amplify
        elif acc_type == "Government":
            prob *= 0.3  # government less reactive on social media
        elif acc_type == "Organization":
            prob *= 0.6

        # Soft cap to prevent any single event from overwhelming the corpus
        return self._rng.random() < min(prob, 0.45)
