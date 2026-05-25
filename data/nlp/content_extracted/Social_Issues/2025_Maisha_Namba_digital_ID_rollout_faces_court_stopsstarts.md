# 2025: Maisha Namba digital ID rollout faces court stops/starts (privacy + inclusion contestation).

**Theme**: Social Issues
**Source Document**: kenya_policies_master_dossier_2020_2026

---

2023–2025: Maisha Namba digital ID rollout faces court stops/starts (privacy + inclusion contestation).
24 Sept 2024: Privatisation Act 2023 declared unconstitutional at High Court (appeal dynamics follow).
3 Oct 2024: HEF suspended by High Court; later reinstated by Court of Appeal (2025).
2024: SHIF contribution timing shifts and implementation mechanics adjust amid litigation/transition risk.
2025: Appellate reversals/stays affecting HEF and privatisation implementation; trade policy shock via
EPA litigation (late 2025).
Jan–Feb 2026: SHA anti-fraud posture and system tightening signals dominate policy-as-enforcement
layer.
4) Policy fi Impact Graph Blueprint (KShield-ready)
4.1 Graph primitives
Recommended directed multi-layer graph. Node types: PolicyNode, MechanismNode,
EconomicIndicatorNode, ServiceDeliveryNode, SocialSignalNode, ThreatIndexNode. Edge types:
PolicyfiMechanism, MechanismfiIndicator, IndicatorfiSocialSignal (Granger + estimands),
ServiceDeliveryfiSocialSignal, SocialSignalfiThreatIndex, CourtActionfiPolicyStatus.
4.2 Edge attributes (minimum)
direction; lag_days; confidence; evidence; estimand (ATE/ATT/LATE/CATE); granger_p; effect_size.
4.3 Concrete templates (examples)
Finance Bill 2024 — Tax shock fi grievance fi mobilization
(cid:127) Mechanisms: proposed levies/tax expansions
(cid:127) Indicators: CPI (food/transport), pump prices, disposable income
(cid:127) Signals: protest counts, arrests/injuries, online anger sentiment
(cid:127) Threat indices: MobilizationReadiness›, Legitimacyfl, Polarization›
(cid:127) Edges: VAT/levy proposalsfionline anger (1–7d); fuelfitransport CPIfiprotests (14–45d); Granger +
CATE by youth share/unemployment.
Housing Levy / Affordable Housing Act — Wage wedge fi formal household stress
(cid:127) Mechanism: 1.5% payroll deduction + employer match
(cid:127) Indicators: net wages, formal employment, consumption proxies
(cid:127) Signals: union statements, court petitions, payroll complaints
(cid:127) Estimands: ATT on formal payroll workers; CATE by income band/sector.
SHA/SHIF reform — Service delivery shock fi legitimacy risk
(cid:127) Mechanisms: contribution rules, implementation shifts, claims capacity
(cid:127) Service nodes: claim acceptance, facility participation, turn-away incidents
(cid:127) Estimands: ATE of transition window; LATE using onboarding date/system uptime instruments (if
available).
Maisha Namba — Identity rollout fi inclusion/privacy contestation
(cid:127) Court stays as regime switches (on/off)
(cid:127) Indicators: enrollment backlog, access delays
(cid:127) Method: interrupted time series around court orders; track advocacy/litigation signals.
HEF model — Fee burden/misclassification fi youth protest propensity
(cid:127) Indicators: fee arrears, dropout proxies, loan uptake
(cid:127) Signals: campus protests/strikes
(cid:127) Threat indices: MobilizationReadiness› (youth), Legitimacyfl.