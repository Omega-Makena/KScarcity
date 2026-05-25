# Synthetic Module — Utilities

The synthetic module generates realistic social media datasets with embedded ground-truth labels for testing the SCARCITY pipeline. It models Kenyan social media behavior with policy-reaction signals, crisis scenarios, account archetypes, and threat-scored content.

---

## accounts.py — Account Generator

### `AccountGenerator`

Generates synthetic social media account profiles.

```python
from scarcity.synthetic.accounts import AccountGenerator

gen = AccountGenerator(seed=42)
accounts = gen.generate_accounts(num_accounts=1000)
```

Each account dict contains:

| Field | Description |
|-------|-------------|
| `account_id` | UUID string |
| `account_type` | Individual (70%), Bot (20%), Organization (5%), Government (5%) |
| `risk_band` | Low / Medium / High / Critical — higher bands cluster into lower network_cluster_ids |
| `followers_count` | Log-normal distributed by type |
| `baseline_post_rate` | Posts per day (Bots: 10–50, Individuals: 0.5–3, Government: 2–8) |
| `home_county` | Kenyan county sampled from `COUNTY_WEIGHTS` |
| `primary_device` | Sampled from `DEVICE_TYPES` |
| `network_cluster_id` | Integer cluster; Critical risk → clusters 1–5 |

Account type probabilities and risk distributions:
- **Individual**: 70% of accounts; risk Low (80%), Medium (15%), High (4%), Critical (1%)
- **Bot**: 20%; high following count, high post rate; 50% High risk
- **Organization**: 5%; large follower counts, Low risk
- **Government**: 5%; very large followers, cluster 0

---

## behavior.py — Behavior Simulator

### `BehaviorSimulator`

Generates a chronological activity schedule for each account over a simulation period.

```python
from scarcity.synthetic.behavior import BehaviorSimulator

sim = BehaviorSimulator(seed=42)
schedule = sim.generate_activity_schedule(account, start_date, duration_days, scenario_manager)
```

**State machine** — each account runs through: `NORMAL → ESCALATING → RECOVERING → NORMAL`

- High/Critical risk accounts have 10% daily chance of entering `ESCALATING`
- `ESCALATING` boosts daily rate 2–5×; 20% chance per day of burning out to `RECOVERING`
- `RECOVERING` drops rate to 10%; 30% chance per day of returning to `NORMAL`

**Scenario modifiers** (applied before state machine):
- `silence` events: daily rate × 0.1 for targeted risk bands
- `migration_signal` events: daily rate × 0.5
- Active policy events: each adds `baseline_rate × tweet_intensity × 0.15 / (1 + idx×0.5)` with diminishing returns

**Post scheduling** — number of posts per day drawn from Poisson(daily_rate). Hour sampled from Beta(5, 2) distribution (favouring 8am–10pm).

Each activity record includes: `timestamp`, `intent`, `state`, and optionally `policy_event_id`, `policy_phase`, `stance`.

**`calculate_trajectory_metrics(activities)`** — computes per-account summary statistics over the schedule.

---

## content.py — Content Generator

### `ContentGenerator`

Generates tweet text from templates with Kenyan code-switching (Swahili/English slang).

```python
from scarcity.synthetic.content import ContentGenerator

gen = ContentGenerator(seed=42)
text = gen.generate_tweet(account, intent="mobilization")
```

**Intents**: `casual`, `frustration`, `mobilization`, `escalation`, `coordination`, `hate_incitement`, `satire_mockery`, `rumor_mill`, `infrastructure_stress`

**Template filling**: placeholders like `{pronoun}`, `{entity}`, `{slang_urban}` are filled from vocabulary lists. ~40% of tweets receive additional slang injection.

**Policy tweets**: `generate_policy_tweet(account, policy_event, phase, stance)` generates phase-appropriate content referencing the specific policy event.

**Score calculation**:
- `calculate_scores(text, intent)` → dict with `imperative_rate`, `urgency_rate`, `coordination_score`, `escalation_score`, `threat_score`
- `calculate_policy_scores(text, policy_event, phase, stance)` → adds `sentiment_score`, `stance_score`, `topic_cluster`, `policy_severity`

---

## vocabulary.py — Lexicon and Templates

Provides all vocabulary data used by `ContentGenerator` and `AccountGenerator`:

| Constant | Contents |
|----------|----------|
| `THREAT_LEXICON` | Categorized threat keywords: urgency, imperative, mobilization, coordination, persistence, escalation, collective_identity |
| `SLANG_CATEGORIES` | Kenyan slang by register: urban_mix, internet_youth, political_frustration, protest_tone, callout_culture, coordination_slang, escalation_energy, rumor_suspicion, sarcastic_mocking |
| `POLITICAL_SPECIFIC` | Named entities (politicians, institutions, counties) |
| `HATE_TERMS` | Hate speech indicators |
| `TEMPLATES` | Per-intent tweet templates with placeholder slots |
| `PRONOUNS` | me/us pronoun pools |
| `VERBS_FAIL` | Failure/frustration verb list |
| `SATIRE_TERMS` | Satirical vocabulary |
| `POLICY_STANCE_WORDS` | Pro/anti/neutral stance vocabulary |
| `POLICY_TEMPLATES` | Phase-specific policy tweet templates |
| `POLICY_IMPACT_CONSEQUENCES` | Economic/social consequence phrases |
| `COUNTY_COORDINATES` | Dict mapping county name → (lat, lon) |
| `COUNTY_WEIGHTS` | Sampling probability per county (Nairobi-weighted) |
| `INTERACTION_WEIGHTS` | Per account-type weights for Tweet/Retweet/Reply/Quote |
| `DEVICE_TYPES` | List of device source labels |

---

## scenarios.py — Crisis Scenario Manager

### `ScenarioManager`

Defines a ground-truth storyline of crisis events that modulates account behavior throughout the simulation.

```python
from scarcity.synthetic.scenarios import ScenarioManager

manager = ScenarioManager(start_date=datetime.now(), duration_days=30)
active = manager.get_active_events(current_date)
policy_active = manager.get_active_policy_events(current_date)
```

**Built-in scenario storyline** (4 events):

| Event | Type | Days | Target |
|-------|------|------|--------|
| Nairobi Grid Failure | infrastructure_stress | 3–5 | All risk bands |
| Regional Protests (Contagion) | mobilization | 5–8 | Medium/High/Critical |
| Secure Channel Migration | migration_signal | 7–10 | High/Critical (rate × 0.5) |
| Tactical Silence | silence | 9 | Critical in Nairobi/Mombasa (rate × 0.1) |

Each event dict has: `name`, `type`, `start_day`, `end_day`, `target_counties`, `target_risk`, `intensity`.

Also holds a `PolicyEventInjector` instance accessed via `manager.policy_injector`.

---

## policy_events.py — Policy Event Injector

### `PolicyPhase`

7-stage lifecycle enum modelling how a policy event unfolds on social media:

```
LEAK → ANNOUNCE → REACT → MOBILIZE → IMPLEMENT → IMPACT → SETTLE
```

| Phase | tweet_intensity | Dominant intents |
|-------|----------------|-----------------|
| LEAK | 0.3 | rumor_mill, casual |
| ANNOUNCE | 1.0 | frustration, satire_mockery, casual |
| REACT | 1.8 (peak) | frustration, escalation, satire_mockery |
| MOBILIZE | 1.5 | mobilization, escalation, coordination |
| IMPLEMENT | 0.8 | frustration, infrastructure_stress, casual |
| IMPACT | 1.2 | frustration, escalation, infrastructure_stress |
| SETTLE | 0.2 | casual, satire_mockery |

### `PolicyEventInjector`

Manages a collection of `PolicyEvent` objects modelled on real 2025–2026 Kenyan policy events:

- Finance Bill cycles (taxes, levies)
- SHIF/NHIF health insurance transition
- Housing Levy rollout
- Fuel subsidy changes
- University funding overhaul
- Digital services tax
- County revenue allocation
- Security operations
- Agricultural import policy
- Constitutional reform attempts

```python
from scarcity.synthetic.policy_events import PolicyEventInjector

injector = PolicyEventInjector(start_date=datetime.now())
active = injector.get_active_events(current_date)          # → List[(PolicyEvent, PolicyPhase)]
event = injector.get_event_by_id("finance_bill_2025")      # → PolicyEvent | None
```

Each `PolicyEvent` carries: `event_id`, phase schedule, `tweet_intensity` per phase, `dominant_intents`, `stance_distribution` (pro/anti/neutral probabilities).

---

## pipeline.py — Synthetic Pipeline

### `SyntheticPipeline`

Top-level orchestrator that wires all components together and writes output CSVs.

```python
from scarcity.synthetic.pipeline import SyntheticPipeline

pipeline = SyntheticPipeline(output_dir="data/synthetic")
pipeline.run(num_accounts=100, duration_days=30, start_date=None)
```

**Output files**:

| File | Contents |
|------|----------|
| `accounts.csv` | One row per account with profile fields + trajectory metrics |
| `tweets.csv` | One row per post with text, scores, policy tracing fields, geolocation |

**Tweet fields** (subset):

| Field | Description |
|-------|-------------|
| `post_id` | Sequential ID |
| `account_id` | Foreign key to accounts |
| `timestamp` | Datetime of post |
| `text` | Generated tweet text |
| `interaction_type` | Tweet / Retweet / Reply / Quote |
| `imperative_rate`, `urgency_rate` | Threat scores |
| `coordination_score`, `escalation_score`, `threat_score` | Operational risk signals |
| `sentiment_score`, `stance_score` | Policy reaction signals |
| `policy_event_id`, `policy_phase` | Ground-truth policy tracing |
| `topic_cluster` | Policy topic label |
| `policy_severity` | Phase-weighted severity |
| `latitude`, `longitude` | County-level location with jitter |

**Interaction logic**: Bots amplify same-policy-event tweets and high-escalation content; other account types sample from the last 100 recent tweets. Falls back to `Tweet` if no candidates.

**Console summary** on completion: counts of policy-reaction, crisis, and organic tweets; per-event coverage with phase and stance breakdowns.
