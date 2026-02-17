# KShield Pulse Module — Overview

The **pulse module** is the social sensing engine of KShield. It processes social media data to detect early warning signals of economic and political stress.

---

## Purpose

Social media provides real-time signals of:
- **Economic distress**: Food prices, unemployment complaints
- **Political instability**: Anti-government sentiment, mobilization
- **Social tensions**: Ethnic polarization, information warfare

The pulse module:
1. **Detects signals** using NLP-enhanced text analysis
2. **Tracks over time** using time-weighted co-occurrence
3. **Computes indices** for threat assessment
4. **Feeds simulations** with shock vectors

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         PulseSensor                              │
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │ Scrapers │───►│Detectors │───►│Cooccur   │───►│ Indices  │ │
│   │(collect) │    │(NLP)     │    │(weight)  │    │(compute) │ │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘ │
│        │                                               │        │
│        ▼                                               ▼        │
│   ┌──────────┐                                   ┌──────────┐  │
│   │ Social   │                                   │ GeoMapper│  │
│   │ APIs     │                                   │ (Kenya)  │  │
│   └──────────┘                                   └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 15 Signal Types

| ID | Signal | Description |
|----|--------|-------------|
| 1 | Survival Cost Stress | Complaints about food, rent, utilities |
| 2 | Distress Framing | Crisis language, suffering narratives |
| 3 | Emotional Exhaustion | Hopelessness, fatigue |
| 4 | Directed Rage | Specific anger at leaders |
| 5 | Bond Fracturing | Us-vs-them language |
| 6 | Regime Slang | Mocking/delegitimizing language |
| 7 | State Stress | Perceptions of state failure |
| 8 | Dehumanization | Extreme othering language |
| 9 | Legitimacy Rejection | Rejection of state authority |
| 10 | Elite Fracture | Elite conflict signals |
| 11 | Counter-Narrative | Alternative narratives |
| 12 | Rumor Velocity | Panic spread |
| 13 | Diaspora Remittance | Remittance signals |
| 14 | Mobilization Language | Call-to-action |
| 15 | Coordination Infrastructure | Organization signals |

---

## Key Components

### PulseSensor (`sensor.py`)

Main orchestrator:

```python
from kshiked.pulse import PulseSensor

sensor = PulseSensor(use_nlp=True)

# Process incoming text
detections = sensor.process_text(
    text="Prices are killing us, can't afford maize",
    metadata={"source": "twitter", "location": "Nairobi"}
)

# Update primitive state
sensor.update_primitives(detections)

# Get current state
state = sensor.get_state()
```

### Signal Detectors (`detectors.py`)

NLP-enhanced detection:

```python
class NLPSignalDetector:
    def __init__(
        self,
        signal_id: SignalID,
        keywords: List[str],
        target_emotions: List[str],
        sentiment_weight: float = 0.3,
        keyword_weight: float = 0.4,
        emotion_weight: float = 0.3
    )
    
    def detect(self, text: str) -> SignalDetection:
        # Combines: keyword score + sentiment + emotion + entities
```

**Detector types**:
- `SurvivalCostStressDetector`
- `DistressFramingDetector`
- `EmotionalExhaustionDetector`
- `DirectedRageDetector`
- `DehumanizationDetector`
- ... (15 total)

### Co-occurrence Analysis (`cooccurrence.py`)

Time-weighted signal tracking:

```python
# Temporal decay functions
decay = ExponentialDecay(half_life_seconds=3600)  # 1 hour
decay = LinearDecay(max_age_seconds=7200)
decay = StepDecay(window_seconds=3600)

# Rolling window
window = RollingWindow(window_seconds=3600, decay=decay)
window.add_detection(detection)

# Get weighted signals
events = window.get_weighted_events()
```

### Threat Indices (`indices.py`)

**Phase 1 (High Priority)**:
- `PolarizationIndex` — Group division
- `LegitimacyErosionIndex` — State authority degradation
- `MobilizationReadinessScore` — Mass action likelihood

**Phase 2 (Medium Priority)**:
- `EliteCohesionIndex` — Elite solidarity (inverted)
- `InformationWarfareIndex` — Disinformation activity

```python
# Compute index
pi = PolarizationIndex.compute(state, recent_signals)
print(pi.value, pi.severity)  # 0.65, "moderate"
```

---

## File Guide

| File | Purpose |
|------|---------|
| `sensor.py` | PulseSensor orchestrator |
| `detectors.py` | NLP signal detectors |
| `cooccurrence.py` | Temporal decay, rolling windows |
| `indices.py` | Threat index computation |
| `primitives.py` | Signal state primitives |
| `mapper.py` | SignalID mappings |
| `nlp.py` | NLP pipeline (sentiment, emotion) |
| `network.py` | Network analysis |
| `social.py` | Social API clients |
| `geo_mapper.py` | Kenya county mapping |
| `visualization.py` | Dashboard visualizations |

---

## Integration

### With Scrapers

```python
from kshiked.pulse.scrapers import TwitterScraper

scraper = TwitterScraper()
for tweet in scraper.stream("Kenya"):
    detections = sensor.process_text(tweet.text)
```

### With Simulation

```python
# Get shock vector for simulation
shocks = sensor.get_shock_vector([
    "GDP (current US$)",
    "Inflation, consumer prices (annual %)"
])
# Returns: {"GDP (current US$)": -0.02, "Inflation": 0.01}
```

### With EventBus

```python
bus.publish("pulse.detection", {
    "signal_id": "SURVIVAL_COST_STRESS",
    "intensity": 0.75,
    "location": "Nairobi"
})
```

---

## Kenya-Specific Features

### Ethnic Tension Tracking

```python
TENSION_PAIRS = [
    ("kikuyu", "luo"),      # 2007-08 violence
    ("kikuyu", "kalenjin"), # Rift Valley
    ("luo", "kalenjin"),
]
```

### County Mapping

```python
from kshiked.pulse import GeoMapper

mapper = GeoMapper()
county = mapper.detect_county("Prices in Mombasa are crazy")
# Returns: "Mombasa"
```
