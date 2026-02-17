# KShield Simulation — Index

> `kshiked.simulation` — Kenya-specific calibration, scenario library, shock compilation, and simulation control.

## Module Purpose

This package bridges **Scarcity's domain-agnostic SFC engine** to Kenya-specific reality:
- Reads World Bank / KNBS data to calibrate SFC parameters
- Provides 9 named scenario templates and 8 policy templates
- Compiles shock vectors into the format expected by `SFCEconomy`
- Controls multi-run execution (Monte Carlo, stress tests)

## Key Components

### `kenya_calibration.py` — Data-Driven Parameter Derivation

| Export | Type | Description |
|--------|------|-------------|
| `calibrate_from_data(loader, steps, policy_mode, overrides)` | Function → `CalibrationResult` | Main entry — reads data, returns `.config`, `.params`, `.overall_confidence` |
| `OUTCOME_DIMENSIONS` | Dict (11 entries) | Metadata for each outcome dimension — label, unit, format, higher_is, category |
| `DEFAULT_DIMENSIONS` | List (5 entries) | Default dimensions for UI display |

**Fallback behaviour**: when data is missing, middle-income-country defaults are used (confidence marked lower).

### `scenario_templates.py` — Scenario & Policy Library

| Export | Count | Description |
|--------|-------|-------------|
| `SCENARIO_LIBRARY` | 9 | Named shock scenarios (Oil Crisis, Drought, Perfect Storm, etc.) |
| `POLICY_TEMPLATES` | 8 | Named policy packages (CBK Tightening, Austerity, Rate Cap 2016, etc.) |
| `ScenarioTemplate` | dataclass | `id`, `name`, `category`, `shocks`, `shock_onset`, `shock_duration`, `shock_shape`, `suggested_policy`, `suggested_dimensions` |
| `build_shock_vectors(steps)` | method | Converts template into per-timestep shock dicts for the SFC engine |

### `compiler.py` — ShockCompiler

Translates pulse detections and scenario templates into SFC-compatible shock vectors.

### `controller.py` — SimulationController

Orchestrates single runs, parameter sweeps, and Monte Carlo batches; delegates execution to `SFCEconomy`.

### `validation.py` — Output Validation

Checks trajectory outputs for accounting identity violations and constraint breaches.

### `fallback_blender.py` — Fallback Blender

Merges multiple calibration sources when primary data has gaps.

## Files

| File | Description |
|------|-------------|
| [00_overview.md](00_overview.md) | ShockCompiler, SimulationController, SFC integration |

## Quick Links

- Source: `kshiked/simulation/`
- Key classes: `ShockCompiler`, `SimulationController`, `ScenarioTemplate`
- Data-driven calibration: `kenya_calibration.py`
- Depends on: `kshiked.core.shocks`, `scarcity.simulation.sfc`
- Full reference: See [SIMULATION_ENGINE.md](../../SIMULATION_ENGINE.md)
