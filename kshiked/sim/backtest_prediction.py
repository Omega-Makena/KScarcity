"""
KShield Backtest Prediction Engine

Usage: python -m kshiked.sim.backtest_prediction
(requires: pip install -e .)
"""
import asyncio
import pandas as pd
import numpy as np
import logging
import os
import sys
from typing import Dict, List
from dataclasses import dataclass
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("kshield.backtest_v4")

# Imports from Scarcity/KShield
try:
    from scarcity.simulation.agents import AgentRegistry, NodeAgent, EdgeLink
    from scarcity.simulation.environment import SimulationEnvironment, EnvironmentConfig
    from scarcity.simulation.dynamics import DynamicsConfig, DynamicsEngine
except ImportError as e:
    logger.error(f"Import Error: {e}")
    sys.exit(1)

@dataclass
class SimulationConfig:
    start_year: int = 2010
    end_year: int = 2022
    calibration_end: int = 2019
    monte_carlo_runs: int = 50
    noise_sigma: float = 0.02
    damping: float = 0.95
    growth_nodes: List[str] = None
    
    def __post_init__(self):
        if self.growth_nodes is None:
            self.growth_nodes = [
                "GDP (current US$)", 
                "Exports of goods and services (BoP, current US$)", 
                "Imports of goods and services (BoP, current US$)"
            ]

class SystemicShock:
    def __init__(self, year: int, impacts: Dict[str, float]):
        self.year = year
        self.impacts = impacts
        self.triggered = False

    def apply(self, current_year: int, state):
        if current_year == self.year and not self.triggered:
            logger.info(f">>> TRIGGERING SYSTEMIC SHOCK ({self.year})")
            node_map = {name: i for i, name in enumerate(state.node_ids)}
            for kpi, pct in self.impacts.items():
                if kpi in node_map:
                    idx = node_map[kpi]
                    val = state.values[idx]
                    delta = val * pct
                    state.values[idx] += delta
                    logger.info(f"    {kpi}: {pct:.1%} ({delta/1e9:+.2f}B)")
            self.triggered = True

class BacktestEngine:
    def __init__(self, csv_path: str, config: SimulationConfig):
        self.csv_path = csv_path
        self.config = config
        self.df_history = None
        self.best_gamma = 0.005

    async def load_data(self):
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
            
        df = pd.read_csv(self.csv_path, skiprows=4)
        target_indicators = [
            "GDP (current US$)",
            "Inflation, consumer prices (annual %)",
            "Exports of goods and services (BoP, current US$)",
            "Imports of goods and services (BoP, current US$)"
        ]
        df = df[df['Indicator Name'].isin(target_indicators)]
        id_vars = ['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code']
        val_vars = [c for c in df.columns if c.isdigit()]
        df_long = df.melt(id_vars=id_vars, value_vars=val_vars, var_name='Year', value_name='Value')
        df_pivot = df_long.pivot(index='Year', columns='Indicator Name', values='Value')
        df_pivot.index = df_pivot.index.astype(int)
        self.df_history = df_pivot.sort_index().interpolate(method='linear').bfill()
        logger.info(f"Data loaded: {self.config.start_year}-{self.config.end_year}")

    def build_graph(self, registry, gamma: float):
        # Growth Driver Node
        driver = NodeAgent(
            node_id="Growth_Driver", agent_type="source", domain=0, regime=0,
            embedding=np.zeros(3, dtype=np.float32), stability=1.0, value=0.0
        )
        registry._nodes["Growth_Driver"] = driver

        for target_id in self.config.growth_nodes:
            if target_id in registry._nodes:
                # Self-loop
                sl = EdgeLink(edge_id=f"{target_id}->{target_id}", source=target_id, target=target_id,
                              weight=1.0, stability=1.0, confidence_interval=1.0, regime=0)
                registry._edges[sl.edge_id] = sl
                # Growth link
                gl = EdgeLink(edge_id=f"Growth->{target_id}", source="Growth_Driver", target=target_id,
                              weight=gamma, stability=1.0, confidence_interval=1.0, regime=0)
                registry._edges[gl.edge_id] = gl

    async def run_simulation(self, gamma: float, shocks: List[SystemicShock] = None, seed: int = 42):
        start_row = self.df_history.loc[self.config.start_year]
        variables = self.df_history.columns.tolist()
        registry = AgentRegistry()
        for name in variables:
            hist = self.df_history[name]
            node = NodeAgent(
                node_id=name, agent_type="variable", domain=0, regime=-1,
                embedding=np.zeros(3, dtype=np.float32), stability=0.8, 
                value=float(start_row[name])
            )
            # Attach custom properties dynamically
            node.custom_damping = 0.99 if "GDP" in name else 0.97
            node.min_value = hist.min() * 0.5
            node.max_value = hist.max() * 1.5
            registry._nodes[name] = node

        self.build_graph(registry, gamma)
        env = SimulationEnvironment(
            registry, EnvironmentConfig(damping=self.config.damping, noise_sigma=self.config.noise_sigma, energy_cap=0.0, seed=seed)
        )
        dynamics = DynamicsEngine(env, DynamicsConfig(global_damping=self.config.damping, delta_t=1.0))
        driver_idx = env.state().node_ids.index("Growth_Driver") if "Growth_Driver" in env.state().node_ids else -1
        results = []
        driver_scale = 10.0

        for year in range(self.config.start_year, self.config.end_year+1):
            s = env.state()
            # Growth driver
            if driver_idx >= 0:
                gdp_val = s.values[s.node_ids.index("GDP (current US$)")]
                s.values[driver_idx] = gdp_val * driver_scale

            # Record snapshot
            snapshot = {"Year": year}
            for i, name in enumerate(s.node_ids):
                snapshot[name] = s.values[i]
            results.append(snapshot)

            # Apply shocks
            if shocks:
                for shock in shocks:
                    shock.apply(year, s)

            # Step dynamics
            dynamics.step()

            # Node-specific damping & caps
            for i, node_id in enumerate(s.node_ids):
                node = registry._nodes[node_id]
                if hasattr(node, 'custom_damping'): 
                    s.values[i] *= node.custom_damping
                if hasattr(node, 'min_value'): 
                    s.values[i] = max(s.values[i], node.min_value)
                if hasattr(node, 'max_value'): 
                    s.values[i] = min(s.values[i], node.max_value)

        return pd.DataFrame(results)

    async def calibrate(self):
        logger.info("Starting Calibration...")
        best_gamma, best_damping, best_score = 0.0, 0.95, float('inf')
        actuals = self.df_history.loc[self.config.start_year:self.config.calibration_end]
        targets = ["GDP (current US$)", "Exports of goods and services (BoP, current US$)"]
        gammas = [0.010, 0.012, 0.014, 0.016]
        dampings = [0.99, 1.00, 1.01]

        for d in dampings:
            self.config.damping = d
            for g in gammas:
                sim_df = await self.run_simulation(g, shocks=[], seed=42)
                sim_subset = sim_df[sim_df["Year"] <= self.config.calibration_end]
                score = 0.0
                for t in targets:
                    act, sim = actuals[t].values, sim_subset[t].values
                    rmse = np.sqrt(np.mean((sim-act)**2))
                    score += rmse / np.mean(act)
                if score < best_score:
                    best_score, best_gamma, best_damping = score, g, d

        self.best_gamma = best_gamma
        self.config.damping = best_damping
        logger.info(f"Calibration done. Best Gamma={best_gamma}, Damping={best_damping}")
        return best_gamma, best_damping

    def generate_report(self, agg_df, filename="backtest_report.md"):
        actuals = self.df_history.loc[self.config.start_year:self.config.end_year]
        targets = [
            "GDP (current US$)",
            "Exports of goods and services (BoP, current US$)",
            "Imports of goods and services (BoP, current US$)",
            "Inflation, consumer prices (annual %)"
        ]
        with open(filename, "w") as f:
            f.write("# KShield Backtest Report\n\n")
            f.write(f"## Calibration (Damping={self.config.damping}, Gamma={self.best_gamma:.4f})\n\n")
            for target in targets:
                f.write(f"### {target}\n| Year | Actual | Sim Mean | Sim Std | Error |\n|---|---|---|---|---|\n")
                scale = 1e9 if "US$" in target else 1.0
                unit = "B" if "US$" in target else ""
                for year in range(self.config.start_year, self.config.end_year+1):
                    act = actuals.loc[year][target]
                    sim_mean = agg_df.loc[year][(target, "mean")]
                    sim_std = agg_df.loc[year][(target, "std")]
                    error = (sim_mean-act)/act*100 if abs(act)>1e-6 else 0.0
                    f.write(f"| {year} | {act/scale:.2f}{unit} | {sim_mean/scale:.2f}{unit} | {sim_std/scale:.2f}{unit} | {error:+.2f}% |\n")
                f.write("\n")

class MonteCarloSimulator:
    def __init__(self, engine, config: SimulationConfig, gamma: float, shocks: List[SystemicShock]):
        self.engine = engine
        self.config = config
        self.gamma = gamma
        self.shocks = shocks

    async def run_single_iteration(self, run_id: int):
        df = await self.engine.run_simulation(self.gamma, shocks=self.shocks, seed=42+run_id)
        df["Run"] = run_id
        return df

    async def run_parallel(self):
        logger.info(f"Starting Parallel Monte Carlo ({self.config.monte_carlo_runs} runs)...")
        tasks = [self.run_single_iteration(i) for i in range(self.config.monte_carlo_runs)]
        all_runs = await asyncio.gather(*tasks)
        return pd.concat(all_runs)

@dataclass
class CountryProfile:
    name: str
    csv_path: str
    currency_symbol: str = "US$"
    historical_shocks: List[SystemicShock] = None
    def __post_init__(self):
        if self.historical_shocks is None:
            self.historical_shocks = []

# Kenya profile — data path derived from project structure, never hardcoded
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_KENYA_CSV = _PROJECT_ROOT / "data" / "simulation" / "API_KEN_DS2_en_csv_v2_14659.csv"

KENYA_PROFILE = CountryProfile(
    name="Kenya",
    csv_path=str(_KENYA_CSV),
    historical_shocks=[
        # 2011: Drought & Inflation Spike (Inflation ~14%)
        SystemicShock(2011, {
            "GDP (current US$)": -0.03,
            "Inflation, consumer prices (annual %)": 0.80 # Large relative jump (~4% -> ~14%)
        }),
        # 2013: Westgate Terror Attack (Minor localized shock)
        SystemicShock(2013, {
            "GDP (current US$)": -0.01
        }),
        # 2017: Election + Drought (Moderate)
        SystemicShock(2017, {
            "GDP (current US$)": -0.02,
            "Inflation, consumer prices (annual %)": 0.15
        }),
        # 2020: COVID-19 Pandemic (Major)
        SystemicShock(2020, {
            "GDP (current US$)": -0.05,
            "Exports of goods and services (BoP, current US$)": -0.10,
            "Imports of goods and services (BoP, current US$)": -0.08,
            "Inflation, consumer prices (annual %)": 0.20
        })
    ]
)

async def run_country_backtest(profile: CountryProfile):
    logger.info(f"=== Starting Backtest for {profile.name} ===")
    config = SimulationConfig(start_year=2010, end_year=2022, monte_carlo_runs=50, noise_sigma=0.015)
    engine = BacktestEngine(profile.csv_path, config)
    await engine.load_data()
    await engine.calibrate()
    simulator = MonteCarloSimulator(engine, config, engine.best_gamma, profile.historical_shocks)
    combined = await simulator.run_parallel()
    agg = combined.groupby("Year").agg(["mean","std"])
    engine.generate_report(agg, filename=f"backtest_{profile.name.lower()}_v4.md")
    logger.info(f"Report generated: backtest_{profile.name.lower()}_v4.md")

if __name__ == "__main__":
    asyncio.run(run_country_backtest(KENYA_PROFILE))
