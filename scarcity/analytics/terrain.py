"""
Terrain Generator for Scarcity Economic Engine.

Implements the user's specific definition of "Terrain":
1. Surface Height = System Response (Performance/Welfare)
2. Axes (Manifold) = Policy Position (Fiscal/Monetary Stance)
3. Walking = Time/Simulation

This module provides the logic to computing this surface by running
batches of parallel simulations.
"""
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Any

# Configure logger
logger = logging.getLogger(__name__)

class TerrainGenerator:
    """
    Generates response surfaces (terrains) by sweeping over policy parameters.
    """

    def __init__(self, engine):
        """
        Initialize with a trained EconomicDiscoveryEngine.
        
        Args:
            engine: The initialized engine instance to use for simulations.
        """
        self.engine = engine

    def generate_surface(self, 
                         initial_state: Dict[str, float], 
                         x_policy: str, 
                         y_policy: str, 
                         z_response: str, 
                         x_range: Tuple[float, float], 
                         y_range: Tuple[float, float], 
                         steps: int = 10,
                         time_horizon: int = 20,
                         max_points: Optional[int] = 400) -> Dict[str, Any]:
        """
        Generates the terrain data by running a grid of simulations.
        
        Args:
            initial_state: The starting point for all simulations.
            x_policy: Policy variable for X axis.
            y_policy: Policy variable for Y axis.
            z_response: Response variable for Z axis (height).
            x_range: (min, max) for X.
            y_range: (min, max) for Y.
            steps: Grid resolution (steps x steps).
            time_horizon: How long to run each simulation (years).
            
        Returns:
            Dictionary with X, Y, Z matrices and overlays.
            Keys:
                - x: Array of X coordinate values.
                - y: Array of Y coordinate values.
                - z: 2D array of Z values (height).
                - overlays: Dictionary containing 'stability' and 'risk' matrices.
        """
        
        # 1. Setup Grid
        if max_points is not None and steps * steps > max_points:
            target = max(2, int(np.floor(np.sqrt(max_points))))
            if target < steps:
                logger.warning(
                    "TerrainGenerator: reducing steps from %d to %d to cap simulations at %d",
                    steps, target, max_points
                )
            steps = target

        x_vals = np.linspace(x_range[0], x_range[1], steps)
        y_vals = np.linspace(y_range[0], y_range[1], steps)
        
        # Matrices for Plotly Surface
        # Note: Plotly Surface expects Z as 2D array, X and Y as 1D arrays matching dimensions
        z_matrix = np.zeros((steps, steps))
        stability_matrix = np.zeros((steps, steps))
        risk_matrix = np.zeros((steps, steps))
        
        count = 0
        
        # 2. Run Simulations
        # TODO: Parallelize this if performance is an issue.
        # For now, sequential execution is safer for state management.
        
        for i, y_val in enumerate(y_vals):
            for j, x_val in enumerate(x_vals):
                # Run Simulation for this grid point
                sim = self.engine.get_simulation_handle()
                sim.set_initial_state(initial_state)
                
                # Apply Policy "Stance"
                # We enforce this policy for the duration of the simulation
                sim.set_policy(x_policy, float(x_val))
                sim.set_policy(y_policy, float(y_val))
                
                # Run
                sim.run(time_horizon)
                
                # Analyze Result
                hist = pd.DataFrame(sim.history)
                
                # --- Z Height: System Response ---
                # Integrating valuable performance over time (e.g. cumulative GDP or final GDP)
                # The user defined terrain as "Response to policy over time".
                # A good proxy is the AVERAGE or FINAL value found on the path.
                # Average is smoother and represents "sustained performance".
                if z_response in hist.columns:
                    # Use mean for a "level" terrain, or sum for "accumulated wealth".
                    # Mean is safer to avoid duration bias.
                    val = hist[z_response].mean()
                else:
                    val = 0.0
                z_matrix[i, j] = val
                
                # --- Overlays: Stability & Risk ---
                
                # 1. Stability (Smoothness/Volatility)
                # Lower std dev = Higher Stability.
                if z_response in hist.columns:
                    volatility = hist[z_response].std()
                    stability_matrix[i, j] = volatility
                
                # 2. Risk (Distress/Crash)
                # Measure how often we hit constraints or crashed.
                # If engine has stress metrics, use them.
                if hasattr(sim, 'meta_history') and sim.meta_history:
                     # Calculate max stress experienced
                     stresses = [m['metrics'].get('system_stress', 0) for m in sim.meta_history]
                     risk_val = max(stresses) if stresses else 0.0
                     risk_matrix[i, j] = risk_val
                     
                     # 3. Opacity / Reachability
                     # If stress > 0.8, we consider it "Unreachable" / "Unstable"
                     # We can mask Z with NaN to make it disappear, or handle in viz
                     if risk_val > 0.8:
                         z_matrix[i, j] = np.nan # Create hole in terrain
                
                count += 1
                
        return {
            "x": x_vals,
            "y": y_vals,
            "z": z_matrix,
            "overlays": {
                "stability": stability_matrix,
                "risk": risk_matrix
            }
        }
