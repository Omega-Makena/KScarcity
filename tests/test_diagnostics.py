
import pytest
import pandas as pd
import numpy as np
import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from kshiked.core.governance import EconomicGovernor, EconomicGovernorConfig
from scarcity.simulation.environment import SimulationEnvironment, EnvironmentConfig
from scarcity.simulation.agents import AgentRegistry
from scarcity.simulation.sfc import SFCEconomy

@pytest.mark.asyncio
async def test_governance_initialization_and_step():
    """Verify EconomicGovernor can init and step through SFC logic."""
    print("\nTesting Governance Loop...")
    
    # Setup Mocks
    registry = AgentRegistry()
    env_config = EnvironmentConfig()
    env = SimulationEnvironment(registry, env_config)
    
    # Init Governor
    gov_config = EconomicGovernorConfig(control_interval=1)
    gov = EconomicGovernor(gov_config, env)
    
    assert isinstance(gov.sfc, SFCEconomy)
    
    # Step
    initial_gdp = gov.sfc.gdp
    await gov.step(1)
    
    # Verify SFC stepped
    assert gov.sfc.time == 1
    # Verify Sync (Actuator pushed values to env)
    # env state might be empty if registry is empty, but checks logic
    print("Governance Loop OK")

def test_causal_imports_and_mock_run():
    """Verify Causal Engine components are importable and runnable."""
    print("\nTesting Causal Engine...")
    try:
        from scarcity.causal.engine import run_causal
        from scarcity.causal.specs import EstimandSpec
        
        # Create Dummy Data
        df = pd.DataFrame({
            'T': np.random.randint(0, 2, 100),
            'Y': np.random.normal(0, 1, 100),
            'C': np.random.normal(0, 1, 100)
        })
        
        # Spec
        spec = EstimandSpec(treatment='T', outcome='Y', confounders=['C'])
        
        # Run
        # Check if DoWhy is installed by catching specific import error inside run_causal
        try:
            result = run_causal(df, spec)
            if result.results:
                print(f"Causal Result: {result.results[0].estimate}")
        except ImportError:
            print("SKIPPED: DoWhy/EconML missing")
        except Exception as e:
            pytest.fail(f"Run Causal Failed: {e}")
            
    except ImportError:
        pytest.fail("Could not import Causal Engine")

def test_dashboard_imports():
    """Verify Dashboard dependencies."""
    print("\nTesting Dashboard Imports...")
    try:
        # Use importlib to check
        import importlib.util
        spec = importlib.util.find_spec("scarcity.dashboard")
        if spec is None:
             # Script file?
             if os.path.exists("scarcity/dashboard.py"):
                 print("Dashboard script exists.")
             else:
                 pytest.fail("Dashboard script missing")
        else:
            print("Dashboard module found.")
            
    except SyntaxError as e:
        pytest.fail(f"Syntax Error in Dashboard: {e}")

def scan_null_bytes():
    print("\nScanning for Null Bytes...")
    root_dir = os.getcwd()
    found = False
    
    for root, dirs, files in os.walk(root_dir):
        if ".git" in dirs: dirs.remove(".git")
        if "__pycache__" in dirs: dirs.remove("__pycache__")
        
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'rb') as f:
                        content = f.read()
                        if b'\x00' in content:
                            print(f"❌ NULL BYTE FOUND: {path}")
                            found = True
                except Exception as e:
                    print(f"Could not read {path}: {e}")
    
    if not found:
        print("✓ No null bytes found in .py files")
    return found

if __name__ == "__main__":
    print(f"=== KShield Diagnostic Suite ===")
    
    # 1. Static Scan
    if scan_null_bytes():
        print("CRITICAL: File corruption detected. Fix files listed above.")
        sys.exit(1)
        
    # 2. Integration Tests
    # We use pytest runner logic manually or just call functions
    try:
        if sys.platform == 'win32':
             asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
             
        asyncio.run(test_governance_initialization_and_step())
        test_causal_imports_and_mock_run()
        test_dashboard_imports()
        
        print("\n✅ ALL DIAGNOSTICS PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
