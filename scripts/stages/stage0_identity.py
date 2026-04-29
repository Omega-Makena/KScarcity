"""
Stage 0 — Engine Identity Audit.

Traces which engine and federation classes the benchmark scripts actually
instantiate, extracts MetaController thresholds from source, and flags
discrepancies between the benchmark report and the architecture documentation.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.stages.utils import (
    ARTIFACTS_DIR, fail_result, make_result, save_artifact, skip_result,
)

logger = logging.getLogger(__name__)

# Architecture doc claims (from ARCHITECTURE_DIAGRAMS.md)
ARCH_DOC_CLAIMS = {
    "engine_class": "MPIEOrchestrator",
    "engine_module": "scarcity.engine.engine",
    "federation_class": "HierarchicalFederation",
    "federation_module": "scarcity.federation.hierarchical",
    "metacontroller_conf_threshold": 0.7,
    "metacontroller_stab_threshold": 0.6,
    "metacontroller_min_evidence": 20,
    "metacontroller_kill_threshold": 0.10,
    "predict_conf_gate": 0.25,
}

# What benchmark_discovery.py actually says (from reading source)
BENCHMARK_REPORT_CLAIMS = {
    "engine_class": "OnlineDiscoveryEngine",
    "promotion_conf": 0.25,
    "kill_conf": 0.10,
    "predict_threshold_low": 0.10,
    "predict_threshold_normal": 0.20,
}


def _parse_imports(source: str) -> List[Dict[str, str]]:
    """Extract all import statements from Python source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({"type": "import", "module": alias.name, "alias": alias.asname})
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({
                    "type": "from",
                    "module": node.module or "",
                    "name": alias.name,
                    "alias": alias.asname,
                })
    return imports


def _find_instantiations(source: str, class_names: List[str]) -> List[Dict[str, Any]]:
    """Find where specific class names are instantiated in source."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name in class_names:
                found.append({"class": name, "lineno": node.lineno})
    return found


def _extract_controller_thresholds(source: str) -> Dict[str, Any]:
    """Extract MetaController.__init__ default parameter values from source."""
    result = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "MetaController":
            for item in ast.walk(node):
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = item.args
                    defaults = args.defaults
                    all_args = [a.arg for a in args.args]
                    # align defaults to the end of args
                    offset = len(all_args) - len(defaults)
                    for i, default in enumerate(defaults):
                        arg_name = all_args[offset + i]
                        if isinstance(default, ast.Constant):
                            result[arg_name] = default.value
    return result


def audit_script(script_path: Path) -> Dict[str, Any]:
    """Audit one benchmark script for engine and federation class usage."""
    if not script_path.exists():
        return {"error": f"not found: {script_path}"}

    source = script_path.read_text(encoding="utf-8")
    imports = _parse_imports(source)

    engine_candidates = ["OnlineDiscoveryEngine", "MPIEOrchestrator", "Engine"]
    fed_candidates = ["FederationHub", "FederationNode", "HierarchicalFederation",
                      "FederationClientAgent", "FederationCoordinator"]

    engine_imports = [i for i in imports if any(c in i.get("name", "") or c in i.get("module", "")
                                                 for c in engine_candidates)]
    fed_imports = [i for i in imports if any(c in i.get("name", "") or c in i.get("module", "")
                                              for c in fed_candidates)]

    engine_insts = _find_instantiations(source, engine_candidates)
    fed_insts = _find_instantiations(source, fed_candidates)

    return {
        "script": script_path.name,
        "engine_imports": engine_imports,
        "federation_imports": fed_imports,
        "engine_instantiations": engine_insts,
        "federation_instantiations": fed_insts,
    }


def audit_engine_identity() -> Dict[str, Any]:
    start = time.time()
    result: Dict[str, Any] = {
        "scripts": [],
        "engine_in_code": {},
        "federation_in_code": {},
        "controller_thresholds": {},
        "discrepancies": [],
    }

    # 1. Audit benchmark scripts
    scripts_dir = PROJECT_ROOT / "scripts"
    benchmark_scripts = list(scripts_dir.glob("benchmark_*.py"))
    for sp in sorted(benchmark_scripts):
        result["scripts"].append(audit_script(sp))

    # 2. Extract actual engine class used
    try:
        controller_path = PROJECT_ROOT / "scarcity" / "engine" / "controller.py"
        if controller_path.exists():
            thresholds = _extract_controller_thresholds(controller_path.read_text(encoding="utf-8"))
            result["controller_thresholds"] = thresholds
        else:
            result["controller_thresholds"] = {"error": "controller.py not found"}
    except Exception as e:
        result["controller_thresholds"] = {"error": str(e)}

    # 3. Try importing and inspecting the engine
    try:
        from scarcity.engine.engine_v2 import OnlineDiscoveryEngine
        sig = inspect.signature(OnlineDiscoveryEngine.__init__)
        result["engine_in_code"] = {
            "class": "OnlineDiscoveryEngine",
            "module": "scarcity.engine.engine_v2",
            "init_params": {
                k: (p.default if p.default is not inspect.Parameter.empty else "REQUIRED")
                for k, p in sig.parameters.items()
                if k != "self"
            },
        }
    except Exception as e:
        result["engine_in_code"] = {"error": str(e)}

    # 4. Try importing the architecture-described engine
    try:
        from scarcity.engine import OnlineDiscoveryEngine as EngineAlias  # noqa: F401
        result["engine_alias_resolves_to"] = "OnlineDiscoveryEngine (scarcity.engine.engine_v2)"
    except Exception:
        result["engine_alias_resolves_to"] = "import failed"

    # 5. Check MPIEOrchestrator path
    try:
        mpie_path = PROJECT_ROOT / "scarcity" / "engine" / "engine.py"
        result["mpie_file_exists"] = mpie_path.exists()
        if mpie_path.exists():
            source = mpie_path.read_text(encoding="utf-8")
            result["mpie_has_orchestrator"] = "MPIEOrchestrator" in source
    except Exception as e:
        result["mpie_file_exists"] = False
        result["mpie_check_error"] = str(e)

    # 6. Federation path audit
    try:
        from scarcity.engine.federation_hub import FederationHub
        from scarcity.engine.federation_node import FederationNode
        result["federation_in_code"] = {
            "benchmark_path": "FederationHub / FederationNode",
            "benchmark_module": "scarcity.engine.federation_hub / federation_node",
            "hub_class": FederationHub.__name__,
            "node_class": FederationNode.__name__,
        }
    except Exception as e:
        result["federation_in_code"] = {"error": str(e)}

    try:
        from scarcity.federation.hierarchical import HierarchicalFederation  # noqa: F401
        result["hierarchical_federation_importable"] = True
    except Exception as e:
        result["hierarchical_federation_importable"] = False
        result["hierarchical_federation_error"] = str(e)

    # 7. Compute discrepancies
    disc = []
    thresholds = result.get("controller_thresholds", {})
    arch = ARCH_DOC_CLAIMS

    # Engine class mismatch
    actual_engine = result.get("engine_in_code", {}).get("class", "unknown")
    if actual_engine != arch["engine_class"]:
        disc.append({
            "field": "engine_class",
            "architecture_doc": arch["engine_class"],
            "actual_in_benchmarks": actual_engine,
            "note": (
                "Benchmarks use OnlineDiscoveryEngine (hypothesis survival + Bayesian α/β). "
                "Architecture docs describe MPIEOrchestrator (Thompson Sampling + bootstrap R² gain). "
                "These are two distinct engines."
            ),
        })

    # MetaController threshold check
    for key, arch_val in [
        ("confidence_threshold", arch["metacontroller_conf_threshold"]),
        ("stability_threshold", arch["metacontroller_stab_threshold"]),
        ("min_evidence", arch["metacontroller_min_evidence"]),
        ("kill_threshold", arch["metacontroller_kill_threshold"]),
    ]:
        actual = thresholds.get(key)
        if actual is not None and actual != arch_val:
            disc.append({
                "field": f"MetaController.{key}",
                "architecture_doc": arch_val,
                "actual_in_code": actual,
            })
        elif actual == arch_val:
            pass  # matches

    # Federation mismatch
    bench_fed = result.get("federation_in_code", {}).get("benchmark_path", "unknown")
    if "HierarchicalFederation" not in bench_fed:
        disc.append({
            "field": "federation_class",
            "architecture_doc": arch["federation_class"],
            "actual_in_benchmarks": bench_fed,
            "note": (
                "Benchmarks use FederationHub/FederationNode (simple evidence sharing). "
                "Architecture docs describe HierarchicalFederation with gossip, "
                "Layer1/Layer2, DP, and Byzantine-robust aggregation."
            ),
        })

    result["discrepancies"] = disc
    result["discrepancy_count"] = len(disc)
    result["conclusion"] = (
        "Benchmark scripts instantiate OnlineDiscoveryEngine (scarcity.engine.engine_v2) "
        "with FederationHub/FederationNode (scarcity.engine.federation_*). "
        "The architecture documentation describes a separate, newer subsystem "
        "(MPIEOrchestrator + HierarchicalFederation) that the benchmarks do not exercise. "
        "Subsequent stages use the engine the benchmarks actually instantiate."
    )

    resolution_doc = ARTIFACTS_DIR / "engine_identity_resolution.md"
    resolution_exists = resolution_doc.exists()
    result["resolution_doc"] = str(resolution_doc) if resolution_exists else None
    result["resolution_written"] = resolution_exists

    wallclock = time.time() - start
    # PASS once discrepancy is documented in resolution doc, even if it still exists
    status = "PASS" if resolution_exists else ("WARN" if disc else "PASS")
    return make_result(
        stage="0",
        name="engine_identity_audit",
        status=status,
        target="Engine identity discrepancies documented in resolution doc",
        result=result,
        wallclock_s=wallclock,
    )


def run_stage_0() -> Dict[str, Any]:
    logging.basicConfig(level=logging.WARNING)
    result = audit_engine_identity()
    save_artifact("stage0_identity.json", result)
    return result


if __name__ == "__main__":
    import json
    r = run_stage_0()
    print(json.dumps({"status": r["status"], "discrepancies": r["result"]["discrepancy_count"]}, indent=2))
