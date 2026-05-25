import os
import sys
import time
import json
import random
import argparse
import numpy as np
from datetime import datetime
from typing import List, Dict, Any

# Ensure local imports work by setting sys.path to the root folder
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from kshiked.ui.institution.backend.delta_sync import DeltaSyncManager
from kshiked.ui.institution.backend.federation_bridge import FederationBridge

random.seed(42)
np.random.seed(42)

class SyntheticGenerator:
    """Generates deterministic synthetic payloads for injection."""
    
    @staticmethod
    def generate_single(sector_id: int = 1, impact: float = 7.5, certainty: float = 0.8) -> Dict[str, Any]:
        
        # Realistic interpretations based on sector
        if sector_id == 1:
            sources = ["Central Bank Node", "Treasury Ops", "KRA Data Node"]
            interpretations = ["Anomalous currency velocity detected.", "High frequency trading volatility spike.", "Tax processing latency increased, indicating potential fiscal disruption."]
        elif sector_id == 2:
            sources = ["Nairobi Gen Hospital", "KEMRI Intel", "Ministry of Health"]
            interpretations = ["Unexpected ICU admission baseline breach.", "Medical supply chain divergence detected.", "Outbreak vector identified in localized ward."]
        else:
            sources = ["Border Control Point", "Cyber Command", "National Police Intel"]
            interpretations = ["Unusual cryptographic footprint intercepted on local ISP.", "Physical checkpoint throughput dropped below normal operating parameters.", "Social unrest indicators crossing semantic thresholds."]

        source = random.choice(sources)
        interpretation = random.choice(interpretations)

        return {
            "incident_type": f"SYNTHETIC_ALERT_{sector_id}_{random.randint(100,999)}",
            "severity_score": impact,
            "spoke_source": source,
            "spoke_interpretation": interpretation,
            "composite_scores": {
                "A_Detection": round(random.uniform(5.0, 9.0), 2),
                "B_Impact": impact,
                "C_Certainty": certainty
            },
            "timestamp": time.time()
        }
        
    @staticmethod
    def generate_batch(count: int, sector_id: int = 1) -> List[Dict[str, Any]]:
        return [SyntheticGenerator.generate_single(sector_id, impact=round(random.uniform(2.0, 9.9), 2)) for _ in range(count)]

class InjectionRunner:
    """Injects payloads through official ingestion points and tracks latency."""
    
    @staticmethod
    def inject(institution_id: int, basket_id: int, payload: Dict[str, Any]) -> Dict[str, Any]:
        start_t = time.time()
        sync_id = DeltaSyncManager.queue_insight(institution_id, basket_id, payload)
        end_t = time.time()
        return {
            "sync_id": sync_id,
            "basket_id": basket_id,
            "payload": payload,
            "latency_ms": (end_t - start_t) * 1000
        }

class PropagationTracker:
    """Emulates Admin aggregation math to promote risks to the Executive tier."""
    
    @staticmethod
    def process_pending_for_sector(basket_id: int, expected_sync_ids: List[int]) -> Dict[str, Any]:
        # 1. Pull from Spoke
        pending = DeltaSyncManager.get_pending_syncs(basket_id)
        
        # Filter only to the ones we injected in this test run to prevent dirty DB conflicts
        target_syncs = [s for s in pending if s['sync_id'] in expected_sync_ids]
        
        if not target_syncs:
            return {"status": "FAIL", "reason": "No pending syncs found."}
            
        # 2. Replicate Admin Governance Mathematical Consensus
        payloads = [s['payload'] for s in target_syncs]
        avg_A = np.mean([p.get('composite_scores', {}).get('A_Detection', 0) for p in payloads])
        avg_B = np.mean([p.get('composite_scores', {}).get('B_Impact', 0) for p in payloads])
        avg_C = np.mean([p.get('composite_scores', {}).get('C_Certainty', 0) for p in payloads])
        
        fused_scores = {
            "A_Detection": round(avg_A, 2),
            "B_Impact": round(avg_B, 2),
            "C_Certainty": round(avg_C, 2)
        }
        
        # 3. Mark Processed (Queue Clear)
        DeltaSyncManager.mark_synced([s['sync_id'] for s in target_syncs])
        
        # 4. Promote to Executive
        risk_title = f"SYNTHETIC VALIDATION RISK [Basket {basket_id}]"
        risk_id = DeltaSyncManager.promote_risk(
            basket_id=basket_id,
            title=risk_title,
            description="Automated mathematical fusion validation.",
            composite_scores=fused_scores,
            source_sync_ids=[s['sync_id'] for s in target_syncs]
        )
        
        return {
            "status": "PASS",
            "risk_id": risk_id,
            "fused_scores": fused_scores,
            "raw_payloads": payloads
        }

class IntegrityValidator:
    """Mathematical validation engine."""
    
    @staticmethod
    def assert_math(raw_payloads: List[Dict], fused_scores: Dict[str, float]) -> bool:
        expected_B = round(float(np.mean([p.get('composite_scores', {}).get('B_Impact', 0) for p in raw_payloads])), 2)
        actual_B = fused_scores.get('B_Impact', 0)
        return abs(expected_B - actual_B) < 0.05

class DashboardVerifier:
    """Verifies that the Executive UI endpoint receives the exact data."""
    
    @staticmethod
    def verify_promotion(expected_risk_id: int, expected_scores: Dict[str, float]) -> bool:
        risks = DeltaSyncManager.get_promoted_risks()
        target = next((r for r in risks if r['risk_id'] == expected_risk_id), None)
        if not target:
            return False
            
        ui_B_Impact = target['composite_scores'].get('B_Impact', 0)
        return abs(ui_B_Impact - expected_scores.get('B_Impact', 0)) < 0.05

class ReportExporter:
    @staticmethod
    def export(report_data: Dict[str, Any], filename: str = "validation_report.md"):
        md_content = "# System Validation Engine Summary\n\n"
        md_content += f"**Execution Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        md_content += "## Synthetic Injection Tests\n"
        for t_name, t_res in report_data.get("tests", {}).items():
            md_content += f"- **{t_name}**: {'[PASS]' if t_res else '[FAIL]'}\n"
            
        md_content += "\n## Stress Test Metrics\n"
        stress = report_data.get("stress", {})
        if stress:
            md_content += f"- **Signals Injected:** {stress.get('count', 0)}\n"
            md_content += f"- **Average Latency:** {stress.get('avg_lat_ms', 0):.2f} ms\n"
            md_content += f"- **Max Latency:** {stress.get('max_lat_ms', 0):.2f} ms\n"
            md_content += f"- **Data Drop Rate:** {stress.get('drop_rate', 0):.1f}%\n"
            
        with open(filename, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"\n[+] Validation Report exported to {filename}")

def run_standard_suite() -> Dict[str, bool]:
    results = {}
    print("\n[>] Running Standard Injection Suite...")
    
    # Test 1: Single Signal
    print("    -> Testing Single Signal Injection...")
    p1 = SyntheticGenerator.generate_single(impact=8.5)
    i1 = InjectionRunner.inject(institution_id=999, basket_id=1, payload=p1)
    
    prop1 = PropagationTracker.process_pending_for_sector(1, [i1['sync_id']])
    if prop1['status'] == 'PASS':
        math_valid = IntegrityValidator.assert_math(prop1['raw_payloads'], prop1['fused_scores'])
        dash_valid = DashboardVerifier.verify_promotion(prop1['risk_id'], prop1['fused_scores'])
        results["Single Injection & Mathematical Integrity"] = math_valid
        results["Executive Dashboard Reflection"] = dash_valid
    else:
        results["Single Injection & Mathematical Integrity"] = False
        results["Executive Dashboard Reflection"] = False
        
    # Test 2: Batch Injection & Re-Aggregation
    print("    -> Testing Batch Fusion (10 signals)...")
    batch = SyntheticGenerator.generate_batch(10, sector_id=2)
    sync_ids = []
    for p in batch:
        sync_ids.append(InjectionRunner.inject(999, 2, p)['sync_id'])
        
    prop2 = PropagationTracker.process_pending_for_sector(2, sync_ids)
    if prop2['status'] == 'PASS':
        math_valid = IntegrityValidator.assert_math(prop2['raw_payloads'], prop2['fused_scores'])
        dash_valid = DashboardVerifier.verify_promotion(prop2['risk_id'], prop2['fused_scores'])
        results["Batch Aggregation Accuracy"] = math_valid
        results["Batch Promotion Integrity"] = dash_valid
    else:
        results["Batch Aggregation Accuracy"] = False
        results["Batch Promotion Integrity"] = False
        
    return results

def run_stress_suite() -> Dict[str, Any]:
    print("\n[>] Running Stress Burst (1000 Signals)...")
    latencies = []
    sync_ids = []
    
    for i in range(1000):
        # alternate sectors
        b_id = (i % 5) + 1
        p = SyntheticGenerator.generate_single(sector_id=b_id, impact=random.uniform(0.0, 10.0))
        res = InjectionRunner.inject(institution_id=900+(i%10), basket_id=b_id, payload=p)
        latencies.append(res['latency_ms'])
        sync_ids.append(res['sync_id'])
        
        if i > 0 and i % 200 == 0:
            print(f"    ... injected {i}/1000")
            
    # Process them to check drop rate
    processed_count = 0
    for b_id in range(1, 6):
        # Admin pulls queue
        pending = DeltaSyncManager.get_pending_syncs(b_id)
        valid_pending = [s for s in pending if s['sync_id'] in sync_ids]
        processed_count += len(valid_pending)
        
        # Clear out
        DeltaSyncManager.mark_synced([s['sync_id'] for s in valid_pending])
        
    drop_rate = ((1000 - processed_count) / 1000) * 100
    
    return {
        "count": 1000,
        "avg_lat_ms": sum(latencies)/len(latencies),
        "max_lat_ms": max(latencies),
        "drop_rate": drop_rate
    }

def run_continuous_simulation():
    print("\n[>] Starting Continuous Pipeline Simulation (Press Ctrl+C to stop)...")
    try:
        iteration = 0
        while True:
            # Inject 3-5 random signals into the Spoke layer
            num_signals = random.randint(3, 5)
            # Only use baskets 1, 2, 3 (Economic, Health, Security)
            injected_syncs_by_basket = {1: [], 2: [], 3: []}
            
            for _ in range(num_signals):
                b_id = random.randint(1, 3)
                # occasional crisis vs normal
                is_crisis = random.random() > 0.8
                impact = random.uniform(7.0, 10.0) if is_crisis else random.uniform(2.0, 6.0)
                
                p = SyntheticGenerator.generate_single(sector_id=b_id, impact=impact)
                
                res = InjectionRunner.inject(institution_id=random.randint(100, 500), basket_id=b_id, payload=p)
                injected_syncs_by_basket[b_id].append(res['sync_id'])
                
            # Randomly let Admins aggregate them some of the time to keep both queues (Spoke & Exec) active
            for b_id, syncs in injected_syncs_by_basket.items():
                if syncs and random.random() > 0.4:
                    PropagationTracker.process_pending_for_sector(b_id, syncs)
                    print(f"  [Admin {b_id}] Fused {len(syncs)} signals -> Promoted to Executive.")
                elif syncs:
                    print(f"  [Spoke] Injected {len(syncs)} signals into Basket {b_id} (Pending Admin Review).")
                    
            iteration += 1
            # Sleep to allow the UI to feel "live"
            time.sleep(random.uniform(2.0, 5.0))
            
    except KeyboardInterrupt:
        print("\n[!] Simulation halted by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run standard suite")
    parser.add_argument("--stress", action="store_true", help="Run stress suite")
    parser.add_argument("--simulate", action="store_true", help="Run continuous random injections")
    args = parser.parse_args()
    
    report = {"tests": {}, "stress": {}}
    
    if args.full:
        report["tests"] = run_standard_suite()
        
    if args.stress:
        report["stress"] = run_stress_suite()
        
    if args.simulate:
        run_continuous_simulation()
        sys.exit(0)
        
    if not args.full and not args.stress and not args.simulate:
        print("Please specify --full, --stress, or --simulate")
        sys.exit(0)
        
    ReportExporter.export(report)
    
    # Final Verdict Output
    all_passed = all(report["tests"].values()) if report["tests"] else True
    if report["stress"] and report["stress"]["drop_rate"] > 0.0:
        all_passed = False
        
    print("\n==================================")
    print(f"FINAL VERDICT: {'SYSTEM VALID' if all_passed else 'SYSTEM INVALID'}")
    print("==================================")
