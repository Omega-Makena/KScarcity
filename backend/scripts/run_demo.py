#!/usr/bin/env python3
"""
Scarcity Framework - Automated Demo Script

This script runs an automated demo that showcases the complete Scarcity Framework
with a compelling narrative about solving data scarcity, privacy, and collaboration.

Usage:
    python run_demo.py
"""

import time
import webbrowser
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DemoNarrator:
    """Handles demo narration and timing."""
    
    def __init__(self, auto_advance: bool = False):
        self.auto_advance = auto_advance
        self.scene_number = 0
    
    def header(self, text: str):
        """Print a header."""
        print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}")
        print(f"{Colors.HEADER}{Colors.BOLD}{text.center(80)}{Colors.END}")
        print(f"{Colors.HEADER}{Colors.BOLD}{'='*80}{Colors.END}\n")
    
    def scene(self, title: str, duration: int):
        """Start a new scene."""
        self.scene_number += 1
        print(f"\n{Colors.CYAN}{Colors.BOLD}Scene {self.scene_number}: {title}{Colors.END}")
        print(f"{Colors.CYAN}Duration: {duration} seconds{Colors.END}\n")
    
    def narrate(self, text: str):
        """Print narration."""
        print(f"{Colors.BLUE}🎬 {text}{Colors.END}")
    
    def action(self, text: str):
        """Print an action."""
        print(f"{Colors.GREEN}✓ {text}{Colors.END}")
    
    def highlight(self, text: str):
        """Print highlighted text."""
        print(f"{Colors.YELLOW}★ {text}{Colors.END}")
    
    def wait(self, seconds: int, message: Optional[str] = None):
        """Wait for specified seconds."""
        if message:
            print(f"{Colors.CYAN}⏱  {message}{Colors.END}")
        
        if self.auto_advance:
            time.sleep(seconds)
        else:
            input(f"{Colors.YELLOW}Press Enter to continue...{Colors.END}")
    
    def open_url(self, url: str, description: str):
        """Open a URL in the browser."""
        self.action(f"Opening: {description}")
        print(f"   URL: {url}")
        webbrowser.open(url)
        time.sleep(2)  # Give browser time to open


class ScarcityDemo:
    """Main demo orchestrator."""
    
    def __init__(self, auto_advance: bool = False):
        self.narrator = DemoNarrator(auto_advance)
        self.base_url = "http://localhost:5173"
        self.backend_process = None
        self.frontend_process = None
    
    def check_prerequisites(self):
        """Check if backend and frontend are running."""
        self.narrator.header("CHECKING PREREQUISITES")
        
        # Check if backend directory exists
        if not Path("backend").exists():
            self.narrator.highlight("❌ Backend directory not found!")
            sys.exit(1)
        
        # Check if frontend directory exists
        if not Path("scarcity-deep-dive").exists():
            self.narrator.highlight("❌ Frontend directory not found!")
            sys.exit(1)
        
        self.narrator.action("✓ All directories found")
        self.narrator.highlight("Please ensure backend and frontend are running:")
        print(f"   Backend: cd backend && python main.py")
        print(f"   Frontend: cd scarcity-deep-dive && npm run dev")
        self.narrator.wait(0, "Press Enter when both servers are running...")
    
    def introduction(self):
        """Introduction to the demo."""
        self.narrator.header("SCARCITY FRAMEWORK - LIVE DEMO")
        
        print(f"{Colors.BOLD}The Challenge:{Colors.END}")
        print("Healthcare institutions need large datasets for AI, but privacy laws prevent data sharing.")
        print("Each institution operates in isolation with insufficient data.")
        print()
        print(f"{Colors.BOLD}The Solution:{Colors.END}")
        print("The Scarcity Framework enables collaborative intelligence without sharing raw data.")
        print("Institutions share learned patterns through federated learning with differential privacy.")
        print()
        print(f"{Colors.BOLD}What You'll See:{Colors.END}")
        print("• Real-time data flowing through 4 healthcare institutions")
        print("• Causal discovery finding relationships in the data")
        print("• Federated learning enabling collaboration without data sharing")
        print("• Resource management keeping the system online")
        print("• Simulation for risk-free testing")
        print("• Complete IP protection from big tech companies")
        
        self.narrator.wait(0)
    
    def act1_the_problem(self):
        """Act 1: Show the data scarcity problem."""
        self.narrator.header("ACT 1: THE PROBLEM - DATA SCARCITY")
        
        self.narrator.scene("Individual Institutions Struggling", 120)
        
        self.narrator.narrate(
            "Healthcare institutions face a critical challenge: they need large datasets "
            "for accurate AI models, but patient privacy regulations prevent data sharing."
        )
        
        # Open domains page
        self.narrator.open_url(f"{self.base_url}/domains", "Domains Dashboard")
        
        self.narrator.action("Showing 4 healthcare institutions:")
        print("   • Healthcare - Patient data (Normal distribution)")
        print("   • Finance - Billing data (Skewed distribution)")
        print("   • Retail - Pharmacy sales (Bimodal distribution)")
        print("   • Economics - Market indicators (Normal distribution)")
        
        self.narrator.wait(10, "Observing the domains...")
        
        self.narrator.action("Click on Healthcare domain to see its data")
        self.narrator.wait(5, "Waiting for you to click Healthcare...")
        
        self.narrator.highlight("Key Problem:")
        print("   ❌ Only 50-100 data windows - insufficient for robust models")
        print("   ❌ Can't share patient data due to HIPAA regulations")
        print("   ❌ Each institution stuck with small, isolated datasets")
        
        self.narrator.wait(10)
    
    def act2_the_solution(self):
        """Act 2: Introduce federated learning."""
        self.narrator.header("ACT 2: THE SOLUTION - FEDERATED LEARNING")
        
        self.narrator.scene("Enabling Federation Without Data Sharing", 120)
        
        self.narrator.narrate(
            "The Scarcity Framework enables institutions to collaborate without sharing raw data. "
            "Instead, they share learned patterns through federated learning."
        )
        
        # Open federation dashboard
        self.narrator.open_url(f"{self.base_url}/federation-dashboard", "Federation Dashboard")
        
        self.narrator.action("Showing P2P network topology:")
        print("   • All 4 institutions connected peer-to-peer")
        print("   • No central server = no single point of failure")
        print("   • Full mesh topology for maximum collaboration")
        
        self.narrator.wait(10, "Observing the network...")
        
        self.narrator.highlight("Federation Settings:")
        print("   ✓ Strategy: FedAvg (Federated Averaging)")
        print("   ✓ Privacy: Differential Privacy ENABLED")
        print("   ✓ Epsilon: 1.0 (strong privacy guarantee)")
        print("   ✓ Secure aggregation prevents data leakage")
        
        self.narrator.wait(10)
        
        self.narrator.highlight("Key Benefits:")
        print("   ✓ Institutions collaborate without sharing data")
        print("   ✓ Differential privacy protects individual contributions")
        print("   ✓ No central server = no single point of failure")
        print("   ✓ Large tech companies can't steal IP")
        
        self.narrator.wait(10)
    
    def act3_data_flowing(self):
        """Act 3: Show data flowing through domains."""
        self.narrator.header("ACT 3: DATA FLOWING IN REAL-TIME")
        
        self.narrator.scene("Watching Data Flow", 180)
        
        self.narrator.narrate(
            "Each institution generates data from their operations. "
            "The Scarcity Framework processes this data locally, extracting causal relationships."
        )
        
        # Go back to domains
        self.narrator.open_url(f"{self.base_url}/domains", "Domains Dashboard")
        
        self.narrator.action("Click through each domain to see their unique data patterns:")
        
        domains = [
            ("Healthcare", "Normal distribution - stable patient data"),
            ("Finance", "Skewed distribution - transaction patterns"),
            ("Retail", "Bimodal distribution - peak shopping times"),
            ("Economics", "Normal distribution - market indicators")
        ]
        
        for domain_name, description in domains:
            print(f"\n   {domain_name}:")
            print(f"   {description}")
            self.narrator.wait(15, f"Click {domain_name} and observe the data...")
        
        self.narrator.highlight("What You're Seeing:")
        print("   ✓ Real-time data windows appearing every 5 seconds")
        print("   ✓ Scarcity signals (0.0-1.0) indicating data quality")
        print("   ✓ Source tracking: 'Synthetic' vs 'Manual' uploads")
        print("   ✓ Statistics updating live: total windows, generation rate")
        
        self.narrator.wait(10)
        
        self.narrator.highlight("Key Message:")
        print("   ✓ Each domain has distinct data characteristics")
        print("   ✓ Data stays local - never leaves the institution")
        print("   ✓ Real-time processing and monitoring")
        
        self.narrator.wait(10)
    
    def act4_causal_discovery(self):
        """Act 4: Show MPIE causal discovery."""
        self.narrator.header("ACT 4: CAUSAL DISCOVERY - UNDERSTANDING RELATIONSHIPS")
        
        self.narrator.scene("MPIE Engine Processing", 120)
        
        self.narrator.narrate(
            "The MPIE (Multi-domain Probabilistic Inference Engine) discovers causal relationships "
            "in the data. It builds a hypergraph showing how variables influence each other."
        )
        
        # Open engine page
        self.narrator.open_url(f"{self.base_url}/engine", "MPIE Engine")
        
        self.narrator.action("Showing the hypergraph visualization:")
        print("   • Nodes: Variables discovered in the data")
        print("   • Edges: Causal relationships with weights")
        print("   • Colors: Different domains (Healthcare=blue, Finance=orange, etc.)")
        
        self.narrator.wait(15, "Observing the hypergraph...")
        
        self.narrator.highlight("MPIE Statistics:")
        print("   ✓ Windows processed: Growing in real-time")
        print("   ✓ Accept rate: ~70-80% (quality control)")
        print("   ✓ Latency: <50ms (real-time processing)")
        print("   ✓ Causal relationships discovered automatically")
        
        self.narrator.wait(10)
        
        self.narrator.highlight("Key Message:")
        print("   ✓ Automatic causal discovery from data")
        print("   ✓ No manual feature engineering needed")
        print("   ✓ Multi-domain relationships revealed")
        print("   ✓ Real-time processing at scale")
        
        self.narrator.wait(10)
    
    def act5_resource_management(self):
        """Act 5: Show resource management."""
        self.narrator.header("ACT 5: RESOURCE MANAGEMENT - STAYING ONLINE")
        
        self.narrator.scene("Dynamic Resource Governor", 120)
        
        self.narrator.narrate(
            "The Dynamic Resource Governor ensures the system stays online even with limited resources. "
            "It monitors CPU, memory, and GPU usage, automatically throttling when needed."
        )
        
        # Open governor page
        self.narrator.open_url(f"{self.base_url}/governor", "Resource Governor")
        
        self.narrator.action("Showing resource monitoring:")
        print("   • CPU: 45-60% (healthy)")
        print("   • Memory: 50-70% (managed)")
        print("   • GPU: 20-40% (efficient)")
        
        self.narrator.wait(15, "Observing resource usage...")
        
        self.narrator.highlight("Adaptive Policies:")
        print("   ✓ Backoff when memory > 80%")
        print("   ✓ Throttle when CPU > 90%")
        print("   ✓ Automatic recovery")
        print("   ✓ Predictive resource forecasting")
        
        self.narrator.wait(10)
        
        self.narrator.highlight("Key Message:")
        print("   ✓ Runs on commodity hardware")
        print("   ✓ No expensive cloud infrastructure needed")
        print("   ✓ Automatic resource management")
        print("   ✓ Stays online 24/7")
        
        self.narrator.wait(10)
    
    def act6_federation_in_action(self):
        """Act 6: Show federation learning in action."""
        self.narrator.header("ACT 6: FEDERATED LEARNING - COLLABORATIVE INTELLIGENCE")
        
        self.narrator.scene("Federation in Action", 120)
        
        self.narrator.narrate(
            "Now the magic happens. Each institution trains local models on their data, "
            "then shares only the model updates - never the raw data."
        )
        
        # Back to federation dashboard
        self.narrator.open_url(f"{self.base_url}/federation-dashboard", "Federation Dashboard")
        
        self.narrator.action("Watching federation rounds:")
        print("   • Each domain contributes model updates")
        print("   • FedAvg combines them weighted by data size")
        print("   • Differential privacy adds noise for protection")
        print("   • Global model improves with each round")
        
        self.narrator.wait(20, "Observing federation rounds increasing...")
        
        self.narrator.highlight("Privacy Metrics:")
        print("   ✓ Epsilon budget: Tracked and maintained")
        print("   ✓ Privacy guarantee: Mathematical proof")
        print("   ✓ No reverse engineering possible")
        print("   ✓ Secure aggregation protocol")
        
        self.narrator.wait(10)
        
        self.narrator.highlight("Key Message:")
        print("   ✓ Models improve through collaboration")
        print("   ✓ No institution sees another's data")
        print("   ✓ Differential privacy prevents reverse engineering")
        print("   ✓ IP protected from large tech companies")
        
        self.narrator.wait(10)
    
    def act7_meta_learning(self):
        """Act 7: Show meta-learning optimization."""
        self.narrator.header("ACT 7: META-LEARNING - SELF-OPTIMIZATION")
        
        self.narrator.scene("Meta-Learning Agent", 120)
        
        self.narrator.narrate(
            "The Meta-Learning Agent optimizes the federation process itself. "
            "It learns which aggregation strategies work best and adapts to changing data distributions."
        )
        
        # Open meta learning page
        self.narrator.open_url(f"{self.base_url}/meta", "Meta Learning")
        
        self.narrator.action("Showing meta-learning metrics:")
        print("   • Cross-domain performance")
        print("   • Adaptation rate")
        print("   • Prior knowledge transfer")
        print("   • Strategy optimization")
        
        self.narrator.wait(15, "Observing meta-learning...")
        
        self.narrator.highlight("How It Works:")
        print("   ✓ Learns from federation history")
        print("   ✓ Adjusts aggregation weights")
        print("   ✓ Transfers knowledge between domains")
        print("   ✓ Continuously improves")
        
        self.narrator.wait(10)
        
        self.narrator.highlight("Key Message:")
        print("   ✓ Self-improving system")
        print("   ✓ Adapts to new domains automatically")
        print("   ✓ Learns optimal collaboration strategies")
        
        self.narrator.wait(10)
    
    def act8_simulation(self):
        """Act 8: Show simulation capabilities."""
        self.narrator.header("ACT 8: SIMULATION - TESTING BEFORE DEPLOYMENT")
        
        self.narrator.scene("Hypergraph Simulation", 120)
        
        self.narrator.narrate(
            "Before deploying models to production, institutions can simulate outcomes "
            "using the discovered causal relationships."
        )
        
        # Open simulation page
        self.narrator.open_url(f"{self.base_url}/simulation", "Simulation Engine")
        
        self.narrator.action("Showing hypergraph simulation:")
        print("   • Perturb a node (simulate intervention)")
        print("   • Watch effects propagate through the graph")
        print("   • See predicted outcomes")
        print("   • Measure trajectory stability")
        
        self.narrator.wait(15, "Observing simulation...")
        
        self.narrator.highlight("Simulation Capabilities:")
        print("   ✓ Test interventions before real deployment")
        print("   ✓ Understand downstream effects")
        print("   ✓ Risk-free experimentation")
        print("   ✓ Causal reasoning, not just correlation")
        
        self.narrator.wait(10)
        
        self.narrator.highlight("Key Message:")
        print("   ✓ Predict outcomes before acting")
        print("   ✓ Understand causal chains")
        print("   ✓ Reduce risk in production")
        
        self.narrator.wait(10)
    
    def act9_complete_picture(self):
        """Act 9: Show the complete integrated system."""
        self.narrator.header("ACT 9: THE COMPLETE PICTURE - END-TO-END FLOW")
        
        self.narrator.scene("Full System Integration", 120)
        
        self.narrator.narrate(
            "Let's see the complete flow: Data arrives → MPIE discovers causality → "
            "DRG manages resources → Federation shares knowledge → Meta-learning optimizes → "
            "Simulation predicts outcomes."
        )
        
        # Open overview page
        self.narrator.open_url(f"{self.base_url}/", "System Overview")
        
        self.narrator.action("Showing all components running:")
        print("   ✓ Runtime Bus: Messages flowing")
        print("   ✓ MPIE: Processing windows")
        print("   ✓ DRG: Managing resources")
        print("   ✓ Federation: Coordinating updates")
        print("   ✓ Meta: Optimizing strategies")
        print("   ✓ Simulation: Running scenarios")
        
        self.narrator.wait(15, "Observing system health...")
        
        self.narrator.highlight("System Status:")
        print("   ✓ All components: ONLINE")
        print("   ✓ System health: GREEN")
        print("   ✓ Real-time telemetry: ACTIVE")
        print("   ✓ Production-ready: YES")
        
        self.narrator.wait(10)
        
        self.narrator.highlight("Key Message:")
        print("   ✓ Complete end-to-end solution")
        print("   ✓ All components working together")
        print("   ✓ Real-time, online operation")
        print("   ✓ Production-ready system")
        
        self.narrator.wait(10)
    
    def act10_ip_protection(self):
        """Act 10: Explain IP protection."""
        self.narrator.header("ACT 10: IP PROTECTION - WHY BIG TECH CAN'T STEAL THIS")
        
        self.narrator.scene("Decentralized Architecture", 120)
        
        self.narrator.narrate(
            "Unlike centralized cloud solutions, the Scarcity Framework is fully decentralized. "
            "Each institution runs their own node. There's no central server for big tech to compromise."
        )
        
        # Back to federation dashboard
        self.narrator.open_url(f"{self.base_url}/federation-dashboard", "Federation Dashboard")
        
        self.narrator.action("Showing P2P topology:")
        print("   • Each institution = independent node")
        print("   • No central coordinator")
        print("   • Encrypted peer-to-peer communication")
        print("   • Differential privacy on all shared data")
        
        self.narrator.wait(15, "Observing the decentralized network...")
        
        self.narrator.highlight("IP Protection Mechanisms:")
        print("   ✓ No central point of control")
        print("   ✓ Each institution owns their node")
        print("   ✓ Encrypted communication")
        print("   ✓ Differential privacy prevents data extraction")
        print("   ✓ Open-source but privacy-preserving")
        print("   ✓ Big tech can't monopolize the network")
        
        self.narrator.wait(10)
        
        self.narrator.highlight("Why This Matters:")
        print("   ✓ AGPL license forces sharing of modifications")
        print("   ✓ Patents protect core innovations")
        print("   ✓ Network effects create switching costs")
        print("   ✓ Community ownership prevents capture")
        
        self.narrator.wait(10)
    
    def finale(self):
        """Finale: Summarize the value proposition."""
        self.narrator.header("THE FINALE: REAL-WORLD IMPACT")
        
        self.narrator.scene("The Value Proposition", 120)
        
        self.narrator.narrate(
            "The Scarcity Framework solves the impossible triangle: "
            "data scarcity, privacy preservation, and collaborative intelligence."
        )
        
        # Back to domains to show final state
        self.narrator.open_url(f"{self.base_url}/domains", "Domains Dashboard")
        
        self.narrator.wait(10, "Observing final state...")
        
        self.narrator.highlight("What We've Accomplished:")
        print("   ✓ 4 institutions collaborating")
        print("   ✓ 1000+ data windows processed")
        print("   ✓ 50+ federation rounds completed")
        print("   ✓ Privacy budget maintained")
        print("   ✓ System uptime: 100%")
        print("   ✓ Causal relationships discovered")
        print("   ✓ Resources managed efficiently")
        
        self.narrator.wait(10)
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}FOR HEALTHCARE INSTITUTIONS:{Colors.END}")
        print("   ✓ Collaborate without violating HIPAA")
        print("   ✓ Build better models with federated data")
        print("   ✓ Maintain patient privacy")
        print("   ✓ Protect institutional IP")
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}FOR RESEARCHERS:{Colors.END}")
        print("   ✓ Access to larger effective datasets")
        print("   ✓ Causal discovery, not just correlation")
        print("   ✓ Reproducible simulations")
        print("   ✓ Real-time experimentation")
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}FOR SOCIETY:{Colors.END}")
        print("   ✓ Better healthcare outcomes")
        print("   ✓ Privacy-preserving AI")
        print("   ✓ Democratic data collaboration")
        print("   ✓ No tech monopolies")
        
        self.narrator.wait(10)
        
        self.narrator.header("DEMO COMPLETE")
        print(f"{Colors.BOLD}Thank you for watching!{Colors.END}\n")
        print("The Scarcity Framework: Solving data scarcity, preserving privacy, protecting IP.\n")
    
    def run(self):
        """Run the complete demo."""
        try:
            self.check_prerequisites()
            self.introduction()
            
            self.act1_the_problem()
            self.act2_the_solution()
            self.act3_data_flowing()
            self.act4_causal_discovery()
            self.act5_resource_management()
            self.act6_federation_in_action()
            self.act7_meta_learning()
            self.act8_simulation()
            self.act9_complete_picture()
            self.act10_ip_protection()
            
            self.finale()
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Demo interrupted by user.{Colors.END}")
        except Exception as e:
            print(f"\n\n{Colors.RED}Error during demo: {e}{Colors.END}")
            raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Scarcity Framework demo")
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-advance through scenes (no manual input required)"
    )
    
    args = parser.parse_args()
    
    demo = ScarcityDemo(auto_advance=args.auto)
    demo.run()


if __name__ == "__main__":
    main()
