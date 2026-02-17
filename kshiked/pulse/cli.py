#!/usr/bin/env python
"""
KShield Pulse CLI - Command Line Interface

Usage:
    python -m kshiked.pulse.cli analyze "text to analyze"
    python -m kshiked.pulse.cli report --output report.html
    python -m kshiked.pulse.cli demo
    python -m kshiked.pulse.cli scrape --platform twitter --query "Kenya"
"""

import argparse
import asyncio
import json
import sys
import os
from datetime import datetime
from typing import Optional


def cmd_analyze(args):
    """Analyze text for threats."""
    from kshiked.pulse import PulseSensor, compute_threat_report
    
    sensor = PulseSensor()
    
    # Process text(s)
    if args.file:
        with open(args.file, 'r') as f:
            texts = f.readlines()
    else:
        texts = [args.text] if args.text else []
    
    if not texts:
        print("Error: Provide --text or --file")
        return 1
    
    print(f"\n{'='*60}")
    print("KShield Pulse - Threat Analysis")
    print(f"{'='*60}\n")
    
    for text in texts:
        text = text.strip()
        if not text:
            continue
            
        print(f"Text: {text[:80]}{'...' if len(text) > 80 else ''}")
        
        detections = sensor.process_text(text)
        
        if detections:
            print(f"  Signals detected: {len(detections)}")
            for d in detections:
                print(f"    - {d.signal_id.name}: {d.intensity:.2f} (conf: {d.confidence:.2f})")
        else:
            print("  No threat signals detected")
        print()
    
    # Generate threat report
    sensor.update_state()
    report = compute_threat_report(sensor.state, sensor._signal_history)
    
    print(f"\n{'='*60}")
    print(f"Overall Threat Level: {report.overall_threat_level}")
    print(f"{'='*60}\n")
    
    print("Threat Indices:")
    print(f"  Polarization:         {report.polarization.value:.2f} [{report.polarization.severity}]")
    print(f"  Legitimacy Erosion:   {report.legitimacy_erosion.value:.2f} [{report.legitimacy_erosion.severity}]")
    print(f"  Mobilization:         {report.mobilization_readiness.value:.2f} [{report.mobilization_readiness.severity}]")
    print(f"  Elite Cohesion:       {report.elite_cohesion.value:.2f} [{report.elite_cohesion.severity}]")
    print(f"  Info Warfare:         {report.information_warfare.value:.2f} [{report.information_warfare.severity}]")
    print(f"  Security Friction:    {report.security_friction.value:.2f} [{report.security_friction.severity}]")
    print(f"  Economic Cascade:     {report.economic_cascade.value:.2f} [{report.economic_cascade.severity}]")
    print(f"  Ethnic Tension:       {report.ethnic_tension.avg_tension:.2f} [{report.ethnic_tension.severity}]")
    
    if report.priority_alerts:
        print("\nPriority Alerts:")
        for alert in report.priority_alerts:
            print(f"  {alert}")
    
    # Output JSON if requested
    if args.json:
        print("\nJSON Output:")
        print(json.dumps(report.to_dict(), indent=2, default=str))
    
    # Save report if requested
    if args.output:
        from kshiked.pulse.visualization import generate_html_report
        generate_html_report(report.to_dict(), output_path=args.output)
        print(f"\nReport saved to: {args.output}")
    
    return 0


def cmd_report(args):
    """Generate a threat report."""
    from kshiked.pulse import PulseSensor, compute_threat_report
    from kshiked.pulse.visualization import generate_html_report, visualize_threat_report
    
    sensor = PulseSensor()
    
    # Demo data if no input
    demo_texts = [
        "Rise up Kenya! Take the streets tomorrow!",
        "These corrupt leaders must pay! Death to traitors!",
        "The election was stolen! Fake government!",
        "Our people are suffering while they feast!",
        "Join the protest at Uhuru Park. Share this message!",
    ]
    
    print("Processing demo threat data...")
    for text in demo_texts:
        sensor.process_text(text)
    
    sensor.update_state()
    report = compute_threat_report(sensor.state, sensor._signal_history)
    
    # Generate visualizations
    output_dir = args.output_dir or "."
    os.makedirs(output_dir, exist_ok=True)
    
    outputs = visualize_threat_report(report.to_dict(), output_dir)
    
    print("\nGenerated files:")
    for name, path in outputs.items():
        print(f"  {name}: {path}")
    
    return 0


def cmd_demo(args):
    """Run interactive demo."""
    from kshiked.pulse import PulseSensor, compute_threat_report
    
    print("\n" + "="*60)
    print("  KShield Pulse - Interactive Demo")
    print("="*60 + "\n")
    
    sensor = PulseSensor()
    print(f"✓ PulseSensor initialized with {len(sensor._detectors)} signal detectors\n")
    
    # Demo texts
    demo_texts = [
        "We are suffering! Fuel prices are killing us!",
        "Rise up Kenya! Take the streets tomorrow at Uhuru Park!",
        "These cockroaches in government must be eliminated!",
        "The election was stolen! Ruto is not my president!",
        "Our Kikuyu land must be protected from them!",
        "Bank run tomorrow! Currency collapse coming!",
        "Join Telegram group for protest coordination",
        "Police are refusing orders. Military dissent reported.",
    ]
    
    print("Processing sample threat content...")
    for i, text in enumerate(demo_texts, 1):
        detections = sensor.process_text(text)
        signals = [d.signal_id.name for d in detections]
        print(f"  [{i}] {text[:50]}...")
        if signals:
            print(f"       → Signals: {', '.join(signals)}")
    
    # Update state and compute report
    sensor.update_state()
    report = compute_threat_report(sensor.state, sensor._signal_history)
    
    print("\n" + "="*60)
    print(f"  THREAT LEVEL: {report.overall_threat_level}")
    print("="*60)
    
    print("\nPhase 1 - High Priority Indices:")
    print(f"  Polarization:         {report.polarization.value:.2f} [{report.polarization.severity}]")
    print(f"  Legitimacy Erosion:   {report.legitimacy_erosion.value:.2f} [{report.legitimacy_erosion.severity}]")
    print(f"  Mobilization:         {report.mobilization_readiness.value:.2f} [{report.mobilization_readiness.severity}]")
    
    print("\nPhase 2 - Medium Priority:")
    print(f"  Elite Cohesion:       {report.elite_cohesion.value:.2f} [{report.elite_cohesion.severity}]")
    print(f"  Info Warfare:         {report.information_warfare.value:.2f} [{report.information_warfare.severity}]")
    print(f"  Security Friction:    {report.security_friction.value:.2f} [{report.security_friction.severity}]")
    
    print("\nPhase 3 - Economic/Ethnic:")
    print(f"  Economic Cascade:     {report.economic_cascade.value:.2f} [{report.economic_cascade.severity}]")
    print(f"  Ethnic Tension:       {report.ethnic_tension.avg_tension:.2f} [{report.ethnic_tension.severity}]")
    print(f"  Highest Pair:         {report.ethnic_tension.highest_tension_pair}")
    
    if report.priority_alerts:
        print("\n⚠️ Priority Alerts:")
        for alert in report.priority_alerts:
            print(f"  {alert}")
    
    print("\n" + "="*60)
    print("  Demo Complete!")
    print("="*60 + "\n")
    
    return 0


def cmd_scrape(args):
    """Run scrapers (placeholder)."""
    print("Scraping functionality requires API credentials.")
    print("Configure your .env file with the required credentials.")
    print(f"\nPlatform: {args.platform}")
    print(f"Query: {args.query}")
    print(f"Limit: {args.limit}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="KShield Pulse - National Threat Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m kshiked.pulse.cli demo
  python -m kshiked.pulse.cli analyze --text "Rise up Kenya!"
  python -m kshiked.pulse.cli analyze --file threats.txt --output report.html
  python -m kshiked.pulse.cli report --output-dir ./reports
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze text for threats")
    analyze_parser.add_argument("--text", "-t", help="Text to analyze")
    analyze_parser.add_argument("--file", "-f", help="File with texts (one per line)")
    analyze_parser.add_argument("--output", "-o", help="Output HTML report path")
    analyze_parser.add_argument("--json", "-j", action="store_true", help="Output JSON")
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate threat report")
    report_parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    report_parser.set_defaults(func=cmd_report)
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run interactive demo")
    demo_parser.set_defaults(func=cmd_demo)
    
    # Scrape command
    scrape_parser = subparsers.add_parser("scrape", help="Run scrapers")
    scrape_parser.add_argument("--platform", "-p", default="twitter", help="Platform to scrape")
    scrape_parser.add_argument("--query", "-q", default="Kenya", help="Search query")
    scrape_parser.add_argument("--limit", "-l", type=int, default=100, help="Max posts")
    scrape_parser.set_defaults(func=cmd_scrape)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
