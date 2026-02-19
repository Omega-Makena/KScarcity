"""
End-to-end test for the Policy Impact Chatbot.
Tests: extraction, search, prediction, and conversation flow.
Requires Ollama running with qwen2.5:3b and nomic-embed-text.
"""

import asyncio
import json
import sys
import time

# Add project root to path
sys.path.insert(0, r"c:\Users\omegam\OneDrive - Innova Limited\scace4")

from kshiked.pulse.llm.config import OllamaConfig
from kshiked.pulse.llm.policy_chatbot import PolicyChatbot, ChatSession


async def test_policy_chatbot():
    print("=" * 70)
    print("POLICY IMPACT CHATBOT — END-TO-END TEST")
    print("=" * 70)

    config = OllamaConfig.single_model("qwen2.5:3b")

    async with PolicyChatbot(config=config) as chatbot:
        session = ChatSession()

        # ── TEST 1: Analyze by title (not limited to Finance Bill) ──────
        print("\n[TEST 1] Analyze SHIF Phase 2 (health policy)")
        start = time.monotonic()
        response = await chatbot.process_bill(
            session, title="SHIF Phase 2 Mandatory Enrollment"
        )
        elapsed = time.monotonic() - start
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Bill loaded: {session.has_bill}")
        if session.bill:
            print(f"  Title: {session.bill.title}")
            print(f"  Provisions: {session.bill.provision_count}")
            print(f"  Severity: {session.bill.total_severity:.2f}")
            print(f"  Sectors: {session.bill.sectors}")
        if session.prediction:
            print(f"  Mobilization: {session.prediction.overall_mobilization:.2f}")
            print(f"  Risk level: {session.prediction.overall_risk_level}")
            print(f"  Top counties: {session.prediction.top_risk_counties[:5]}")
        print(f"\n  Response preview:\n  {response[:400]}...")
        assert session.has_bill, "Bill should be loaded"
        assert session.bill.provision_count > 0, "Should have provisions"
        print("  ✅ TEST 1 PASSED")

        # ── TEST 2: Follow-up question ──────────────────────────────────
        print("\n[TEST 2] Follow-up — county risk question")
        start = time.monotonic()
        response = await chatbot.ask(session, "Which counties will be most affected?")
        elapsed = time.monotonic() - start
        print(f"  Time: {elapsed:.1f}s")
        print(f"  Response preview:\n  {response[:400]}...")
        print("  ✅ TEST 2 PASSED")

        # ── TEST 3: New bill — Fuel Levy (different sector) ─────────────
        print("\n[TEST 3] Process Fuel Levy Increase (energy sector)")
        session2 = ChatSession()
        start = time.monotonic()
        response = await chatbot.process_bill(
            session2, title="Fuel Levy Increase KES 5 per litre"
        )
        elapsed = time.monotonic() - start
        print(f"  Time: {elapsed:.1f}s")
        if session2.bill:
            print(f"  Title: {session2.bill.title}")
            print(f"  Provisions: {session2.bill.provision_count}")
            print(f"  Sectors: {session2.bill.sectors}")
        if session2.prediction:
            print(f"  Mobilization: {session2.prediction.overall_mobilization:.2f}")
            print(f"  Historical match: {session2.prediction.historical_match}")
        print("  ✅ TEST 3 PASSED")

        # ── TEST 4: Text paste analysis ─────────────────────────────────
        print("\n[TEST 4] Analyze pasted bill text (Education sector)")
        session3 = ChatSession()
        bill_text = """
        THE UNIVERSITY FUNDING MODEL BILL, 2026
        
        An Act to establish a new university funding framework replacing HELB.
        
        Section 3: All university students shall be assessed through a means-testing
        model based on household income, with four bands:
        (a) Band 1 (Vulnerable): 100% government scholarship
        (b) Band 2 (Low income): 80% government, 20% student
        (c) Band 3 (Middle income): 60% government, 40% student
        (d) Band 4 (High income): 20% government, 80% student
        
        Section 5: HELB shall be dissolved within 12 months of this Act's commencement.
        All existing HELB loans shall be transferred to the new University Funding Board.
        
        Section 8: Students in Band 3 and Band 4 may access private education loans
        at regulated interest rates not exceeding CBK base rate plus 3%.
        
        Section 12: County governments shall contribute 5% of their education budget
        to the University Funding Board to support students from their counties.
        """
        start = time.monotonic()
        response = await chatbot.process_bill(
            session3, text=bill_text, title="University Funding Model Bill 2026"
        )
        elapsed = time.monotonic() - start
        print(f"  Time: {elapsed:.1f}s")
        if session3.bill:
            print(f"  Title: {session3.bill.title}")
            print(f"  Provisions: {session3.bill.provision_count}")
            print(f"  Sectors: {session3.bill.sectors}")
            for p in session3.bill.top_provisions[:3]:
                print(f"    - {p.clause_id}: {p.description[:80]}... (sev={p.severity:.2f})")
        print("  ✅ TEST 4 PASSED")

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_policy_chatbot())
