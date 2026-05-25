"""Debug: test LLM extraction for title-only input."""
import asyncio
import json
from kshiked.pulse.llm.config import OllamaConfig, AnalysisTask
from kshiked.pulse.llm.ollama import OllamaProvider
from kshiked.pulse.llm.policy_extractor import PolicyExtractor, BILL_EXTRACTION_SYSTEM

async def test():
    config = OllamaConfig.single_model("qwen2.5:3b")
    async with OllamaProvider(config=config) as provider:
        extractor = PolicyExtractor(provider)
        
        # Test 1: extract_from_title
        print("=" * 60)
        print("TEST 1: extract_from_title('Finance Bill 2026')")
        print("=" * 60)
        bill = await extractor.extract_from_title("Finance Bill 2026")
        print(f"Title: {bill.title}")
        print(f"Summary: {bill.summary}")
        print(f"Provisions: {bill.provision_count}")
        print(f"Sectors: {bill.sectors}")
        print(f"Severity: {bill.total_severity}")
        for p in bill.provisions:
            print(f"  - {p.clause_id}: {p.description[:80]} (sev={p.severity})")

        # Test 2: raw _generate_json to see what LLM returns
        print("\n" + "=" * 60)
        print("TEST 2: Raw _generate_json output")
        print("=" * 60)
        prompt = (
            'The user mentioned this Kenyan policy: "SHIF Phase 2"\n\n'
            'Based on your knowledge of Kenyan policy and the current 2025-2026 landscape,\n'
            'provide a structured analysis as if you were reading the bill.\n'
            'Include likely provisions, affected groups, severity, and hashtags.\n'
            'If this is a known bill (Finance Bill, Housing Levy, SHIF, etc.), be specific.\n\n'
            'Return the full JSON structure with provisions.'
        )
        data = await provider._generate_json(
            prompt, BILL_EXTRACTION_SYSTEM,
            task=AnalysisTask.POLICY_IMPACT,
        )
        print(f"Type: {type(data)}")
        if data:
            print(f"Keys: {list(data.keys())}")
            print(f"Provisions: {len(data.get('provisions', []))}")
            print(json.dumps(data, indent=2)[:3000])
        else:
            print("RETURNED NONE!")

        # Test 3: raw text generation to see what LLM actually outputs
        print("\n" + "=" * 60)
        print("TEST 3: Raw text output")
        print("=" * 60)
        raw = await provider._generate_text(
            prompt, BILL_EXTRACTION_SYSTEM,
            task=AnalysisTask.POLICY_IMPACT,
        )
        print(f"Raw output ({len(raw)} chars):")
        print(raw[:2000])

asyncio.run(test())
print("\nDONE")
