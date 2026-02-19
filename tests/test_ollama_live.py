"""Quick live test of the Ollama LLM pipeline."""
import asyncio
import json
from dataclasses import asdict

async def test():
    from kshiked.pulse.llm.ollama import OllamaProvider
    from kshiked.pulse.llm.config import OllamaConfig, AnalysisTask

    config = OllamaConfig()
    task = AnalysisTask.THREAT_CLASSIFICATION
    model = config.get_model_for_task(task)
    print(f"Default model: {model}")
    print(f"Embedding model: {config.embedding_model}")
    print(f"Read timeout: {config.read_timeout}s")

    async with OllamaProvider(config) as provider:
        # Test 1: English threat tweet
        text1 = (
            "These politicians are stealing from Kenyans. If the Finance Bill passes, "
            "people will take to the streets. Enough is enough. #RejectFinanceBill"
        )
        print("\n--- TEST 1: English threat tweet ---")
        print(f"Input: {text1[:80]}...")
        result1 = await provider.classify_threat(text1)
        print(f"Tier: {result1.tier}")
        print(f"Confidence: {result1.confidence}")
        print(f"Base Risk: {result1.base_risk}")
        print(f"Reasoning: {result1.reasoning[:200]}")

        # Test 2: Sheng/Swahili mixed tweet
        text2 = (
            "Hii serikali ni useless sana. Watu wa mtaa wako na njaa "
            "but wanaongeza tax. Tutachoma hii jiji"
        )
        print("\n--- TEST 2: Sheng/Swahili tweet ---")
        print(f"Input: {text2}")
        result2 = await provider.classify_threat(text2)
        print(f"Tier: {result2.tier}")
        print(f"Confidence: {result2.confidence}")
        print(f"Base Risk: {result2.base_risk}")
        print(f"Reasoning: {result2.reasoning[:200]}")

        # Test 3: Low-threat informational post
        text3 = "Kenya's GDP grew by 5.2% in Q3 2025. Tourism sector recovering well after reopening."
        print("\n--- TEST 3: Low-threat informational ---")
        print(f"Input: {text3}")
        result3 = await provider.classify_threat(text3)
        print(f"Tier: {result3.tier}")
        print(f"Confidence: {result3.confidence}")
        print(f"Base Risk: {result3.base_risk}")
        print(f"Reasoning: {result3.reasoning[:200]}")

        print("\n=== All inference tests complete ===")

if __name__ == "__main__":
    asyncio.run(test())
