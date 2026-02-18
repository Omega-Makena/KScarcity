"""
Policy Impact Chatbot — Conversational Engine

Orchestrates the full chatbot pipeline:
1. Accept bill input (text, PDF, URL, or just a title)
2. Extract provisions via PolicyExtractor
3. Search evidence via PolicySearchEngine
4. Generate predictions via PolicyPredictor
5. Handle follow-up questions with conversation memory

Manages session state, chat history, and context window.

Usage:
    chatbot = PolicyChatbot()
    await chatbot.initialize()
    
    # Process a bill
    response = await chatbot.process_bill(text="...", title="Finance Bill 2026")
    
    # Follow-up questions
    response = await chatbot.ask("Which counties should we watch?")
    
    # Clean up
    await chatbot.close()
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .config import OllamaConfig
from .ollama import OllamaProvider
from .embeddings import OllamaEmbeddings
from .policy_extractor import PolicyExtractor, BillAnalysis
from .policy_search import PolicySearchEngine, SearchResults
from .policy_predictor import PolicyPredictor, ImpactPrediction

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Chat Message Models
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ChatMessage:
    """Single message in the conversation."""
    role: str               # "user" or "assistant"
    content: str            # Display text (markdown)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata can contain: bill, prediction, evidence, visualizations, etc.


@dataclass
class ChatSession:
    """Full conversation session with context memory."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    messages: List[ChatMessage] = field(default_factory=list)
    bill: Optional[BillAnalysis] = None
    prediction: Optional[ImpactPrediction] = None
    evidence: Optional[SearchResults] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def has_bill(self) -> bool:
        return self.bill is not None

    @property
    def has_prediction(self) -> bool:
        return self.prediction is not None

    def add_user_message(self, content: str):
        self.messages.append(ChatMessage(role="user", content=content))

    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None):
        self.messages.append(ChatMessage(
            role="assistant", content=content, metadata=metadata or {}
        ))


# ═══════════════════════════════════════════════════════════════════════════
# Chatbot Engine
# ═══════════════════════════════════════════════════════════════════════════

class PolicyChatbot:
    """
    Main chatbot engine — orchestrates extraction, search, and prediction.
    
    Designed to work with Streamlit session state.
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig.single_model("qwen2.5:3b")
        self._provider: Optional[OllamaProvider] = None
        self._embeddings: Optional[OllamaEmbeddings] = None
        self._extractor: Optional[PolicyExtractor] = None
        self._searcher: Optional[PolicySearchEngine] = None
        self._predictor: Optional[PolicyPredictor] = None
        self._initialized = False

    # ─── Lifecycle ──────────────────────────────────────────────────────

    async def initialize(self):
        """Initialize all components. Call once before use."""
        if self._initialized:
            return

        self._provider = OllamaProvider(config=self.config)
        await self._provider.__aenter__()

        self._embeddings = OllamaEmbeddings(config=self.config)
        await self._embeddings.__aenter__()

        self._extractor = PolicyExtractor(self._provider)
        self._searcher = PolicySearchEngine(self._embeddings)
        self._predictor = PolicyPredictor(self._provider)

        # Health check
        healthy = await self._provider.health_check()
        if not healthy:
            logger.warning("Ollama not reachable — chatbot will have limited functionality")

        self._initialized = True
        logger.info("PolicyChatbot initialized")

    async def close(self):
        """Clean up resources."""
        if self._provider:
            await self._provider.__aexit__(None, None, None)
        if self._embeddings:
            await self._embeddings.__aexit__(None, None, None)
        self._initialized = False

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, *args):
        await self.close()

    # ─── Bill Processing ────────────────────────────────────────────────

    async def process_bill(
        self,
        session: ChatSession,
        text: str = "",
        title: str = "",
        pdf_bytes: Optional[bytes] = None,
        url: str = "",
    ) -> str:
        """
        Process a bill through the full pipeline.
        
        Accepts one of: text, pdf_bytes, url, or just title.
        Returns formatted markdown response.
        """
        if not self._initialized:
            await self.initialize()

        # Determine input method and extract
        if pdf_bytes:
            session.add_user_message(f"[Uploaded PDF: {title or 'document'}]")
            bill = await self._extractor.extract_from_pdf(pdf_bytes, title=title)
        elif url:
            session.add_user_message(f"Analyze this policy: {url}")
            bill = await self._extractor.extract_from_url(url, title=title)
        elif text and len(text.strip()) > 50:
            display = text[:100] + "..." if len(text) > 100 else text
            session.add_user_message(f"Analyze this policy:\n\n{display}")
            bill = await self._extractor.extract_from_text(text, title=title)
        elif title:
            session.add_user_message(f"Analyze: {title}")
            bill = await self._extractor.extract_from_title(title)
        else:
            return "Please provide a bill title, text, PDF, or URL to analyze."

        session.bill = bill

        if not bill.provisions:
            # Retry once — LLM sometimes returns empty on cold start
            logger.info("No provisions on first try — retrying extraction")
            if title:
                bill = await self._extractor.extract_from_title(title)
            elif text:
                bill = await self._extractor.extract_from_text(text, title=title)
            session.bill = bill

        if not bill.provisions:
            response = (
                f"**{bill.title}**\n\n"
                f"{bill.summary or 'Could not extract provisions from this input.'}\n\n"
                "Try providing the full bill text or a more specific title."
            )
            session.add_assistant_message(response)
            return response

        # Search for evidence
        try:
            evidence = await self._searcher.search_all(bill)
            session.evidence = evidence
        except Exception as e:
            logger.warning(f"Evidence search failed: {e}")
            evidence = None

        # Generate prediction
        try:
            prediction = await self._predictor.predict(bill, evidence)
            session.prediction = prediction
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            prediction = None

        # Format response
        response = self._format_bill_analysis(bill, prediction, evidence)
        session.add_assistant_message(response, metadata={
            "type": "bill_analysis",
            "bill_title": bill.title,
            "provision_count": bill.provision_count,
            "has_prediction": prediction is not None,
        })
        return response

    # ─── Follow-up Questions ─────────────────────────────────────────────

    async def ask(self, session: ChatSession, question: str) -> str:
        """
        Handle a follow-up question about the current bill.
        
        If no bill loaded, tries to interpret as a bill title.
        """
        if not self._initialized:
            await self.initialize()

        session.add_user_message(question)

        # If no bill loaded, check if this is a bill reference
        if not session.has_bill:
            # Try to interpret as a bill title
            if any(kw in question.lower() for kw in [
                "bill", "levy", "tax", "act", "policy", "amendment",
                "shif", "nhif", "housing", "finance", "education",
                "fuel", "digital", "security", "gazette",
            ]):
                response = await self.process_bill(
                    session, title=question
                )
                return response
            else:
                response = (
                    "No bill loaded yet. You can:\n"
                    "- Paste bill text in the input area\n"
                    "- Upload a PDF\n"
                    "- Enter a bill URL\n"
                    "- Or just type a bill name (e.g., 'Finance Bill 2026')"
                )
                session.add_assistant_message(response)
                return response

        # Route to appropriate handler based on question type
        q_lower = question.lower()

        # County-specific question
        if any(w in q_lower for w in ["county", "counties", "nairobi", "mombasa", "kisumu", "region"]):
            return await self._handle_county_question(session, question)

        # Provision drill-down
        if any(w in q_lower for w in ["section", "clause", "provision", "about"]):
            return await self._handle_provision_question(session, question)

        # Comparison question
        if any(w in q_lower for w in ["compare", "similar", "history", "historical", "2024", "happened"]):
            return await self._handle_comparison_question(session, question)

        # Search for more evidence
        if any(w in q_lower for w in ["tweet", "evidence", "show me", "find", "search"]):
            return await self._handle_search_question(session, question)

        # General question — use LLM with full context
        return await self._handle_general_question(session, question)

    # ─── Question Handlers ──────────────────────────────────────────────

    async def _handle_county_question(self, session: ChatSession, question: str) -> str:
        """Handle county-specific questions."""
        answer = await self._predictor.answer_question(
            question, session.bill, session.prediction, session.evidence
        )

        if not answer:
            # Fallback: format county data from prediction
            if session.prediction and session.prediction.county_risks:
                lines = ["**County Risk Assessment:**\n"]
                sorted_counties = sorted(
                    session.prediction.county_risks.items(),
                    key=lambda x: x[1], reverse=True
                )
                for county, risk in sorted_counties[:15]:
                    emoji = "🔴" if risk >= 0.7 else "🟠" if risk >= 0.5 else "🟡" if risk >= 0.3 else "🟢"
                    lines.append(f"{emoji} **{county}**: {risk:.0%}")
                answer = "\n".join(lines)
            else:
                answer = "County-level risk data not available for this bill."

        session.add_assistant_message(answer)
        return answer

    async def _handle_provision_question(self, session: ChatSession, question: str) -> str:
        """Handle provision drill-down questions."""
        # Try to find the referenced provision
        target_provision = None
        q_lower = question.lower()
        for p in session.bill.provisions:
            if p.clause_id.lower() in q_lower or p.description[:30].lower() in q_lower:
                target_provision = p
                break

        if target_provision:
            evidence_text = ""
            if session.evidence:
                evidence_text = session.evidence.summary_text(max_items=5)

            impact = await self._predictor.predict_provision(
                target_provision,
                bill_title=session.bill.title,
                evidence_text=evidence_text,
            )
            answer = self._format_provision_impact(target_provision, impact)
        else:
            # General provision question via LLM
            answer = await self._predictor.answer_question(
                question, session.bill, session.prediction, session.evidence
            )

        if not answer:
            answer = "I couldn't find a specific provision matching your question. " \
                     "Try mentioning the section number or a specific topic."

        session.add_assistant_message(answer)
        return answer

    async def _handle_comparison_question(self, session: ChatSession, question: str) -> str:
        """Handle historical comparison questions."""
        answer = await self._predictor.answer_question(
            question, session.bill, session.prediction, session.evidence
        )
        if not answer and session.prediction:
            answer = (
                f"**Historical Match:** {session.prediction.historical_match}\n"
                f"**Similarity:** {session.prediction.historical_similarity:.0%}\n"
                f"**Outcome:** {session.prediction.historical_outcome}"
            )
        session.add_assistant_message(answer or "No historical comparison data available.")
        return answer or "No historical comparison data available."

    async def _handle_search_question(self, session: ChatSession, question: str) -> str:
        """Search for additional evidence."""
        try:
            results = await self._searcher.search_query(question)
            if results.all_results:
                lines = [f"**Found {results.total_found} results:**\n"]
                for r in results.top_evidence[:10]:
                    emoji = "🐦" if r.source == "tweet" else "📰" if r.source == "news" else "⚠️"
                    lines.append(f"{emoji} [{r.source}] (sim={r.similarity:.2f}) {r.text[:200]}")
                    if r.source == "news" and r.metadata.get("url"):
                        lines.append(f"   🔗 {r.metadata['url']}")
                        excerpt = str(r.metadata.get("evidence_excerpt", "") or "").strip()
                        if excerpt:
                            lines.append(f"   📌 Evidence: {excerpt[:220]}")
                        pointer_parts = []
                        if r.metadata.get("article_id"):
                            pointer_parts.append(f"article_id={r.metadata['article_id']}")
                        if r.metadata.get("content_record_id"):
                            pointer_parts.append(f"content_id={r.metadata['content_record_id']}")
                        if r.metadata.get("content_storage_path"):
                            pointer_parts.append(f"path={r.metadata['content_storage_path']}")
                        if pointer_parts:
                            lines.append(f"   🧾 Trace: {' | '.join(pointer_parts)}")
                answer = "\n\n".join(lines)
            else:
                answer = "No matching results found. Try different keywords."
        except Exception as e:
            logger.error(f"Search failed: {e}")
            answer = "Search failed — please try again."

        session.add_assistant_message(answer)
        return answer

    async def _handle_general_question(self, session: ChatSession, question: str) -> str:
        """Handle general follow-up questions via LLM."""
        answer = await self._predictor.answer_question(
            question, session.bill, session.prediction, session.evidence
        )

        if not answer:
            answer = "I couldn't generate an answer. Try rephrasing your question."

        trace_block = self._format_news_trace_block(session.evidence.news if session.evidence else [])
        if trace_block:
            answer = f"{answer}\n\n{trace_block}"

        session.add_assistant_message(answer)
        return answer

    def _format_news_trace_block(self, news_results: List[Any], limit: int = 3) -> str:
        """Format mandatory URL + excerpt + pointer trace lines for news-backed answers."""
        if not news_results:
            return ""
        lines = ["### Evidence Trace"]
        seen = set()
        count = 0
        for item in sorted(news_results, key=lambda r: r.similarity, reverse=True):
            if count >= limit:
                break
            url = str(item.metadata.get("url", "") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            title = str(item.metadata.get("title", "News reference") or "News reference")
            excerpt = str(item.metadata.get("evidence_excerpt", "") or "").strip() or item.text[:200]
            trace_parts = []
            if item.metadata.get("article_id"):
                trace_parts.append(f"article_id={item.metadata['article_id']}")
            if item.metadata.get("content_record_id"):
                trace_parts.append(f"content_id={item.metadata['content_record_id']}")
            if item.metadata.get("content_storage_path"):
                trace_parts.append(f"path={item.metadata['content_storage_path']}")

            lines.append(f"- [{title}]({url})")
            lines.append(f"  excerpt: \"{excerpt}\"")
            if trace_parts:
                lines.append(f"  trace: {' | '.join(trace_parts)}")
            count += 1

        return "\n".join(lines) if count else ""

    # ─── Response Formatters ────────────────────────────────────────────

    def _format_bill_analysis(
        self,
        bill: BillAnalysis,
        prediction: Optional[ImpactPrediction],
        evidence: Optional[SearchResults],
    ) -> str:
        """Format the main bill analysis response as markdown."""
        lines = []

        # Header
        lines.append(f"### 📋 {bill.title}\n")

        if bill.summary:
            lines.append(f"{bill.summary}\n")

        # Quick stats
        lines.append(f"**Sectors:** {', '.join(bill.sectors)}")
        lines.append(f"**Provisions found:** {bill.provision_count}")
        lines.append(f"**Overall severity:** {bill.total_severity:.0%}")

        if bill.hashtags:
            lines.append(f"**Likely hashtags:** {' '.join(bill.hashtags[:6])}")

        # Prediction summary
        if prediction:
            lines.append(f"\n---\n### {prediction.risk_emoji} Impact Prediction\n")
            lines.append(f"**Mobilization probability:** {prediction.overall_mobilization:.0%}")
            lines.append(f"**Risk level:** {prediction.overall_risk_level.upper()}")

            if prediction.predicted_peak_timeline:
                lines.append(f"**Predicted peak:** {prediction.predicted_peak_timeline}")
            if prediction.predicted_phase:
                lines.append(f"**Current phase:** {prediction.predicted_phase}")

            if prediction.historical_match:
                lines.append(
                    f"\n**Historical match:** {prediction.historical_match} "
                    f"({prediction.historical_similarity:.0%} similarity)"
                )
                if prediction.historical_outcome:
                    lines.append(f"*{prediction.historical_outcome}*")

            if prediction.top_risk_counties:
                lines.append(f"\n**Top risk counties:** {', '.join(prediction.top_risk_counties[:8])}")

            if prediction.dominant_narratives:
                lines.append(f"**Dominant narratives:** {', '.join(prediction.dominant_narratives[:5])}")

            if prediction.likely_mobilizers:
                lines.append(f"**Likely mobilizers:** {', '.join(prediction.likely_mobilizers[:5])}")

        # Top provisions
        lines.append(f"\n---\n### Key Provisions\n")
        for i, p in enumerate(bill.top_provisions[:5], 1):
            sev_emoji = "🔴" if p.severity >= 0.8 else "🟠" if p.severity >= 0.6 else "🟡" if p.severity >= 0.4 else "🟢"
            lines.append(
                f"**{i}. {p.clause_id}** — {p.description}\n"
                f"   {sev_emoji} Severity: {p.severity:.0%} | "
                f"Sector: {p.sector} | "
                f"Affects: {', '.join(p.affected_groups[:4])}"
            )
            if p.monetary_impact:
                lines.append(f"   💰 {p.monetary_impact}")
            lines.append("")

        # Evidence summary
        if evidence and evidence.total_found > 0:
            lines.append(f"\n---\n*📊 Analysis backed by {evidence.total_found} data points "
                        f"({len(evidence.tweets)} tweets, {len(evidence.news)} news articles, "
                        f"{len(evidence.incidents)} incidents, {len(evidence.policy_events)} historical events)*")

        if evidence and evidence.news:
            lines.append("\n---\n### Linked News Evidence")
            seen_urls = set()
            ranked_news = sorted(evidence.news, key=lambda r: r.similarity, reverse=True)
            for idx, news_item in enumerate(ranked_news[:8], 1):
                url = str(news_item.metadata.get("url", "") or "").strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                title = str(news_item.metadata.get("title", "News reference") or "News reference")
                excerpt = str(news_item.metadata.get("evidence_excerpt", "") or "").strip()
                if not excerpt:
                    excerpt = news_item.text[:220]

                trace_parts = []
                if news_item.metadata.get("article_id"):
                    trace_parts.append(f"article_id={news_item.metadata['article_id']}")
                if news_item.metadata.get("content_record_id"):
                    trace_parts.append(f"content_id={news_item.metadata['content_record_id']}")
                if news_item.metadata.get("content_storage_path"):
                    trace_parts.append(f"path={news_item.metadata['content_storage_path']}")

                lines.append(f"{idx}. [{title}]({url})")
                lines.append(f"   - Evidence excerpt: \"{excerpt}\"")
                if trace_parts:
                    lines.append(f"   - Trace pointer: {' | '.join(trace_parts)}")

        # Prompt for follow-up
        lines.append("\n---\n*Ask me about specific provisions, counties, historical comparisons, "
                     "or counter-narratives.*")

        return "\n".join(lines)

    def _format_provision_impact(
        self,
        provision,
        impact: Any,
    ) -> str:
        """Format a single provision impact drill-down."""
        lines = [
            f"### {provision.clause_id}: {provision.description}\n",
            f"**Mobilization probability:** {impact.mobilization_probability:.0%}",
            f"**Risk level:** {impact.risk_level}",
            f"**Predicted timeline:** {impact.predicted_timeline}",
        ]
        if impact.likely_mobilizers:
            lines.append(f"**Likely mobilizers:** {', '.join(impact.likely_mobilizers)}")
        if impact.narrative_archetypes:
            lines.append(f"**Narratives:** {', '.join(impact.narrative_archetypes)}")
        if impact.counter_narratives:
            lines.append(f"\n**Counter-narrative suggestions:**")
            for cn in impact.counter_narratives:
                lines.append(f"- {cn}")
        return "\n".join(lines)
