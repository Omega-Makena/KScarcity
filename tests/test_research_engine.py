import pytest
from unittest.mock import MagicMock, patch
import asyncio

from kshiked.ui.institution.backend.research_engine import (
    ResearchEngine, EngineContext, ScopeViolationError
)
from kshiked.pulse.llm.signals import KShieldSignal, ThreatSignal, ThreatTier, ThreatCategory
from kshiked.ui.institution.backend.project_signals import ProjectSignalManager


@pytest.fixture
def mock_ollama_provider():
    with patch('kshiked.ui.institution.backend.research_engine.OllamaProvider') as mock:
        instance = mock.return_value
        # Mock async return for generate
        instance.generate = MagicMock(return_value=asyncio.Future())
        instance.generate.return_value.set_result(
            "Mock LLM Response indicating a potential threat."
        )
        yield instance

@pytest.fixture
def spoke_context():
    return EngineContext(role="spoke", user_id="spoke_user", sector_id=1, project_id=None)

@pytest.fixture
def admin_context():
    return EngineContext(role="admin", user_id="admin_user", sector_id=1, project_id=None)

@pytest.fixture
def exec_context():
    return EngineContext(role="executive", user_id="exec_user", sector_id=None, project_id=None)

@pytest.fixture
def project_context():
    return EngineContext(role="spoke", user_id="spoke_user", sector_id=1, project_id=101)


class TestEngineScoping:
    def test_spoke_cannot_query_national_scope(self, spoke_context):
        engine = ResearchEngine(spoke_context)
        with pytest.raises(ScopeViolationError):
            engine._check_scope("national")

    def test_spoke_can_query_sector_scope(self, spoke_context):
        engine = ResearchEngine(spoke_context)
        # Should not raise
        engine._check_scope("sector")

    def test_admin_cannot_query_national_scope(self, admin_context):
        engine = ResearchEngine(admin_context)
        with pytest.raises(ScopeViolationError):
            engine._check_scope("national")

    def test_executive_can_query_national_scope(self, exec_context):
        engine = ResearchEngine(exec_context)
        # Should not raise
        engine._check_scope("national")

    def test_querying_external_project_raises(self, spoke_context):
        engine = ResearchEngine(spoke_context)
        with pytest.raises(ScopeViolationError):
            engine._check_scope("project", target_project_id=202)


class TestEnginePrediction:
    @patch('kshiked.ui.institution.backend.research_engine.SFCEconomy')
    @pytest.mark.asyncio
    async def test_predict_wraps_sfc_economy(self, MockSFC, spoke_context):
        # Mock predicting logic
        engine = ResearchEngine(spoke_context)
        result = await engine.predict("test_indicator", horizon="30d", scope="sector")
        
        assert result.success is True
        assert result.metadata["metric"] == "test_indicator"
        assert result.metadata["horizon"] == "30d"
        assert "A localized upward trend is expected" in result.plain_language_summary


class TestEngineClassification:
    @pytest.mark.asyncio
    async def test_classify_uses_llm(self, spoke_context, mock_ollama_provider):
        engine = ResearchEngine(spoke_context)
        engine.llm = mock_ollama_provider # Inject mock
        
        result = await engine.classify("There is a massive drought.", entity="ClimateReport", scope="sector")
        
        assert result.success is True
        assert "Classified ClimateReport" in result.plain_language_summary
        mock_ollama_provider.generate.assert_called_once()


from kshiked.pulse.llm.signals import (
    KShieldSignal, ThreatSignal, ThreatTier, ThreatCategory, 
    ContextAnalysis, EconomicGrievance, SocialGrievance, AdvancedIndices
)
from datetime import datetime

class TestProjectSignals:
    @patch('kshiked.ui.institution.backend.project_signals.get_connection')
    def test_signal_relevance_matching(self, mock_get_conn):
        # Setup mock db
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_get_conn.return_value.__enter__.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Mock active projects
        mock_cursor.fetchall.return_value = [
            {"id": 1, "title": "Drought Relief", "description": "Mitigate drought impact"},
            {"id": 2, "title": "Cyber Security Update", "description": "Patching servers"}
        ]
        
        # Create a matching signal
        threat = ThreatSignal(
            category=ThreatCategory.CAT_7_ETHNIC_MOBILIZATION,
            tier=ThreatTier.TIER_2,
            intent=0.8, capability=0.8, specificity=0.8, reach=0.8, trajectory=0.8,
            classification_reason="High level of mobilization"
        )
        context = ContextAnalysis(
            economic_strain=EconomicGrievance.E1_DELEGITIMIZATION,
            social_fracture=SocialGrievance.S1_POLARIZATION,
            economic_dissatisfaction_score=0.5, social_dissatisfaction_score=0.5,
            shock_marker=0.5, polarization_marker=0.5
        )
        indices = AdvancedIndices(
            lei_score=0.5, institution_target="none", si_score=0.5,
            cognitive_rigidity=0.5, identity_fusion=0.5, maturation_score=50,
            maturation_stage="Narrative", aa_score=0.5, evasion_technique="none"
        )
        
        signal = KShieldSignal(
            source_id="news_1",
            timestamp=datetime.now(),
            content_text="Significant drought expected in the northern region causing resource tensions.",
            threat=threat,
            context=context,
            indices=indices
        )
        
        updates = ProjectSignalManager.evaluate_live_signal(signal)
        
        # Should match project 1 well above threshold
        assert len(updates) == 1
        assert updates[0]["project_id"] == 1
        assert updates[0]["impact_score"] > 0.4
