from __future__ import annotations

from scarcity.simulation.types import Sector

from kshiked.simulation.sectors.config import FeedbackChannel
from kshiked.simulation.sectors.feedback_map import FeedbackMapper


def test_multiplicative_reduction_with_population_normalization() -> None:
    mapper = FeedbackMapper(
        channels=[
            FeedbackChannel(
                name="labor",
                source_compartments=["infected"],
                normalization="population",
                target_field="labor_supply_factor",
                transform="multiplicative_reduction",
                sensitivity=1.0,
                min_value=0.0,
                max_value=1.0,
            )
        ],
        sector_name="test",
    )

    feedback = mapper.compute_feedback({"infected": 10.0, "total": 100.0}, population=100.0)
    assert abs(feedback.labor_supply_factor - 0.9) < 1e-9


def test_additive_scalar_channel() -> None:
    mapper = FeedbackMapper(
        channels=[
            FeedbackChannel(
                name="fiscal",
                source_compartments=["hospitalized"],
                target_field="additional_gov_spending",
                transform="additive",
                sensitivity=0.01,
            )
        ],
        sector_name="test",
    )

    feedback = mapper.compute_feedback({"hospitalized": 50.0})
    assert abs(feedback.additional_gov_spending - 0.5) < 1e-12


def test_dict_target_mapping_by_sector_key() -> None:
    mapper = FeedbackMapper(
        channels=[
            FeedbackChannel(
                name="demand",
                source_compartments=["drag"],
                target_field="demand_shift",
                target_sector="agri",
                transform="multiplicative_reduction",
                sensitivity=0.2,
                min_value=0.0,
                max_value=1.0,
            )
        ],
        sector_name="test",
    )

    feedback = mapper.compute_feedback({"drag": 0.5})
    assert feedback.demand_shift is not None
    assert abs(feedback.demand_shift[Sector.AGRICULTURE] - 0.9) < 1e-12


def test_alias_fx_pressure_to_fx_outflow_pressure() -> None:
    mapper = FeedbackMapper(
        channels=[
            FeedbackChannel(
                name="fx",
                source_compartments=["risk"],
                target_field="fx_pressure",
                transform="additive",
                sensitivity=2.0,
            )
        ],
        sector_name="test",
    )

    feedback = mapper.compute_feedback({"risk": 0.3})
    assert abs(feedback.fx_outflow_pressure - 0.6) < 1e-12
