"""
Identification Layer.

Wraps DoWhy's identification logic. Translates our `EstimandSpec`
into a DoWhy `CausalModel` and performs identification (Backdoor, IV, Mediation).
"""
import logging
from typing import Optional

import pandas as pd
from dowhy import CausalModel
from dowhy.causal_identifier import IdentifiedEstimand

from scarcity.causal.specs import EstimandSpec, EstimandType

logger = logging.getLogger(__name__)


class Identifier:
    """
    Adapts the Spec into a DoWhy CausalModel and performs Identification.
    """

    def __init__(self, spec: EstimandSpec, graph: Optional[str] = None):
        self.spec = spec
        self.spec.validate()
        self.graph = graph

    def identify(self, data: pd.DataFrame) -> tuple[CausalModel, IdentifiedEstimand]:
        """
        Constructs the causal graph and identifies the estimand.

        Args:
            data: The pandas DataFrame containing the data.

        Returns:
            A tuple of (CausalModel, IdentifiedEstimand).
        """
        common_causes = self.spec.confounders
        instruments = [self.spec.instrument] if self.spec.instrument else None
        effect_modifiers = self.spec.effect_modifiers

        model = CausalModel(
            data=data,
            treatment=self.spec.treatment,
            outcome=self.spec.outcome,
            common_causes=common_causes,
            instruments=instruments,
            effect_modifiers=effect_modifiers,
            graph=self.graph,
            proceed_when_unidentifiable=True,
        )

        identify_method = "default"
        if self.spec.type == EstimandType.LATE:
            identify_method = "default"
        elif self.spec.type in (EstimandType.MEDIATION_NDE, EstimandType.MEDIATION_NIE):
            identify_method = "default"

        logger.info(f"Identifying effect for {self.spec.type} using method='{identify_method}'")
        identified_estimand = model.identify_effect(
            proceed_when_unidentifiable=True,
            method_name=identify_method,
        )

        if not identified_estimand:
            raise ValueError(f"Effect could not be identified for spec: {self.spec}")

        return model, identified_estimand
