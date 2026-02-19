"""
Feature Layer.

Responsible for data hygiene and validation before it enters the identification/estimation
pipeline. Ensures strict schema compliance with the Spec.
"""
import logging
from typing import List

import pandas as pd

from scarcity.causal.specs import EstimandSpec

logger = logging.getLogger(__name__)


class FeatureBuilder:
    """
    Validates and prepares data for the Causal Pipeline.
    """

    @staticmethod
    def validate_and_clean(data: pd.DataFrame, spec: EstimandSpec) -> pd.DataFrame:
        """
        Ensures data has all required columns and is numeric/clean where needed.
        Drops rows with NaNs in critical columns (Treatment, Outcome, Confounders).
        """
        required_cols = {spec.treatment, spec.outcome}
        required_cols.update(spec.confounders)
        required_cols.update(spec.effect_modifiers)
        if spec.instrument:
            required_cols.add(spec.instrument)
        if spec.mediator:
            required_cols.add(spec.mediator)
        if spec.time_column and spec.time_column in data.columns:
            required_cols.add(spec.time_column)
        for key in spec.panel_keys:
            if key in data.columns:
                required_cols.add(key)

        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns from Spec: {missing}")

        df = data[list(required_cols)].copy()

        initial_len = len(df)
        df = df.dropna()
        dropped = initial_len - len(df)

        if dropped > 0:
            logger.warning(f"FeatureLayer dropped {dropped} rows due to missing values in critical columns.")

        if len(df) == 0:
            raise ValueError("Dataset is empty after dropping NaNs in required columns.")

        return df
