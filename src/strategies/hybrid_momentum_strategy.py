"""Hybrid strategy combining ML model predictions with momentum strategy signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy
from .trend_following import TrendFollowingStrategy


@dataclass
class HybridComponents:
    """Container for hybrid strategy components."""
    ml_strategy: BaseStrategy
    momentum_strategy: BaseStrategy


class HybridMomentumMLStrategy(BaseStrategy):
    """Strategy that fuses ML predictions with momentum signals.

    Parameters are supplied via ``initialize`` using a configuration dictionary
    containing:

    ``ml_model_type``: Name of the ML model (e.g. ``"xgboost"``)
    ``ml_model_params``: Parameters forwarded to :class:`TrendFollowingStrategy`
    ``momentum_strategy``: Name of momentum strategy registered in
        :class:`StrategyRegistry`
    ``agree_only``: If True, take trades only when both components agree
    ``weights``: Tuple of (ml_weight, momentum_weight) used when
        ``agree_only`` is False
    """

    def __init__(self,
                 ml_strategy: Optional[BaseStrategy] = None,
                 momentum_strategy: Optional[BaseStrategy] = None) -> None:
        super().__init__()
        self.components = HybridComponents(ml_strategy, momentum_strategy)
        self.agree_only: bool = True
        self.weights: Tuple[float, float] = (0.5, 0.5)
        self.is_trained: bool = False

    # ------------------------------------------------------------------
    def initialize(self, config: Dict) -> None:  # type: ignore[override]
        """Initialize hybrid strategy with configuration."""
        super().initialize(config)
        self.config = config
        self.agree_only = config.get("agree_only", True)
        self.weights = tuple(config.get("weights", (0.5, 0.5)))  # type: ignore

        # Create ML strategy if not supplied
        if self.components.ml_strategy is None:
            ml_model_type = config.get("ml_model_type", "xgboost")
            ml_params = config.get("ml_model_params", {})
            ml_cfg = {
                "name": f"{ml_model_type.title()} Hybrid", 
                "model_type": ml_model_type,
                "model_params": ml_params,
                "timeframe": config.get("timeframe", "5min"),
                "use_adaptive_thresholds": config.get("use_adaptive_thresholds", "auto"),
            }
            ml_strategy = TrendFollowingStrategy()
            ml_strategy.initialize(ml_cfg)
            self.components.ml_strategy = ml_strategy

        # Create momentum strategy if not supplied
        if self.components.momentum_strategy is None:
            from .strategy_registry import StrategyRegistry

            mom_name = config.get("momentum_strategy", "tema")
            mom_cfg = {
                "name": mom_name.upper(),
                "symbol": config.get("symbol", "SPY"),
                "primary_timeframe": config.get("timeframe", "5min"),
            }
            momentum_strategy = StrategyRegistry.get_strategy(mom_name, mom_cfg)
            self.components.momentum_strategy = momentum_strategy

    # ------------------------------------------------------------------
    def train(self, data: pd.DataFrame) -> None:  # type: ignore[override]
        """Train underlying ML strategy."""
        self.components.ml_strategy.train(data)
        self.is_trained = True

    # ------------------------------------------------------------------
    def predict(self, test_data: pd.DataFrame):  # type: ignore[override]
        """Generate hybrid signals for ``test_data``.

        Returns
        -------
        tuple
            (signals_dataframe, combined_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Hybrid strategy must be trained before prediction.")

        ml_strategy = self.components.ml_strategy
        momentum_strategy = self.components.momentum_strategy

        ml_signals, ml_pred = ml_strategy.predict(test_data)

        # Momentum strategy is rule-based – generate its signals directly
        mom_features, _, mom_dates = momentum_strategy.generate_features(test_data)
        mom_signals = momentum_strategy.generate_signals(mom_features, None, mom_dates)

        combined_df = pd.DataFrame(index=ml_signals.index.union(mom_signals.index))
        combined_df["ml_signal"] = ml_signals["signal"].reindex(combined_df.index).fillna(0)
        combined_df["ml_prob"] = pd.Series(ml_pred, index=ml_signals.index).reindex(combined_df.index).fillna(0.5)
        combined_df["mom_signal"] = mom_signals["signal"].reindex(combined_df.index).fillna(0)

        if self.agree_only:
            combined_df["combined_signal"] = np.where(
                (combined_df["ml_signal"] == 1) & (combined_df["mom_signal"] == 1), 1,
                np.where(
                    (combined_df["ml_signal"] == -1) & (combined_df["mom_signal"] == -1),
                    -1,
                    0,
                ),
            )
            combined_df["combined_prob"] = np.where(
                combined_df["combined_signal"] == 1,
                1.0,
                np.where(combined_df["combined_signal"] == -1, 0.0, 0.5),
            )
        else:
            mom_prob = (combined_df["mom_signal"] + 1) / 2
            combined_prob = (
                self.weights[0] * combined_df["ml_prob"] +
                self.weights[1] * mom_prob
            ).clip(0, 1)
            probs = combined_prob.values
            combined_df["combined_prob"] = combined_prob
            combined_df["combined_signal"] = [
                self._prob_to_signal(p, probs) for p in probs
            ]

        signals = combined_df[["combined_signal"]].rename(columns={"combined_signal": "signal"})
        return signals, combined_df["combined_prob"].values

    # ------------------------------------------------------------------
    def generate_features(self, data):  # type: ignore[override]
        """Delegate feature generation to the ML strategy."""
        return self.components.ml_strategy.generate_features(data)

    # ------------------------------------------------------------------
    def generate_signals(self, features, predictions, dates):  # type: ignore[override]
        """Not used directly. Use :meth:`predict` instead."""
        raise NotImplementedError("HybridMomentumMLStrategy uses predict() for signal generation")

    # ------------------------------------------------------------------
    def backtest(self, data, train_data=None, test_data=None, timeframe='daily'):  # type: ignore[override]
        """Simple backtest wrapper for compatibility."""
        if train_data is None or test_data is None:
            train_size = int(len(data) * 0.7)
            train_data = data.iloc[:train_size]
            test_data = data.iloc[train_size:]
        self.train(train_data)
        signals, _ = self.predict(test_data)
        return {'signals': signals}
