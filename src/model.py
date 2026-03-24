from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor

from .data import TransitionDataset, to_model_features


@dataclass
class EvalMetrics:
    mse: float
    exact_match_accuracy: float
    avg_manhattan_error: float


class WorldModel:
    def __init__(self, random_state: int = 42) -> None:
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=1e-3,
            alpha=1e-4,
            max_iter=300,
            random_state=random_state,
        )

    def fit(self, dataset: TransitionDataset) -> None:
        x = to_model_features(dataset.states, dataset.actions)
        y = dataset.next_states
        self.model.fit(x, y)

    def predict_next_state_continuous(self, state: np.ndarray, action: int) -> np.ndarray:
        x = np.zeros((1, 6), dtype=np.float32)
        x[0, 0:2] = state
        x[0, 2 + action] = 1.0
        return self.model.predict(x)[0]

    def predict_batch(self, dataset: TransitionDataset) -> np.ndarray:
        x = to_model_features(dataset.states, dataset.actions)
        return self.model.predict(x)

    @staticmethod
    def discrete_from_continuous(pred: np.ndarray, grid_size: int) -> np.ndarray:
        clipped = np.clip(np.rint(pred), 0, grid_size - 1)
        return clipped.astype(np.int64)

    def evaluate(self, dataset: TransitionDataset, grid_size: int) -> EvalMetrics:
        pred = self.predict_batch(dataset)
        mse = float(mean_squared_error(dataset.next_states, pred))

        pred_disc = self.discrete_from_continuous(pred, grid_size)
        target_disc = dataset.next_states.astype(np.int64)
        exact = np.all(pred_disc == target_disc, axis=1)
        exact_match_accuracy = float(np.mean(exact))

        manhattan = np.abs(pred_disc - target_disc).sum(axis=1)
        avg_manhattan_error = float(np.mean(manhattan))

        return EvalMetrics(
            mse=mse,
            exact_match_accuracy=exact_match_accuracy,
            avg_manhattan_error=avg_manhattan_error,
        )

    def summary(self, metrics: EvalMetrics) -> Dict[str, float]:
        return {
            "mse": metrics.mse,
            "exact_match_accuracy": metrics.exact_match_accuracy,
            "avg_manhattan_error": metrics.avg_manhattan_error,
        }
