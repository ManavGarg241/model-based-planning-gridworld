from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .environment import GridWorld, State


@dataclass
class TransitionDataset:
    states: np.ndarray      # shape: (N, 2)
    actions: np.ndarray     # shape: (N,)
    next_states: np.ndarray # shape: (N, 2)


def generate_random_transitions(
    env: GridWorld,
    num_transitions: int,
    seed: int = 42,
) -> TransitionDataset:
    rng = np.random.default_rng(seed)
    free_states = env.all_free_states()

    states = np.zeros((num_transitions, 2), dtype=np.float32)
    actions = np.zeros((num_transitions,), dtype=np.int64)
    next_states = np.zeros((num_transitions, 2), dtype=np.float32)

    state: State = free_states[rng.integers(0, len(free_states))]

    for i in range(num_transitions):
        action = int(rng.integers(0, 4))
        nxt = env.step(state, action)

        states[i] = np.array(state, dtype=np.float32)
        actions[i] = action
        next_states[i] = np.array(nxt, dtype=np.float32)

        # Mix local trajectories and random resets for better coverage.
        if rng.random() < 0.15:
            state = free_states[rng.integers(0, len(free_states))]
        else:
            state = nxt

    return TransitionDataset(states=states, actions=actions, next_states=next_states)


def to_model_features(states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    action_one_hot = np.zeros((actions.shape[0], 4), dtype=np.float32)
    action_one_hot[np.arange(actions.shape[0]), actions] = 1.0
    return np.concatenate([states.astype(np.float32), action_one_hot], axis=1)


def train_test_split(
    dataset: TransitionDataset,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[TransitionDataset, TransitionDataset]:
    n = dataset.states.shape[0]
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    split = int(n * (1.0 - test_ratio))
    train_idx = idx[:split]
    test_idx = idx[split:]

    train = TransitionDataset(
        states=dataset.states[train_idx],
        actions=dataset.actions[train_idx],
        next_states=dataset.next_states[train_idx],
    )
    test = TransitionDataset(
        states=dataset.states[test_idx],
        actions=dataset.actions[test_idx],
        next_states=dataset.next_states[test_idx],
    )
    return train, test
