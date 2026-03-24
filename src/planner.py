from __future__ import annotations

from itertools import product
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .environment import GridWorld, State
from .model import WorldModel


_MODEL_STEP_CACHE: Dict[Tuple[int, int, Tuple[State, ...]], Dict[Tuple[State, int], State]] = {}


def _cache_key(model: WorldModel, env: GridWorld) -> Tuple[int, int, Tuple[State, ...]]:
    return (id(model), env.size, tuple(sorted(env.obstacles)))


def _predict_one_step_cached(
    model: WorldModel,
    env: GridWorld,
    state: State,
    action: int,
) -> State:
    key = _cache_key(model, env)
    env_cache = _MODEL_STEP_CACHE.setdefault(key, {})
    pair = (state, action)

    if pair not in env_cache:
        curr = np.array(state, dtype=np.float32)
        pred = model.predict_next_state_continuous(curr, action)
        pred_disc = WorldModel.discrete_from_continuous(pred.reshape(1, -1), env.size)[0]
        env_cache[pair] = (int(pred_disc[0]), int(pred_disc[1]))

    return env_cache[pair]


def _simulate_sequence_with_model(
    model: WorldModel,
    env: GridWorld,
    state: State,
    actions: Sequence[int],
) -> Tuple[State, List[State]]:
    curr = state
    path = [state]

    for action in actions:
        next_state = _predict_one_step_cached(model, env, curr, action)
        path.append(next_state)
        curr = next_state

    return path[-1], path


def choose_action_by_planning(
    model: WorldModel,
    env: GridWorld,
    state: State,
    depth: int = 4,
) -> int:
    best_reaching_sequence = None
    best_reaching_steps = float("inf")

    best_fallback_sequence = None
    best_fallback_score = float("inf")

    for seq in product(env.valid_actions, repeat=depth):
        end_state, path = _simulate_sequence_with_model(model, env, state, seq)

        if env.goal in path[1:]:
            first_goal_idx = path.index(env.goal)
            if first_goal_idx < best_reaching_steps:
                best_reaching_steps = first_goal_idx
                best_reaching_sequence = seq
            continue

        # Fallback: prefer trajectories that get close to goal while making progress.
        min_dist_along_path = min(
            abs(p[0] - env.goal[0]) + abs(p[1] - env.goal[1])
            for p in path[1:]
        )
        first_next = path[1]
        no_progress_penalty = 1.0 if first_next == state else 0.0
        invalid_penalty = 0.0
        if not env.in_bounds(end_state):
            invalid_penalty += 5.0
        if end_state in env.obstacles:
            invalid_penalty += 3.0

        score = min_dist_along_path + no_progress_penalty + invalid_penalty
        if score < best_fallback_score:
            best_fallback_score = score
            best_fallback_sequence = seq

    if best_reaching_sequence is not None:
        return int(best_reaching_sequence[0])
    if best_fallback_sequence is None:
        return int(np.random.randint(0, 4))
    return int(best_fallback_sequence[0])
