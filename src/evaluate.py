from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .environment import GridWorld, State
from .planner import choose_action_by_planning
from .model import WorldModel


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    path: List[State]
    actions: List[int]


def run_episode_random(
    env: GridWorld,
    max_steps: int = 40,
    seed: int = 0,
    start_override: State | None = None,
) -> EpisodeResult:
    rng = np.random.default_rng(seed)
    state = env.start if start_override is None else start_override
    path = [state]
    actions: List[int] = []

    for step in range(1, max_steps + 1):
        action = int(rng.integers(0, 4))
        actions.append(action)
        state = env.step(state, action)
        path.append(state)
        if env.is_terminal(state):
            return EpisodeResult(True, step, path, actions)

    return EpisodeResult(False, max_steps, path, actions)


def run_episode_planning(
    env: GridWorld,
    model: WorldModel,
    max_steps: int = 40,
    depth: int = 4,
    start_override: State | None = None,
) -> EpisodeResult:
    state = env.start if start_override is None else start_override
    path = [state]
    actions: List[int] = []
    decision_cache: Dict[State, int] = {}

    for step in range(1, max_steps + 1):
        action = decision_cache.get(state)
        if action is None:
            action = choose_action_by_planning(model, env, state, depth=depth)
            decision_cache[state] = action
        actions.append(action)
        state = env.step(state, action)
        path.append(state)
        if env.is_terminal(state):
            return EpisodeResult(True, step, path, actions)

    return EpisodeResult(False, max_steps, path, actions)


def _aggregate(results: List[EpisodeResult]) -> Dict[str, float]:
    success_rate = float(np.mean([r.success for r in results]))
    successful_steps = [r.steps for r in results if r.success]
    avg_steps_success = float(np.mean(successful_steps)) if successful_steps else float("inf")
    avg_steps_all = float(np.mean([r.steps for r in results]))

    return {
        "success_rate": success_rate,
        "avg_steps_successful_episodes": avg_steps_success,
        "avg_steps_all_episodes": avg_steps_all,
    }


def benchmark_policies(
    env: GridWorld,
    model: WorldModel,
    episodes: int = 100,
    max_steps: int = 40,
    depth: int = 4,
    seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, float], List[EpisodeResult], List[EpisodeResult]]:
    random_results: List[EpisodeResult] = []
    planning_results: List[EpisodeResult] = []

    rng = np.random.default_rng(seed)
    free_states = env.all_free_states()

    for _ in range(episodes):
        start_state = free_states[rng.integers(0, len(free_states))]
        random_results.append(
            run_episode_random(
                env,
                max_steps=max_steps,
                seed=int(rng.integers(0, 1_000_000)),
                start_override=start_state,
            )
        )
        planning_results.append(
            run_episode_planning(
                env,
                model,
                max_steps=max_steps,
                depth=depth,
                start_override=start_state,
            )
        )

    return (
        _aggregate(random_results),
        _aggregate(planning_results),
        random_results,
        planning_results,
    )


def generalization_test(
    env: GridWorld,
    model: WorldModel,
    episodes: int = 60,
    max_steps: int = 40,
    depth: int = 4,
) -> Dict[str, float]:
    planning_results = [
        run_episode_planning(env, model, max_steps=max_steps, depth=depth)
        for _ in range(episodes)
    ]
    return _aggregate(planning_results)
