from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


State = Tuple[int, int]


@dataclass
class GridWorld:
    """Deterministic grid world with static obstacles."""

    size: int
    start: State
    goal: State
    obstacles: List[State]

    def __post_init__(self) -> None:
        self.action_to_delta: Dict[int, State] = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1),   # right
        }
        self.valid_actions = list(self.action_to_delta.keys())
        self._validate()

    def _validate(self) -> None:
        all_cells = {(r, c) for r in range(self.size) for c in range(self.size)}
        if self.start not in all_cells:
            raise ValueError(f"Start {self.start} is out of bounds")
        if self.goal not in all_cells:
            raise ValueError(f"Goal {self.goal} is out of bounds")
        if self.start in self.obstacles:
            raise ValueError("Start cannot be an obstacle")
        if self.goal in self.obstacles:
            raise ValueError("Goal cannot be an obstacle")
        unknown_obstacles = set(self.obstacles) - all_cells
        if unknown_obstacles:
            raise ValueError(f"Obstacle out of bounds: {unknown_obstacles}")

    def is_terminal(self, state: State) -> bool:
        return state == self.goal

    def in_bounds(self, state: State) -> bool:
        r, c = state
        return 0 <= r < self.size and 0 <= c < self.size

    def is_blocked(self, state: State) -> bool:
        return state in set(self.obstacles)

    def step(self, state: State, action: int) -> State:
        """
        Deterministic transition:
        if movement would leave the grid or hit obstacle, stay in place.
        """
        if action not in self.action_to_delta:
            raise ValueError(f"Unknown action: {action}")

        dr, dc = self.action_to_delta[action]
        candidate = (state[0] + dr, state[1] + dc)

        if not self.in_bounds(candidate) or self.is_blocked(candidate):
            return state
        return candidate

    def all_free_states(self) -> List[State]:
        obstacle_set = set(self.obstacles)
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if (r, c) not in obstacle_set
        ]


def default_environment() -> GridWorld:
    return GridWorld(
        size=6,
        start=(0, 0),
        goal=(5, 5),
        obstacles=[(1, 1), (1, 2), (2, 2), (3, 3), (4, 1)],
    )


def generalization_environment() -> GridWorld:
    # Changed obstacle layout and start position for generalization checks.
    return GridWorld(
        size=6,
        start=(0, 2),
        goal=(5, 5),
        obstacles=[(1, 1), (2, 1), (2, 2), (3, 4), (4, 2)],
    )


def environment_layout_b() -> GridWorld:
    return GridWorld(
        size=6,
        start=(5, 0),
        goal=(0, 5),
        obstacles=[(1, 2), (2, 2), (3, 1), (3, 2), (4, 4)],
    )


def environment_layout_c() -> GridWorld:
    return GridWorld(
        size=6,
        start=(2, 0),
        goal=(5, 4),
        obstacles=[(0, 3), (1, 3), (2, 3), (4, 1), (4, 2), (4, 3)],
    )


def evaluation_environments() -> list[tuple[str, GridWorld]]:
    return [
        ("Layout-A-default", default_environment()),
        ("Layout-B-diagonal-shift", environment_layout_b()),
        ("Layout-C-corridor", environment_layout_c()),
    ]
