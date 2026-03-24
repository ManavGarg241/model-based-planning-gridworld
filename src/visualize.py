from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .environment import GridWorld, State


def print_grid_with_path(env: GridWorld, path: List[State]) -> None:
    grid = [["." for _ in range(env.size)] for _ in range(env.size)]
    for r, c in env.obstacles:
        grid[r][c] = "#"

    for r, c in path:
        if (r, c) not in env.obstacles and (r, c) != env.goal:
            grid[r][c] = "*"

    sr, sc = path[0]
    grid[sr][sc] = "S"

    gr, gc = env.goal
    grid[gr][gc] = "G"

    print("\nGrid path visualization:")
    for row in grid:
        print(" ".join(row))


def plot_path(
    env: GridWorld,
    path: List[State],
    title: str = "Agent Path",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    board = np.zeros((env.size, env.size), dtype=np.float32)
    for r, c in env.obstacles:
        board[r, c] = -1.0

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(board, cmap="Greys", vmin=-1.0, vmax=0.5)

    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, marker="o", color="tab:blue", linewidth=2)

    ax.scatter([env.goal[1]], [env.goal[0]], c="tab:green", s=120, marker="*", label="Goal")
    ax.scatter([path[0][1]], [path[0][0]], c="tab:red", s=90, marker="s", label="Start")

    ax.set_title(title)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.grid(True, linewidth=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def _draw_path_on_axis(
    ax: plt.Axes,
    env: GridWorld,
    path: List[State],
    line_color: str,
    title: str,
) -> None:
    board = np.zeros((env.size, env.size), dtype=np.float32)
    for r, c in env.obstacles:
        board[r, c] = -1.0

    ax.imshow(board, cmap="Greys", vmin=-1.0, vmax=0.5)

    ys = [p[0] for p in path]
    xs = [p[1] for p in path]
    ax.plot(xs, ys, marker="o", color=line_color, linewidth=2)

    ax.scatter([env.goal[1]], [env.goal[0]], c="tab:green", s=110, marker="*", label="Goal")
    ax.scatter([path[0][1]], [path[0][0]], c="tab:red", s=85, marker="s", label="Start")

    ax.set_title(title)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.grid(True, linewidth=0.5)


def plot_policy_comparison(
    env: GridWorld,
    random_path: List[State],
    planning_path: List[State],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    _draw_path_on_axis(axes[0], env, random_path, "tab:orange", "Random Policy Path")
    _draw_path_on_axis(axes[1], env, planning_path, "tab:blue", "Planning Policy Path")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Random vs Planning Path Comparison", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_metrics_comparison(
    labels: List[str],
    random_success: List[float],
    planning_success: List[float],
    random_steps: List[float],
    planning_steps: List[float],
    save_path: str | None = None,
    show: bool = True,
) -> None:
    x = np.arange(len(labels))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].bar(x - width / 2, random_success, width, label="Random", color="tab:orange")
    axes[0].bar(x + width / 2, planning_success, width, label="Planning", color="tab:blue")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_title("Success Rate Across Environments")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    axes[1].bar(x - width / 2, random_steps, width, label="Random", color="tab:orange")
    axes[1].bar(x + width / 2, planning_steps, width, label="Planning", color="tab:blue")
    axes[1].set_ylabel("Avg Steps (All Episodes)")
    axes[1].set_title("Average Steps Across Environments")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_predicted_vs_actual_paths(
    env: GridWorld,
    predicted_path: List[State],
    actual_path: List[State],
    title: str = "Predicted vs Actual Path",
    save_path: str | None = None,
    show: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    board = np.zeros((env.size, env.size), dtype=np.float32)
    for r, c in env.obstacles:
        board[r, c] = -1.0
    ax.imshow(board, cmap="Greys", vmin=-1.0, vmax=0.5)

    pred_y = [p[0] for p in predicted_path]
    pred_x = [p[1] for p in predicted_path]
    act_y = [p[0] for p in actual_path]
    act_x = [p[1] for p in actual_path]

    ax.plot(pred_x, pred_y, marker="o", color="tab:purple", linewidth=2, label="Predicted")
    ax.plot(act_x, act_y, marker="x", color="tab:blue", linewidth=2, label="Actual")
    ax.scatter([env.goal[1]], [env.goal[0]], c="tab:green", s=120, marker="*", label="Goal")
    ax.scatter([actual_path[0][1]], [actual_path[0][0]], c="tab:red", s=90, marker="s", label="Start")

    ax.set_title(title)
    ax.set_xticks(range(env.size))
    ax.set_yticks(range(env.size))
    ax.set_xlim(-0.5, env.size - 0.5)
    ax.set_ylim(env.size - 0.5, -0.5)
    ax.grid(True, linewidth=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=180, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)
