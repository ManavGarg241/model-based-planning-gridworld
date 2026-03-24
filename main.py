from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np

from src.data import generate_random_transitions, train_test_split
from src.environment import (
    default_environment,
    evaluation_environments,
    generalization_environment,
)
from src.evaluate import (
    benchmark_policies,
    run_episode_planning,
    run_episode_random,
)
from src.model import WorldModel
from src.visualize import (
    plot_metrics_comparison,
    plot_path,
    plot_policy_comparison,
    plot_predicted_vs_actual_paths,
    print_grid_with_path,
)


def format_metrics(name: str, metrics: dict) -> str:
    lines = [f"{name}:"]
    for k, v in metrics.items():
        lines.append(f"  - {k}: {v:.4f}" if v != float("inf") else f"  - {k}: inf")
    return "\n".join(lines)


def build_analysis(
    model_metrics: dict,
    random_metrics: dict,
    planning_metrics: dict,
    generalization_metrics: dict,
) -> str:
    lines = []
    lines.append("Analysis")
    lines.append("--------")

    gain = planning_metrics["success_rate"] - random_metrics["success_rate"]
    lines.append(
        f"Planning success gain over random: {gain:.3f} (absolute)."
    )

    lines.append(
        "Higher transition prediction quality typically improves planning because "
        "the search tree becomes a better proxy for real environment rollouts."
    )

    if model_metrics["exact_match_accuracy"] < 0.85:
        lines.append(
            "Failure risk: world model is not accurate enough, so planned sequences "
            "can drift from true dynamics."
        )
    else:
        lines.append(
            "Model appears accurate enough for short-horizon planning in this grid."
        )

    if generalization_metrics["success_rate"] < planning_metrics["success_rate"]:
        lines.append(
            "Generalization drop detected: changed map/start hurts planning because "
            "training transitions did not fully cover shifted dynamics near new obstacles."
        )
    else:
        lines.append(
            "Generalization remained stable under changed layout/start in this setup."
        )

    lines.append(
        "Planning can break when compounding model errors accumulate over depth, "
        "especially near walls/obstacles where one wrong prediction changes reachable states."
    )

    return "\n".join(lines)


class BiasedWorldModel:
    """Wraps the trained model and injects small systematic prediction bias."""

    def __init__(self, base_model: WorldModel, bias_shift: tuple[float, float] = (0.0, 0.52)) -> None:
        self.base_model = base_model
        self.bias_shift = np.array(bias_shift, dtype=np.float32)

    def predict_next_state_continuous(self, state: np.ndarray, action: int) -> np.ndarray:
        pred = self.base_model.predict_next_state_continuous(state, action)
        # Add small bias only near central corridor so failures are interpretable.
        if 1 <= int(state[0]) <= 4 and 1 <= int(state[1]) <= 4:
            return pred + self.bias_shift
        return pred


def rollout_predicted_and_actual(
    env,
    model,
    start_state,
    actions: List[int],
) -> tuple[List[tuple[int, int]], List[tuple[int, int]]]:
    predicted = [start_state]
    actual = [start_state]

    curr_pred = np.array(start_state, dtype=np.float32)
    curr_actual = start_state
    for action in actions:
        pred_cont = model.predict_next_state_continuous(curr_pred, action)
        pred_disc = WorldModel.discrete_from_continuous(pred_cont.reshape(1, -1), env.size)[0]
        pred_state = (int(pred_disc[0]), int(pred_disc[1]))

        curr_actual = env.step(curr_actual, action)

        predicted.append(pred_state)
        actual.append(curr_actual)
        curr_pred = pred_disc.astype(np.float32)

    return predicted, actual


def main() -> None:
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = default_environment()

    # 1) Dataset generation (within requested 5k-20k transitions).
    dataset = generate_random_transitions(env, num_transitions=10000, seed=42)
    train_data, test_data = train_test_split(dataset, test_ratio=0.2, seed=42)

    # 2) Train world model.
    world_model = WorldModel(random_state=42)
    world_model.fit(train_data)

    # 3) Evaluate model quality.
    eval_metrics = world_model.evaluate(test_data, grid_size=env.size)
    model_metrics = world_model.summary(eval_metrics)

    # 4) Compare random vs planning on default environment.
    random_metrics, planning_metrics, random_episodes, planning_episodes = benchmark_policies(
        env=env,
        model=world_model,
        episodes=60,
        max_steps=40,
        depth=3,
        seed=42,
    )

    # 5) Single generalization environment with changed obstacles/start.
    gen_env = generalization_environment()
    _, gen_metrics, _, _ = benchmark_policies(
        env=gen_env,
        model=world_model,
        episodes=40,
        max_steps=40,
        depth=3,
        seed=17,
    )

    # 6) Multi-environment comparison (2-3 layouts).
    env_results = []
    for idx, (label, eval_env) in enumerate(evaluation_environments()):
        r_metrics, p_metrics, _, _ = benchmark_policies(
            env=eval_env,
            model=world_model,
            episodes=40,
            max_steps=40,
            depth=3,
            seed=100 + idx,
        )
        env_results.append(
            {
                "environment": label,
                "start": list(eval_env.start),
                "goal": list(eval_env.goal),
                "random": r_metrics,
                "planning": p_metrics,
            }
        )

    # 7) Print outputs.
    print("\n=== WORLD MODEL EVALUATION ===")
    print(format_metrics("Model", model_metrics))

    print("\n=== POLICY COMPARISON ===")
    print(format_metrics("Random policy", random_metrics))
    print(format_metrics("Planning policy", planning_metrics))

    print("\n=== GENERALIZATION TEST ===")
    print(format_metrics("Planning on changed environment", gen_metrics))

    print("\n=== MULTI-ENVIRONMENT SUMMARY ===")
    for row in env_results:
        print(f"{row['environment']} start={tuple(row['start'])} goal={tuple(row['goal'])}")
        print(format_metrics("  Random", row["random"]))
        print(format_metrics("  Planning", row["planning"]))

    analysis = build_analysis(model_metrics, random_metrics, planning_metrics, gen_metrics)
    print("\n" + analysis)

    # 8) Visualization: default successful planning path.
    best_planning_episode = min(planning_episodes, key=lambda e: e.steps)
    print_grid_with_path(env, best_planning_episode.path)

    # Save plot image on every run; show window is optional.
    plot_image_path = output_dir / "planning_path.png"
    show_plot = False
    plot_path(
        env,
        best_planning_episode.path,
        title="Planning Path (Learned Model)",
        save_path=str(plot_image_path),
        show=show_plot,
    )
    print(f"Saved visualization to: {plot_image_path.resolve()}")

    # 9) Visualization: random vs planning path on the same start state.
    random_demo = run_episode_random(env, max_steps=35, seed=11, start_override=env.start)
    planning_demo = run_episode_planning(env, world_model, max_steps=35, depth=3, start_override=env.start)
    policy_compare_path = output_dir / "random_vs_planning.png"
    plot_policy_comparison(
        env,
        random_demo.path,
        planning_demo.path,
        save_path=str(policy_compare_path),
        show=False,
    )
    print(f"Saved baseline comparison to: {policy_compare_path.resolve()}")

    # 10) Failure case: small model bias causing planning degradation.
    biased_model = BiasedWorldModel(world_model)
    biased_planning_episode = run_episode_planning(env, biased_model, max_steps=35, depth=3, start_override=env.start)
    predicted_path, actual_path = rollout_predicted_and_actual(
        env,
        biased_model,
        env.start,
        biased_planning_episode.actions,
    )
    failure_path = output_dir / "failure_predicted_vs_actual.png"
    plot_predicted_vs_actual_paths(
        env,
        predicted_path,
        actual_path,
        title="Failure Case: Biased Prediction vs Actual Trajectory",
        save_path=str(failure_path),
        show=False,
    )
    print(f"Saved failure-case visualization to: {failure_path.resolve()}")

    mismatches = sum(1 for p, a in zip(predicted_path[1:], actual_path[1:]) if p != a)
    mismatch_rate = mismatches / max(1, len(actual_path) - 1)

    # 11) Metrics chart across environments.
    metric_plot_path = output_dir / "multi_env_metrics.png"
    labels = [row["environment"] for row in env_results]
    random_success = [row["random"]["success_rate"] for row in env_results]
    planning_success = [row["planning"]["success_rate"] for row in env_results]
    random_steps = [row["random"]["avg_steps_all_episodes"] for row in env_results]
    planning_steps = [row["planning"]["avg_steps_all_episodes"] for row in env_results]
    plot_metrics_comparison(
        labels,
        random_success,
        planning_success,
        random_steps,
        planning_steps,
        save_path=str(metric_plot_path),
        show=False,
    )
    print(f"Saved multi-environment metrics plot to: {metric_plot_path.resolve()}")

    # 12) Save metrics for resume / report reference.
    output = {
        "model_metrics": model_metrics,
        "random_metrics": random_metrics,
        "planning_metrics": planning_metrics,
        "generalization_metrics": gen_metrics,
        "multi_environment_results": env_results,
        "failure_case": {
            "biased_planning_success": biased_planning_episode.success,
            "biased_planning_steps": biased_planning_episode.steps,
            "predicted_actual_mismatch_rate": mismatch_rate,
            "mismatch_count": mismatches,
            "trajectory_length": len(actual_path) - 1,
        },
        "analysis": analysis,
    }
    output_path = Path("results.json")
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nSaved results to: {output_path.resolve()}")

    # 13) Quick goal-directed planning demo from env.start.
    demo_episode = run_episode_planning(env, world_model, max_steps=30, depth=3)
    print(
        f"Demo from start={env.start} -> success={demo_episode.success}, "
        f"steps={demo_episode.steps}"
    )
    print(
        "Failure-case demo with biased model "
        f"-> success={biased_planning_episode.success}, "
        f"steps={biased_planning_episode.steps}, mismatch_rate={mismatch_rate:.3f}"
    )


if __name__ == "__main__":
    main()
