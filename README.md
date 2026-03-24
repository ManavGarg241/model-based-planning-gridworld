# World Model Learning and Planning in Grid World

## Goal
Learn a **world model** (predict next state from current state + action) and use it to **plan actions** in a simple deterministic environment.

This project is intentionally focused and high-signal:
- No complex RL algorithms (no PPO, DQN)
- Deterministic grid world dynamics
- Short-horizon planning with a learned transition model

---

## What is a World Model?
A world model learns environment dynamics:

\[
(s_t, a_t) \rightarrow \hat{s}_{t+1}
\]

Instead of learning a policy directly, we learn to predict what will happen next. Then we plan by simulating candidate action sequences with the learned model and selecting the best one.

---

## What Was Built

### 1) Environment (Grid World)
- Grid size: `6x6`
- Deterministic transitions
- Defined start, goal, and obstacles
- Actions: `up`, `down`, `left`, `right`
- Transition function: `next_state = step(state, action)`

### 2) Dataset Generation
- Random policy generates transitions `(s_t, a_t, s_{t+1})`
- Dataset size: `10,000` transitions (within required `5k-20k`)
- Includes random resets to improve state coverage

### 3) World Model Training
- Input: state `(x, y)` + action one-hot vector
- Output: next state `(x', y')`
- Model: MLP (`2 hidden layers`: 64, 64)
- Loss proxy: MSE (scikit-learn MLPRegressor optimizes squared error)

### 4) Model Evaluation
- MSE on holdout set
- Exact next-state prediction accuracy (after rounding to grid cells)
- Average Manhattan error

### 5) Planning (Core)
- From current state, try action sequences of depth `3` by default (can increase to `4` or `5`)
- Simulate transitions with learned model
- Choose sequence that reaches the goal (or fallback to the closest trajectory)
- Execute first action in real environment

### 6) Baseline Comparison
- Compare:
  - Random policy
  - Planning with learned model
- Metrics:
  - Success rate
  - Steps to goal

### 7) Generalization Test
- Evaluates on 3 different layouts with different start/goal settings
- Also includes a changed-layout environment to test transfer without retraining

### 8) Analysis
- Reports relation between model accuracy and planning success
- Notes failure modes from model prediction errors
- Explains when planning breaks (error compounding)
- Includes an explicit failure case with a slightly biased model

### 9) Visualization
- Text grid visualization of path
- `outputs/planning_path.png`: planning trajectory
- `outputs/random_vs_planning.png`: side-by-side random vs planning paths
- `outputs/multi_env_metrics.png`: success-rate and average-steps bars across layouts
- `outputs/failure_predicted_vs_actual.png`: model-predicted vs actual rollout in failure case

---

## Project Structure

- `main.py`: full experiment pipeline
- `src/environment.py`: grid world and transitions
- `src/data.py`: dataset generation and splitting
- `src/model.py`: MLP world model + metrics
- `src/planner.py`: depth-limited planning
- `src/evaluate.py`: random/planning benchmark + metrics
- `src/visualize.py`: single-path, policy-comparison, and metrics plots
- `results.json`: saved metrics and analysis (generated after run)

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run experiment:

```bash
python main.py
```

3. Inspect outputs:
- Console metrics and analysis
- `results.json` for report/resume evidence
- `outputs/*.png` for visual evidence (baseline, multi-env, and failure-case plots)

---

## Key Insights to Expect
1. Better transition prediction quality usually improves planning success.
2. Planning outperforms random when world model is reasonably accurate.
3. Planning failures often come from compounding one-step prediction errors over depth.
4. Generalization can drop if changed obstacle dynamics were underrepresented in training data.

---

## Limitations
1. Deterministic and small environment only.
2. Short-horizon planner (depth-limited brute force).
3. Model predicts coordinates directly; no uncertainty estimate.
4. No online model updates during control.

---

## Summary
Built a model-based planning system in a deterministic grid world by learning transition dynamics with an MLP from random interaction data and using depth-limited lookahead planning. Demonstrated improved success rate over random baseline, quantified prediction and control performance, and evaluated generalization under changed layouts.
