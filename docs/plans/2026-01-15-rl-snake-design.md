---
title: "Self-Evolving Snake RL Design"
tags: [thesis, rl, dqn, ga, snake]
status: approved
owner: marco
updated: 2026-01-15
---

## Background
- Problem: Build a Snake game that can learn to play through reinforcement learning and show "self-evolution" in its performance and configuration.
- Goal: Implement a reproducible RL pipeline (DQN baseline) and a genetic algorithm (GA) that evolves hyperparameters and reward weights, with measurable improvement.
- Motivation: Align with the thesis topic "self-evolving snake game" while keeping implementation and experiments feasible on a MacBook (CPU and optional MPS).

## Scope
- In scope:
  - 10x10 grid Snake environment with wall collisions (death on hit).
  - DQN baseline with experience replay and target network.
  - GA to evolve hyperparameters and reward shaping weights.
  - Headless training mode and a separate render/demo mode.
  - Logging of reward, score, loss, and GA generation results.
- Out of scope:
  - Multiplayer, online play, or web deployment.
  - Large-scale training infrastructure or distributed RL.
  - Complex 3D graphics or non-grid physics.

## Requirements and Approach
### Key requirements
- Environment implements Gym-like API: `reset()`, `step(action)`, `render()`, `close()`.
- Discrete action space: up, down, left, right.
- Reward shaping supports weights that can be evolved by GA.
- Two training tracks for comparison: baseline DQN and DQN + GA.
- Runs on CPU by default; optional MPS acceleration for PyTorch.
- Reproducibility: seed control for env, RNG, and model init.
- Final evaluation uses 50 episodes per model for comparison.

### Approach overview
- State representation: grid channels (snake body, snake head, food) flattened for DQN input.
- DQN details:
  - Experience replay buffer.
  - Target network update every N steps.
  - Epsilon-greedy policy with decay.
  - Double DQN enabled by default for stability.
- GA details:
  - Individual = vector of hyperparameters and reward weights.
  - Short training budget per individual, then evaluation score.
  - Selection by top-K, crossover by mixing parameters, mutation by small noise.
  - Best individual carried to next generation (elitism).

### Architecture
- `env/`: Snake environment and rendering utilities.
- `agents/`: DQN agent, networks, and replay buffer.
- `train/`: Trainer loop, evaluation runner, and logging.
- `evolve/`: GA controller and config mutation.
- `config/`: YAML config and validation.
- `scripts/`: CLI entry points for train and demo.

### Data flow
1. Trainer resets env and gets initial state.
2. Agent selects action -> env `step()`.
3. Transition stored in replay buffer.
4. Sample batch -> update policy network.
5. Periodically sync target network.
6. GA loop: evaluate population -> select -> crossover -> mutate -> next generation.

### Error handling
- Validate config ranges (learning rate, epsilon, reward weights).
- Guard against invalid actions and out-of-range positions.
- Handle model checkpoint load failures with clear messages.
- Early stop if training diverges (NaN loss or zero reward for long spans).

### Testing
- Unit tests: env step mechanics, collision detection, reward outputs.
- Integration tests: short training run for N steps with deterministic seed.
- Smoke test: demo mode renders and exits cleanly.

## Milestones
- M1: Environment and baseline Snake gameplay (grid, collisions, food).
- M2: DQN baseline training and evaluation metrics.
- M3: GA evolution loop and comparison experiments.
- M4: Demo mode and final plots for thesis.

## Acceptance criteria
- Functional:
  - Agent can learn to reach food with average score improving over time.
  - GA run produces better or more stable results than baseline in at least one metric.
  - Demo mode can replay a trained model in real time.
- Quality:
  - Logs and plots are reproducible from a single config file.
  - Training finishes within CPU time budget on a MacBook.

## Dependencies and Risks
> [!warning]
> Dependencies: Python 3.10+, PyTorch (with optional MPS), pygame, numpy, matplotlib.
> Risks: Training instability or slow convergence. Mitigation: small board (10x10), reward shaping, early stopping, and limited training budgets per GA individual.

> [!note]
> Decisions: Double DQN enabled by default; evaluation uses 50 episodes per model.
