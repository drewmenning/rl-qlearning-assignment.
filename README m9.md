# rl-qlearning-assignment

Reinforcement Learning: Q-Learning implementation on OpenAI Gymnasium's `Taxi-v3` environment.

## Overview

This project implements the **Q-Learning** algorithm from scratch on the `Taxi-v3` discrete environment. The agent learns to navigate a 5×5 grid, pick up a passenger, and drop them off at the correct location — purely through trial-and-error interaction with the environment.

## Environment: Taxi-v3

| Property | Value |
|---|---|
| State space | 500 discrete states |
| Action space | 6 (N, S, E, W, Pick Up, Drop Off) |
| Reward: step | -1 |
| Reward: illegal action | -10 |
| Reward: successful drop-off | +20 |

## Results

| Metric | Random Policy | Q-Learned Policy |
|---|---|---|
| Avg reward (200 eps) | ~-800 | ~+7 |
| Success rate | ~0% | ~95%+ |
| Avg steps to solve | 200 (timeout) | ~13 |

After **5,000 training episodes**, the agent achieves a high success rate, demonstrating clear learning from an initially random policy.

## Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| Alpha (α) | 0.8 | Learning rate |
| Gamma (γ) | 0.95 | Discount factor |
| Epsilon (ε) | 1.0 → 0.01 | Exploration rate (decaying) |
| Epsilon decay | 0.995 | Multiplicative decay per episode |
| Episodes | 5,000 | Total training episodes |

## Files

```
rl-qlearning-assignment/
├── rl_qlearning_assignment.ipynb   # Main Jupyter notebook (all parts A–D)
├── README.md                       # This file
└── images/
    ├── qlearning_results.png       # Training curves + heatmap
    ├── qtable_heatmap.png          # Full Q-table visualization
    └── policy_comparison.png       # Random vs trained policy bar chart
```

## How to Run

### Google Colab (recommended)
1. Upload `rl_qlearning_assignment.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. Run all cells (`Runtime → Run all`)

### Local
```bash
pip install gymnasium matplotlib seaborn numpy
jupyter notebook rl_qlearning_assignment.ipynb
```

## Key Findings

1. **Exploration is critical early on** — epsilon-greedy with slow decay allows the agent to discover high-reward states before committing to a policy.
2. **High alpha works for deterministic environments** — Taxi-v3 is fully deterministic, so aggressive updates (α=0.8) converge faster without instability.
3. **Convergence takes ~3,000+ episodes** — the moving average reward stabilizes around episode 3,000–4,000 even though improvement starts around episode 1,000.
4. **Discrete Q-tables scale poorly** — Taxi-v3 has 500 states; real-world problems may have millions, requiring Deep Q-Networks (DQN) instead.

## References

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Ch. 18
- Watkins, C. J. C. H. (1989). *Learning from delayed rewards* (original Q-Learning paper)
