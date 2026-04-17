# LunarLander PPO from Scratch

This project implements Proximal Policy Optimization (PPO) from scratch in PyTorch on the LunarLander environment from Gymnasium.

## Files
- `LunarLander_PPO.ipynb`: final PPO implementation and main submission notebook
- `baselines/original_actor_critic.ipynb`: an earlier baseline version used during exploration before the final PPO implementation

## Approach
The final submission uses an actor-critic PPO setup with:
- parallel environments for rollout collection
- generalized advantage estimation (GAE)
- clipped PPO objective
- value loss and entropy bonus
- observation normalization

## Environment
The original prompt mentioned LunarLander-v2, but Gymnasium currently deprecates v2, so I used LunarLander-v3 as the maintained equivalent version.

## Results
The final trained PPO agent achieved a mean evaluation reward of about 287.75 over 20 evaluation episodes.

## Design Decisions
- I first built a simpler baseline to understand the environment and training loop
- then moved to a PPO implementation for more stable and stronger performance
- used multiple parallel environments to collect data faster
- used observation normalization for more stable training
- saved the best checkpoint based on evaluation performance

