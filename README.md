# LunarLander PPO from Scratch

This project implements Proximal Policy Optimization (PPO) from scratch in PyTorch on the LunarLander environment from Gymnasium.

## Approach
I used an actor-critic PPO setup with:
- parallel environments for rollout collection
- generalized advantage estimation (GAE)
- clipped PPO objective
- value loss and entropy bonus
- observation normalization

## Environment
The original prompt mentioned LunarLander-v2, but Gymnasium currently deprecates v2, so I used LunarLander-v3 as the maintained equivalent version.

## Results
The final trained agent achieved a mean evaluation reward of about 287.75 over 20 evaluation episodes.

## Design Decisions
- used multiple parallel environments to collect data faster
- used observation normalization for more stable training
- saved the best checkpoint based on evaluation performance
- evaluated the final model separately after training

