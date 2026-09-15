# Simulation Feature Planning

This directory contains planning and status documents for major feature development.

## Active Features

| Feature | Status | Documents |
|---------|--------|-----------|
| Centralized Animal Policy | Planning | [01-centralized-policy.md](./01-centralized-policy.md), [01a-policy-implementation-detail.md](./01a-policy-implementation-detail.md) |
| Genetic Algorithm System | Planning | [02-genetic-algorithm.md](./02-genetic-algorithm.md) |
| Performance and RL foundation | In progress | [03-performance-and-rl-foundation.md](./03-performance-and-rl-foundation.md) |
| Reinforcement learning | In progress | [04-reinforcement-learning.md](./04-reinforcement-learning.md) |
| Project review | Reference | [05-review.md](./05-review.md) |

## Overview

These two features are interdependent and together will enable:
1. A well-abstracted decision-making system compatible with future RL integration
2. Evolutionary dynamics through genetic diversity and selection pressure

The policy system provides the "brain" that can be parameterized, while the genetic algorithm provides the mechanism for those parameters to evolve across generations.
