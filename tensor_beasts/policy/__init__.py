"""
Policy module for centralized animal decision-making.

This module provides a policy abstraction that separates decision-making
from action execution in the animal update loop.

Following RL convention:
- Observation: Complete input to policy (from observations.py)
- Action: Output from policy
"""

from tensor_beasts.policy.base import Action, AnimalPolicy
from tensor_beasts.policy.rule_based import RuleBasedPolicy
from tensor_beasts.policy.parameterized import ParameterizedPolicy, ParameterBounds, DEFAULT_BOUNDS

__all__ = [
    "Action",
    "AnimalPolicy",
    "RuleBasedPolicy",
    "ParameterizedPolicy",
    "ParameterBounds",
    "DEFAULT_BOUNDS",
]
