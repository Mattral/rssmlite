"""
Actor and Critic, trained entirely on rollouts imagined by the RSSM's
dynamics/reward/continue heads — never touching the real environment during
this phase. This is what makes the approach sample-efficient: one real
episode can be "replayed" through imagination thousands of times.

Handles both action spaces in Section 7's target envs: Discrete (CartPole,
Acrobot, LunarLander) via a straight-through categorical, Continuous
(Pendulum) via a reparameterized Normal. No tanh squashing on the
continuous head yet — a documented simplification, see README caveats.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal, OneHotCategorical

from rssmlite.utils import mlp, straight_through_sample


class Actor(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int, discrete: bool, hidden_dim: int = 200):
        super().__init__()
        self.discrete = discrete
        self.action_dim = action_dim
        self.net = mlp(feature_dim, hidden_dim, action_dim)
        if not discrete:
            # State-independent log-std, a common simplification (e.g. PPO's
            # default) that avoids the head needing to learn variance from
            # very little imagined data early in training.
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (action, entropy). `action` is a *differentiable* sample
        — both branches support backprop straight through to `feature`,
        which is what lets the actor loss backprop through the imagined
        rollout (Dreamer's "dynamics backprop" trick, no REINFORCE needed).
        """
        if self.discrete:
            logits = self.net(feature)
            # Reuse the same straight-through trick as the stochastic latent,
            # treating the action as a single categorical group.
            action = straight_through_sample(logits.unsqueeze(-2)).squeeze(-2)
            entropy = OneHotCategorical(logits=logits).entropy()
            return action, entropy

        mean = self.net(feature)
        std = torch.nn.functional.softplus(self.log_std) + 1e-4
        dist = Normal(mean, std)
        action = dist.rsample()
        entropy = dist.entropy().sum(-1)
        return action, entropy


class Critic(nn.Module):
    """Feature -> scalar value estimate, trained to match lambda-returns
    computed over imagined rollouts."""

    def __init__(self, feature_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.net = mlp(feature_dim, hidden_dim, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature).squeeze(-1)


def lambda_return(
    rewards: torch.Tensor,
    continues: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """TD(lambda) returns over an imagined rollout (Dreamer's actor-critic
    objective, Hafner et al. 2023 Eq. 6).

    Args:
        rewards: (B, H) predicted reward for entering each imagined state.
        continues: (B, H) predicted P(episode continues) at each state.
        values: (B, H+1) critic estimate at the real starting state (index
            0) and every imagined state (indices 1..H). Index H doubles as
            the bootstrap target for the final step.
        gamma, lam: discount and the usual TD-lambda mixing coefficient.

    Returns:
        (B, H) lambda-returns, one per imagined step (matching `rewards`).
    """
    horizon = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    returns[:, -1] = rewards[:, -1] + gamma * continues[:, -1] * values[:, -1]
    for t in reversed(range(horizon - 1)):
        bootstrap = (1 - lam) * values[:, t + 1] + lam * returns[:, t + 1]
        returns[:, t] = rewards[:, t] + gamma * continues[:, t] * bootstrap
    return returns
