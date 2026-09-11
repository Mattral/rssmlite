"""
Actor, Critic, EMA target critic, and lambda-return computation.

Key design change vs v1: the Critic now has a paired `CriticEMA` (exponential
moving average copy) used exclusively to compute bootstrap targets for
lambda-returns. Without this, the critic's target and its own predictions
co-move during training, causing the divergence observed in early T4 runs
(critic_loss exploding to 40-50 and never recovering). EMA target networks
are standard in off-policy RL (DQN, SAC, TD3, DreamerV3) for exactly this
reason.
"""

from __future__ import annotations

import copy

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
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.discrete:
            logits = self.net(feature)
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
    def __init__(self, feature_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.net = mlp(feature_dim, hidden_dim, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature).squeeze(-1)

    def make_ema(self, tau: float = 0.02) -> "CriticEMA":
        """Create a paired EMA target network. `tau` is the update rate —
        smaller = slower-moving target = more stable bootstrap values.
        DreamerV3 uses tau=0.02."""
        return CriticEMA(self, tau)


class CriticEMA:
    """Exponential moving average copy of a Critic, used only for computing
    bootstrap targets. Never receives gradient updates — only updated via
    soft_update() after each critic gradient step.

    target_params = (1 - tau) * target_params + tau * online_params
    """

    def __init__(self, critic: Critic, tau: float = 0.02):
        self.tau = tau
        self._target = copy.deepcopy(critic)
        for p in self._target.parameters():
            p.requires_grad_(False)

    def __call__(self, feature: torch.Tensor) -> torch.Tensor:
        return self._target(feature)

    def soft_update(self, critic: Critic) -> None:
        for target_p, online_p in zip(self._target.parameters(), critic.parameters()):
            target_p.data.lerp_(online_p.data, self.tau)

    def to(self, device) -> "CriticEMA":
        self._target = self._target.to(device)
        return self


def lambda_return(
    rewards: torch.Tensor,
    continues: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """TD(lambda) returns. values[:, -1] is the EMA-target bootstrap value
    at the imagined horizon — using the slow-moving target here rather than
    the online critic is what stabilises training.

    Args:
        rewards:   (B, H)   imagined rewards at steps 1..H
        continues: (B, H)   P(episode continues) at steps 1..H
        values:    (B, H+1) EMA-target value at step 0 (start) and 1..H
    """
    horizon = rewards.shape[1]
    returns = torch.zeros_like(rewards)
    returns[:, -1] = rewards[:, -1] + gamma * continues[:, -1] * values[:, -1]
    for t in reversed(range(horizon - 1)):
        bootstrap = (1 - lam) * values[:, t + 1] + lam * returns[:, t + 1]
        returns[:, t] = rewards[:, t] + gamma * continues[:, t] * bootstrap
    return returns
