"""
The non-recurrent pieces of the RSSM: encoder, posterior net, decoder,
reward head, continue head. All are plain MLPs — v1 targets are
state-vector Gymnasium environments (Section 7 of SPEC.md), not pixels,
so there's no CNN here. A pixel encoder/decoder can be added later without
touching `rssm.py`, since `RSSM` only calls these by their public methods.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rssmlite.utils import symlog


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_hidden_layers: int = 2) -> nn.Sequential:
    """Small helper so each network below is a one-liner, not boilerplate."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    for _ in range(num_hidden_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """Observation -> embedding. Applies symlog first so reward/position/
    velocity components on very different scales don't dominate training."""

    def __init__(self, obs_dim: int, embed_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.net = _mlp(obs_dim, hidden_dim, embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(symlog(obs))


class PosteriorNet(nn.Module):
    """(deterministic state, observation embedding) -> posterior logits over
    the stochastic latent. This is the network that "cheats" by looking at
    the real observation — only used during training, never during
    imagination (see RSSM.imagine)."""

    def __init__(
        self,
        deter_dim: int,
        embed_dim: int,
        num_categoricals: int,
        num_classes: int,
        hidden_dim: int = 200,
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.net = _mlp(deter_dim + embed_dim, hidden_dim, num_categoricals * num_classes)

    def forward(self, deter: torch.Tensor, embed: torch.Tensor) -> torch.Tensor:
        logits = self.net(torch.cat([deter, embed], dim=-1))
        return logits.view(*logits.shape[:-1], self.num_categoricals, self.num_classes)


class Decoder(nn.Module):
    """(deterministic + stochastic feature) -> reconstructed observation, in
    symlog space. Caller applies `symexp` to get back real units."""

    def __init__(self, feature_dim: int, obs_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.net = _mlp(feature_dim, hidden_dim, obs_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class RewardHead(nn.Module):
    """Feature -> scalar reward prediction, in symlog space (matches
    Decoder's convention; caller applies `symexp` for real units)."""

    def __init__(self, feature_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.net = _mlp(feature_dim, hidden_dim, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature).squeeze(-1)


class ContinueHead(nn.Module):
    """Feature -> logit for P(episode continues). Trained with BCE against
    `1 - done`. Lets imagined rollouts learn to predict episode termination
    instead of always imagining a fixed horizon."""

    def __init__(self, feature_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.net = _mlp(feature_dim, hidden_dim, 1)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature).squeeze(-1)
