"""Swappable dynamics backbones for `RSSM`. Each backbone must implement
`forward(deter, stoch, action) -> deter` and `prior_logits(deter) -> logits`.
"""

from rssmlite.backbones.gru import GRUDynamics

__all__ = ["GRUDynamics"]

# TransformerDynamics lands in P2 (roadmap) as the ablation/novelty variant.
