"""
Small, stateless math helpers used across the RSSM: symlog scaling and
KL balancing with free bits (DreamerV3, Hafner et al. 2023, §2-3).

Kept deliberately tiny and dependency-free so `rssm.py` can import these
without pulling in any training-loop logic.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Compress large magnitudes while leaving small values ~linear.

    symlog(x) = sign(x) * log(1 + |x|)

    Used on network inputs/targets (observations, rewards) so the model
    doesn't need to learn wildly different scales per environment.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Inverse of `symlog`. Used to map predictions back to real units."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def kl_balancing(
    post_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    alpha: float = 0.8,
    free_bits: float = 1.0,
) -> torch.Tensor:
    """DreamerV3-style balanced, free-bits-clipped KL loss.

    Two problems with a plain KL(posterior || prior) term:
      1. It's cheaper for the model to move the *prior* toward the
         posterior than to learn a good posterior — the dynamics network
         "gives up" and just predicts whatever the encoder says. Balancing
         fixes this by training the prior faster than the posterior:
         we take a weighted mix of KL(sg(post) || prior) and
         KL(post || sg(prior)), weighted more toward the first.
      2. The KL term can dominate the loss and collapse the posterior to
         the prior (posterior collapse). Free bits fixes this by not
         penalizing KL below a small threshold — the model gets that much
         "budget" for free.

    Args:
        post_logits: (..., num_categoricals, num_classes) posterior logits.
        prior_logits: same shape, prior (dynamics-predicted) logits.
        alpha: weight on the "train prior toward stopped-gradient posterior"
            term. DreamerV3 default is 0.8.
        free_bits: minimum nats of KL per categorical group that are not
            penalized. DreamerV3 default is 1.0.

    Returns:
        Scalar loss (mean over batch and time).
    """
    post_dist = torch.distributions.OneHotCategorical(logits=post_logits)
    prior_dist = torch.distributions.OneHotCategorical(logits=prior_logits)

    post_sg_dist = torch.distributions.OneHotCategorical(logits=post_logits.detach())
    prior_sg_dist = torch.distributions.OneHotCategorical(logits=prior_logits.detach())

    # KL(sg(post) || prior): trains the prior to catch up to the posterior.
    kl_prior_train = torch.distributions.kl_divergence(post_sg_dist, prior_dist)
    # KL(post || sg(prior)): trains the posterior, but only weakly, so it
    # can't just collapse onto whatever the prior currently predicts.
    kl_post_train = torch.distributions.kl_divergence(post_dist, prior_sg_dist)

    # Sum over the categorical groups (last non-batch dim after kl_divergence
    # reduces the class dim), free-bits-clip each term, then combine.
    kl_prior_train = kl_prior_train.sum(-1)
    kl_post_train = kl_post_train.sum(-1)

    kl_prior_train = torch.clamp(kl_prior_train, min=free_bits)
    kl_post_train = torch.clamp(kl_post_train, min=free_bits)

    loss = alpha * kl_prior_train + (1 - alpha) * kl_post_train
    return loss.mean()


def straight_through_sample(logits: torch.Tensor) -> torch.Tensor:
    """Sample a one-hot categorical latent with a straight-through gradient.

    Forward pass: a hard one-hot sample (so downstream modules see a
    discrete latent, matching what happens at imagination time). Backward
    pass: gradients flow as if it were the softmax (Gumbel-softmax-free
    straight-through trick), so the encoder/posterior net stays trainable.
    """
    probs = F.softmax(logits, dim=-1)
    sample = torch.distributions.OneHotCategorical(probs=probs).sample()
    return sample + probs - probs.detach()
