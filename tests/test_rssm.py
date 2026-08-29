"""Shape and loss-computation tests for RSSM (roadmap P1.1).

Deliberately small dims (deter_dim=16, 4x4 categoricals) so this runs in
well under a second on CPU — no GPU needed for these tests.
"""

import torch

from rssmlite import RSSM

OBS_DIM = 6
ACTION_DIM = 3
BATCH = 4
SEQ_LEN = 5


def make_rssm() -> RSSM:
    return RSSM(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        deter_dim=16,
        embed_dim=16,
        num_categoricals=4,
        num_classes=4,
        hidden_dim=32,
    )


def make_batch():
    obs = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
    action = torch.randn(BATCH, SEQ_LEN, ACTION_DIM)
    reward = torch.randn(BATCH, SEQ_LEN)
    cont = torch.ones(BATCH, SEQ_LEN)
    return obs, action, reward, cont


def test_initial_state_shapes():
    rssm = make_rssm()
    state = rssm.initial_state(BATCH, torch.device("cpu"))
    assert state["deter"].shape == (BATCH, 16)
    assert state["stoch"].shape == (BATCH, 16)  # 4 categoricals * 4 classes


def test_observe_output_shapes():
    rssm = make_rssm()
    obs, action, _, _ = make_batch()
    rollout = rssm.observe(obs, action)

    assert rollout["deter"].shape == (BATCH, SEQ_LEN, 16)
    assert rollout["stoch"].shape == (BATCH, SEQ_LEN, 16)
    assert rollout["post_logits"].shape == (BATCH, SEQ_LEN, 4, 4)
    assert rollout["prior_logits"].shape == (BATCH, SEQ_LEN, 4, 4)


def test_stoch_is_one_hot_per_categorical():
    """Each of the 4 categorical groups should sum to 1 (one-hot), even
    though gradients flow through via the straight-through estimator."""
    rssm = make_rssm()
    obs, action, _, _ = make_batch()
    rollout = rssm.observe(obs, action)
    stoch = rollout["stoch"].view(BATCH, SEQ_LEN, 4, 4)
    sums = stoch.sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_imagine_output_shapes():
    rssm = make_rssm()
    state = rssm.initial_state(BATCH, torch.device("cpu"))
    horizon = 7

    def random_policy(feature):
        return torch.randn(feature.shape[0], ACTION_DIM)

    rollout = rssm.imagine(state, random_policy, horizon)
    assert rollout["deter"].shape == (BATCH, horizon, 16)
    assert rollout["stoch"].shape == (BATCH, horizon, 16)
    assert rollout["action"].shape == (BATCH, horizon, ACTION_DIM)


def test_loss_is_finite_and_has_expected_keys():
    rssm = make_rssm()
    obs, action, reward, cont = make_batch()
    losses = rssm.loss(obs, action, reward, cont)

    assert set(losses.keys()) == {"total", "recon", "reward", "continue", "kl"}
    for name, value in losses.items():
        assert value.dim() == 0, f"{name} loss should be scalar"
        assert torch.isfinite(value), f"{name} loss is not finite"


def test_loss_is_differentiable():
    """Gradients should reach the encoder, i.e. the straight-through
    estimator and the recurrent unroll aren't silently blocking backprop."""
    rssm = make_rssm()
    obs, action, reward, cont = make_batch()
    losses = rssm.loss(obs, action, reward, cont)
    losses["total"].backward()

    encoder_grad = next(rssm.encoder.parameters()).grad
    assert encoder_grad is not None
    assert torch.isfinite(encoder_grad).all()
    assert encoder_grad.abs().sum() > 0


def test_kl_free_bits_floor():
    """With free_bits set very high, KL loss should sit at (close to) the
    balanced floor rather than being driven to zero — confirms free bits
    is actually clamping, not a no-op."""
    from rssmlite.utils import kl_balancing

    logits_a = torch.zeros(BATCH, 4, 4)
    logits_b = torch.zeros(BATCH, 4, 4)  # identical -> true KL is ~0
    loss = kl_balancing(logits_a, logits_b, alpha=0.8, free_bits=2.0)
    assert torch.isclose(loss, torch.tensor(2.0), atol=1e-4)
