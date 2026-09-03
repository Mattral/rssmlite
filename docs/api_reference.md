# API Reference

## `rssmlite.RSSM`

The world model. Composes an encoder, GRU dynamics backbone, categorical
stochastic latent, decoder, reward head, and continue head.

```python
RSSM(
    obs_dim: int,
    action_dim: int,
    deter_dim: int = 200,
    embed_dim: int = 200,
    num_categoricals: int = 32,
    num_classes: int = 32,
    hidden_dim: int = 200,
    dynamics: nn.Module | None = None,   # defaults to GRUDynamics
    kl_alpha: float = 0.8,
    kl_free_bits: float = 1.0,
)
```

**Methods:**

`initial_state(batch_size, device) -> dict`
Returns `{"deter": zeros, "stoch": zeros}` — the starting state for a new
episode or the beginning of an imagined rollout.

`observe(obs_seq, action_seq) -> dict`
Teacher-forced pass over a real sequence. Returns `deter`, `stoch`,
`post_logits`, `prior_logits` stacked over the time dimension — everything
needed by `loss()`.

`imagine(initial_state, policy, horizon) -> dict`
Rolls forward `horizon` steps using only the prior (no observations).
`policy(feature) -> action` is called at each step. Returns `deter`,
`stoch`, `action`.

`loss(obs_seq, action_seq, reward_seq, continue_seq) -> dict`
Reconstruction + reward + continue + KL loss. Returns a dict with keys
`total`, `recon`, `reward`, `continue`, `kl`.

`feature(state) -> Tensor`
Concatenates `deter` and `stoch` into a single feature vector — what
every head (decoder, actor, critic) actually reads.

---

## `rssmlite.RSSMAgent`

Wraps an `RSSM` with an actor-critic trained purely on imagined rollouts.

```python
RSSMAgent(
    obs_dim: int,
    action_dim: int,
    discrete: bool,
    rssm_kwargs: dict | None = None,
    hidden_dim: int = 200,
    world_model_lr: float = 3e-4,
    actor_lr: float = 8e-5,
    critic_lr: float = 8e-5,
    gamma: float = 0.99,
    lam: float = 0.95,
    entropy_coef: float = 3e-4,
    replay_capacity_episodes: int = 500,
)
```

**Constructors:**

`RSSMAgent.from_env(env, **kwargs) -> RSSMAgent`
Infers `obs_dim`, `action_dim`, `discrete` from a Gymnasium env.

`RSSMAgent.from_config(path, env=None) -> RSSMAgent`
Builds from a YAML config file. Pass `env` to infer dims from the env
rather than specifying them in the config.

**Training:**

`agent.train(env, steps, seed_episodes=5, batch_size=16, seq_len=50, horizon=15, max_episode_steps=500, checkpoint_dir=None, checkpoint_every=10_000, log_every=1, log_fn=print)`
Main loop. Alternates collecting one real episode (via posterior
filtering) with one world-model gradient step and one actor-critic step.
`steps` is a **total** budget across the whole run, not "N more steps."

**Evaluation:**

`agent.imagine_rollout(steps=15) -> dict`
Rolls the current policy forward in imagination from a fresh initial
state. Returns `deter`, `stoch`, `action`, `obs_pred`, `reward_pred`.

`agent.visualize_latent_space(rollout=None) -> Figure`
2D PCA of stochastic latents from a rollout. Requires
`pip install rssmlite[viz]`.

**Checkpointing:**

`agent.save_checkpoint(path)`
Saves full state: model weights + optimizer state + env step count.
Supports Google Drive paths on Colab.

`RSSMAgent.load_checkpoint(path) -> RSSMAgent`
Fully reconstructs the agent including optimizer state. Training
resumes exactly where it left off.

---

## `rssmlite.ReplayBuffer`

```python
ReplayBuffer(capacity_episodes: int = 500)
```

`buffer.add_episode(obs, action, reward, done)`
Each argument is a `(episode_len, ...)` numpy array. `done[t]` should
be `True` only on the last transition.

`buffer.sample(batch_size, seq_len) -> dict`
Returns `(batch_size, seq_len, ...)` tensors: `obs`, `action`, `reward`,
`done`. Sequences never cross episode boundaries.

`buffer.can_sample(seq_len) -> bool`
Whether at least one stored episode is long enough to sample from.

---

## `rssmlite.reconstruction_report`

```python
reconstruction_report(agent, batch) -> dict
```

Keys: `mse_symlog` (scalar), `mse_real` (scalar), `per_dim_mse`
`(obs_dim,)` tensor. `batch` is a dict from `ReplayBuffer.sample()`.

---

## `rssmlite.run_evaluation_episodes`

```python
run_evaluation_episodes(agent, env, n_episodes=10) -> dict
```

Runs `n_episodes` greedy evaluation episodes. Returns `mean_return`,
`std_return`, `min_return`, `max_return`, `episode_lengths`.

---

## `rssmlite.backbones.GRUDynamics`

```python
GRUDynamics(
    deter_dim, stoch_dim, action_dim,
    num_categoricals, num_classes,
    hidden_dim=200,
)
```

The default dynamics backbone. Implements
`forward(deter, stoch, action) -> deter` and
`prior_logits(deter) -> (B, num_categoricals, num_classes)`.
Any object implementing those two methods can be passed as the
`dynamics` argument to `RSSM`.

---

## Utility functions (`rssmlite.utils`)

`symlog(x)` / `symexp(x)` — sign-preserving log/exp compression.

`kl_balancing(post_logits, prior_logits, alpha=0.8, free_bits=1.0)` —
balanced, free-bits-clipped KL loss (DreamerV3, §2).

`straight_through_sample(logits)` — hard one-hot sample with
straight-through gradient.

`mlp(in_dim, hidden_dim, out_dim, num_hidden_layers=2)` — shared MLP
builder (SiLU activations).
