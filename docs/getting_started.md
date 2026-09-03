# Getting Started

## Installation

**From PyPI** (stable releases):
```bash
pip install rssmlite
pip install rssmlite[envs]   # adds gymnasium
pip install rssmlite[viz]    # adds matplotlib for plots
```

**From source** (recommended for development):
```bash
git clone https://github.com/Mattral/rssmlite.git
cd rssmlite
pip install -e ".[dev,envs,viz]"
pytest tests/   # should be 21 passed
```

**Python version:** 3.9–3.12. **PyTorch:** ≥ 2.0.

---

## Your first training run

### Via the CLI

```bash
python scripts/train.py --config configs/cartpole.yaml
```

This will:
1. Create a `CartPole-v1` environment.
2. Seed the replay buffer with 5 random episodes.
3. Train the world model + actor-critic, logging every iteration.
4. Checkpoint to `./checkpoints/` every 10 000 environment steps.

Expected output (first few lines):
```
[step 312] episode_reward=22.0 wm/total=2.41 wm/recon=0.14 ... ac/actor_loss=-0.03
[step 589] episode_reward=35.0 wm/total=2.18 ...
```

To resume after a crash or Colab disconnect:
```bash
python scripts/train.py --config configs/cartpole.yaml \
  --resume checkpoints/checkpoint_10000.pt \
  --checkpoint-dir checkpoints
```

Note: `steps` in the config is a **total** budget across the whole run
(including pre-resume steps), not "N additional steps from here." If you
resume a checkpoint at step 50 000 and your config says `steps: 200000`,
training will continue until step 200 000.

### Via the Python API

```python
import gymnasium as gym
from rssmlite import RSSMAgent

env = gym.make("CartPole-v1")
agent = RSSMAgent.from_config("configs/cartpole.yaml", env=env)
agent.train(env, steps=200_000, checkpoint_dir="./ckpts")
```

Or for quick experimentation without a config file:
```python
agent = RSSMAgent.from_env(env)   # uses default hyperparameters
agent.train(env, steps=50_000)
```

---

## Inspecting what the model learned

```python
# Visualize an imagined trajectory
rollout = agent.imagine_rollout(steps=15)
print(rollout["reward_pred"])   # predicted rewards over 15 imagined steps

# 2D PCA of the stochastic latent space
fig = agent.visualize_latent_space()
fig.savefig("latent_space.png")

# Reconstruction quality on a real batch from the replay buffer
from rssmlite import reconstruction_report
batch = agent.buffer.sample(batch_size=32, seq_len=50)
report = reconstruction_report(agent, batch)
print(f"MSE (real units): {report['mse_real']:.4f}")
print(f"per-feature MSE: {report['per_dim_mse']}")
```

---

## Running on Colab

See [`docs/colab_guide.md`](colab_guide.md) for Colab-specific setup
(mounting Drive, checkpointing, avoiding disconnect data loss).

The notebooks in `notebooks/` are the easiest starting point:
- `01_world_model_intro.ipynb` — what an RSSM is and how to read its outputs.
- `02_cartpole_experiment.ipynb` — a full CartPole run with plots.
- `03_lunarlander_experiment.ipynb` — the hardest env in v1 scope.

---

## Available environments

| Config | Env ID | Action type | Expected solve time (T4) |
|---|---|---|---|
| `configs/cartpole.yaml` | CartPole-v1 | Discrete | ~20 min |
| `configs/acrobot.yaml` | Acrobot-v1 | Discrete | ~30 min |
| `configs/pendulum.yaml` | Pendulum-v1 | Continuous | ~20 min |
| `configs/lunarlander.yaml` | LunarLander-v3 | Discrete | ~60 min |

"Solve time" estimates are approximate and unverified — they reflect the
expected T4 budget from the project spec, not measured results. See
`ROADMAP.md P1.3` for the current status of convergence verification.
