# Colab Guide

All target environments and notebooks are designed to run on a **free Colab
T4 GPU** with no local setup. This page covers the Colab-specific details
that don't belong in `getting_started.md`.

---

## Bootstrap cell (same in every notebook)

Every `rssmlite` notebook starts with this cell. Copy it verbatim:

```python
# ── Bootstrap ──────────────────────────────────────────────────────────────
# Run this cell first. It installs rssmlite, mounts Google Drive for
# checkpointing, and confirms the GPU is a T4.

!pip install -q rssmlite[envs,viz]

from google.colab import drive
drive.mount("/content/drive")

import torch
gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Device: {gpu}")
# ───────────────────────────────────────────────────────────────────────────
```

---

## Checkpointing to Google Drive

Pass any mounted Drive path as `checkpoint_dir`:

```python
CHECKPOINT_DIR = "/content/drive/MyDrive/rssmlite-ckpts/cartpole"

agent.train(
    env,
    steps=200_000,
    checkpoint_dir=CHECKPOINT_DIR,
    checkpoint_every=10_000,
)
```

`rssmlite` creates the directory if it doesn't exist. Checkpoints are
named `checkpoint_{step}.pt`. Each one saves the full optimizer state, so
resume is exact — not just weight loading.

---

## Resuming after a disconnect

Colab free-tier sessions disconnect after ~90 minutes of inactivity (or
sooner if the runtime is reclaimed). The expected recovery flow is:

1. Re-run the bootstrap cell (reinstalls the package, remounts Drive).
2. Find the latest checkpoint:
   ```python
   import os, glob
   ckpts = sorted(glob.glob(f"{CHECKPOINT_DIR}/checkpoint_*.pt"))
   print("Latest:", ckpts[-1])
   ```
3. Resume:
   ```python
   from rssmlite import RSSMAgent
   agent = RSSMAgent.load_checkpoint(ckpts[-1])
   print(f"Resuming from step {agent._env_steps}")
   agent.train(env, steps=200_000, checkpoint_dir=CHECKPOINT_DIR)
   ```

`steps=200_000` is a total budget — if the checkpoint is at step 80 000,
training will continue until step 200 000.

---

## Keeping the session alive

The standard keep-alive trick: paste this in your browser console
(F12 → Console):

```javascript
function KeepAlive() {
  document.querySelector("#top-toolbar").click();
  window.setTimeout(KeepAlive, 60000);
}
KeepAlive();
```

This doesn't prevent the hard 12-hour Colab session limit, but it avoids
the shorter "inactivity" disconnect. Checkpoint every 10 000 steps anyway
— don't rely on keep-alive alone.

---

## Checking VRAM usage

All target environments are state-vector (not pixel), so VRAM stays low
(well under 2 GB for the full CartPole model). Confirm with:

```python
import torch
print(f"VRAM allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
print(f"VRAM cached:    {torch.cuda.memory_reserved() / 1e9:.2f} GB")
```

If you see >4 GB, something unexpected is happening (e.g. a large batch
size or `seq_len`). The `configs/*.yaml` defaults are tuned to stay safe
on a T4.

---

## Moving the model to GPU

`RSSMAgent` does **not** automatically move to GPU — this is intentional
(it avoids hidden device surprises). Move it explicitly after construction:

```python
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent = RSSMAgent.from_config("configs/cartpole.yaml", env=env)
agent.rssm.to(device)
agent.actor.to(device)
agent.critic.to(device)
```

A `.to(device)` convenience method on `RSSMAgent` itself is on the roadmap
for a future release.
