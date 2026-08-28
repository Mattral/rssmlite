# rssmlite

A lightweight, well-documented, PyTorch-only library for learning RSSM-based
(Recurrent State-Space Model) world models — in the DreamerV2/V3 tradition —
on Gymnasium environments. Designed to run end-to-end on a free Google Colab
T4 GPU with zero external datasets.

> **Status:** early placeholder release (`v0.0.1`). This version exists to
> reserve the package name; it does not yet contain the RSSM implementation.
> Follow development at [github.com/Mattral/rssmlite](https://github.com/Mattral/rssmlite).

## What's coming

- `RSSM` — encoder, GRU-based recurrent state, categorical stochastic latent,
  decoder, reward head, continue head (swappable GRU/Transformer backbone).
- `RSSMAgent` — actor-critic trained entirely on imagined rollouts.
- `ReplayBuffer` — sequence storage and fixed-length sampling.
- Config-driven support for CartPole, Acrobot, Pendulum, and LunarLander.
- Runnable Colab notebooks with "Open in Colab" badges, zero manual setup.

## Install

```bash
pip install rssmlite
```

(v0.0.1 installs only a version marker — the real API arrives in v0.1.0.)

## License

MIT — see [LICENSE](https://github.com/Mattral/rssmlite/blob/main/LICENSE).
