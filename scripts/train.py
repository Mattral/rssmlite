#!/usr/bin/env python3
"""CLI training entry point (roadmap P1.3).

Usage:
    python scripts/train.py --config configs/cartpole.yaml
    python scripts/train.py --config configs/cartpole.yaml --resume ckpts/checkpoint_50000.pt
    python scripts/train.py --config configs/cartpole.yaml --checkpoint-dir /content/drive/MyDrive/rssmlite-ckpts

Only orchestrates: parse args, build/resume an agent, call `agent.train()`.
All actual training logic lives in `rssmlite.agent.RSSMAgent`.
"""

import argparse

import gymnasium as gym
import yaml

from rssmlite import RSSMAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config, e.g. configs/cartpole.yaml")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint .pt file to resume from")
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Where to save checkpoints. Point this at a mounted Google Drive path on Colab.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = gym.make(config["env"]["id"])

    if args.resume:
        print(f"Resuming from {args.resume}")
        agent = RSSMAgent.load_checkpoint(args.resume)
    else:
        agent = RSSMAgent.from_config(args.config, env=env)

    train_kwargs = dict(config.get("train", {}))
    train_kwargs.setdefault("max_episode_steps", config["env"].get("max_episode_steps", 500))

    agent.train(env, checkpoint_dir=args.checkpoint_dir, **train_kwargs)
    env.close()


if __name__ == "__main__":
    main()
