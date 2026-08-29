"""
rssmlite
========

A lightweight PyTorch library for RSSM-based world models (DreamerV3-style)
on Gymnasium environments, designed to run end-to-end on a free Colab T4
GPU with no external datasets.

Public API (grows as the roadmap progresses — see ROADMAP.md):
    RSSM         — the world model (P1.1, done)
    RSSMAgent    — RSSM + actor-critic trained in imagination (P1.3, TODO)
    ReplayBuffer — sequence storage/sampling (P1.2, TODO)

Project: https://github.com/Mattral/rssmlite
Author:  Min Htet Myet
"""

from rssmlite.rssm import RSSM

__version__ = "0.1.0.dev0"

__all__ = ["RSSM", "__version__"]
