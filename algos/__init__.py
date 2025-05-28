"""
Multi-Agent Reinforcement Learning Algorithms
"""
from .maddpg import MADDPG
from .mappo import MAPPO
from .masac import MASAC
from .matd3 import MATD3

__all__ = [
    'MADDPG',
    'MAPPO',
    'MASAC',
    'MATD3'
]

ALGO_REGISTRY = {
    'maddpg': MADDPG,
    'mappo': MAPPO,
    'masac': MASAC,
    'matd3': MATD3
}