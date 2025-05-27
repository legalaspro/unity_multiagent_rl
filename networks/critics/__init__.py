"""
Critic network implementations for various value function types.
"""
from .single_q_net import SingleQNet
from .twin_q_net import TwinQNet
from .v_net import VNet

__all__ = [
    'SingleQNet',
    'TwinQNet',
    'VNet'
]