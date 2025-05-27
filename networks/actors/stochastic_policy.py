import torch
import torch.nn as nn
from typing import Optional, List, Tuple

from gymnasium import spaces
from networks.utlis.weight_init import init_weights
from networks.modules.act import ActModule


LOG_STD_MAX: float = 2.0
LOG_STD_MIN: float = -5.0

class StochasticPolicy(nn.Module):
    """
    Stochastic Policy Network. Outputs actions given states. 
    Suitable for PPO or Discrete SAC.
    """
    
    def __init__(
            self,
            state_size: int,
            action_space: spaces.Space,
            *,
            state_dependent_std: bool = False,
            hidden_sizes: List[int] = [256, 256], 
            device: torch.device = torch.device("cpu")):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_space (spaces.Space): Action space of the environment.
            hidden_sizes (List[int]): Sizes of hidden layers.
            device (torch.device): Device to use for training.
        """
        super().__init__()
        self.device = device
        self.action_space = action_space

        
        # ------- Base MLP -------
        layers: List[nn.Module] = []
        in_dim = state_size
        layers += [nn.LayerNorm(in_dim)]
        for i, hidden_dim in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
            in_dim = hidden_dim
        self.base = nn.Sequential(*layers)

        # ------ Action Head -------
        self.action_head = ActModule(
            in_dim=in_dim, 
            action_space=action_space,
            log_std_bounds=(LOG_STD_MIN, LOG_STD_MAX),
            state_dependent_std=state_dependent_std
        )

        # Initialize weights
        self.base.apply(lambda m: init_weights(m, gain=nn.init.calculate_gain('relu')))

        self.to(device)
    
    def forward(self, state:torch.Tensor, *, deterministic:bool=False):
        """
        Forward pass of the policy network.
        For PPO rollouts or getting actions for the environment.
        """
        state_features = self.base(state)
        action, log_prob = self.action_head.sample(state_features, deterministic=deterministic)
        return action, log_prob

    def evaluate(self, state:torch.Tensor, action:torch.Tensor):
        """
        Evaluate the log-probability of an action.
        For PPO loss calculation.
        """
        state_features = self.base(state)
        log_prob, entropy, _ = self.action_head.evaluate(state_features, action)
        return log_prob, entropy
    
    def discrete_outputs(self, state: torch.Tensor):
        """
        Return logits, log‑probabilities, and probabilities for categorical heads.
        Useful for some Discrete SAC variants.
        """
        x = self.base(state)
        return self.action_head.discrete_outputs(x)
