import torch
import torch.nn as nn
from typing import Optional, List

from gymnasium import spaces
from networks.utlis.weight_init import init_weights
from networks.modules.reparam_act import ReparamActModule


LOG_STD_MAX: float = 2.0
LOG_STD_MIN: float = -5.0

class ReparamStochasticPolicy(nn.Module):
    """Reparameterizable Stochastic Policy Network. Outputs actions given states. """
    
    def __init__(self,
                 state_size: int,
                 action_space: spaces.Space,
                 hidden_sizes: List[int] = [256, 256],
                 gumbel_tau: float = 1.0, 
                 device: torch.device = torch.device("cpu")):
        """
       Initialize parameters and build model.

        Args:
            state_size (int): Dimension of each state.
            action_space (spaces.Space): Action space of the environment.
            hidden_sizes (List[int]): Sizes of hidden layers.
            gumbel_tau (float): Default temperature τ for Gumbel-Softmax.
            device (torch.device): Device to use for training.
        """
        super().__init__()
        self.device = device
        self.action_space = action_space
        
        # ------- Base MLP -------
        layers: List[nn.Module] = []
        in_dim = state_size
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim)) 
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.base = nn.Sequential(*layers)

        # ------ Action Head -------
        self.action_head = ReparamActModule( 
            in_dim=in_dim, 
            action_space=action_space,
            log_std_bounds=(LOG_STD_MIN, LOG_STD_MAX),
            default_gumbel_tau=gumbel_tau 
        )

        # Initialize weights
        self.base.apply(lambda m: init_weights(m, gain=nn.init.calculate_gain('relu')))

        self.to(device)
    
    def sample(
            self,
            state: torch.Tensor,
            *,
            deterministic: bool = False,
            compute_log_prob: bool = True,
            gumbel_tau: Optional[float] = None, 
            ):
        """
        Sample an action from the policy.
        """
        state_features = self.base(state)
        action, log_prob, dist_info = self.action_head.sample(
            state_features,
            deterministic=deterministic,
            compute_log_prob=compute_log_prob,
            gumbel_tau=gumbel_tau,
        )
        return action, log_prob, dist_info

