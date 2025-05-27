import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from gymnasium import spaces
from typing import Tuple, Union, List, Optional
from networks.modules.distributions import TanhNormal

def init_layer(m: nn.Module, gain: float = 1.0) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None: # Bias might not exist for all linear layers
            nn.init.zeros_(m.bias)

class ReparamActModule(nn.Module):
    """
    Universal reparameterizable action head for SAC-like policies,
    supporting Discrete (Gumbel-Softmax) and Box (Squashed Gaussian) action spaces.
    """
    def __init__(
        self,
        in_dim: int,
        action_space: spaces.Space,
        *,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        default_gumbel_tau: float = 1.0,
    ):
        """
        Initialize the Act Module

        Args:
            in_dim (int): Size of the *input feature vector* (last hidden layer of the policy).
            action_space (spaces.Space): Action space of the environment. 
                (can be Discrete, MultiDiscrete or Box)
            log_std_bounds (Tuple[float, float], optional): Clamp for continuous log-σ. Defaults to (-5.0, 2.0).
            default_gumbel_tau (float, optional): Default temperature τ for Gumbel-Softmax. Defaults to 1.0.
        """
        super(ReparamActModule, self).__init__()
        self.action_space = action_space
        self.log_std_min, self.log_std_max = log_std_bounds
        self.default_gumbel_tau = default_gumbel_tau

        if isinstance(action_space, spaces.Discrete):
            self.action_type = "Discrete"
            self.logits = nn.Linear(in_dim, action_space.n)
            init_layer(self.logits, gain=0.01)
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_type = "MultiDiscrete"
            self.num_components = len(action_space.nvec)
            self.logit_heads = nn.ModuleList()
            for num_choices in action_space.nvec:
                layer = nn.Linear(in_dim, num_choices)
                init_layer(layer, gain=0.01) # Small gain for output layers
                self.logit_heads.append(layer)
        elif isinstance(action_space, spaces.Box):
            self.action_type = "Box"

            action_dim = math.prod(action_space.shape)
            self.mean = nn.Linear(in_dim, action_dim)
            self.log_std = nn.Linear(in_dim, action_dim)

            init_layer(self.mean, gain=0.01) # Small gain for mean output
            init_layer(self.log_std, gain=1.0) # Larger gain for log_std, then squashed

            # store bounds for later rescaling
            low, high = map(
                lambda x: torch.as_tensor(x, dtype=torch.float32), 
                (action_space.low, action_space.high)
            )
            self.register_buffer("action_scale", (high - low) / 2)
            self.register_buffer("action_bias", (high + low) / 2)
        else:
            raise NotImplementedError(
                f"Action space type {type(action_space)} is not supported. "
                "Supported types are Discrete, MultiDiscrete, Box."
            )

    def forward(self, state: torch.Tensor) -> Union[torch.Tensor, List[torch.Tensor], TanhNormal]:
        """
        Computes action parameters or distribution from the input state.

        Args:
            state (torch.Tensor): The input state tensor (e.g., from policy backbone).

        Returns:
            Union[torch.Tensor, List[torch.Tensor], TanhNormal]:
            - For "Discrete": Logits tensor of shape (batch_size, num_actions).
            - For "MultiDiscrete": A list of logits tensors, one for each discrete component.
            - For "Box": A TanhNormal distribution instance.
        """
        if self.action_type == "Discrete":
            return self.logits(state) # (B, A)
        elif self.action_type == "MultiDiscrete":
            return [head(state) for head in self.logit_heads] # List of [batch_size, ad_i]
        elif self.action_type == "Box":
            mean = self.mean(state)
            # smooth log-σ in (log_std_min, log_std_max)
            log_std = torch.tanh(self.log_std(state).clamp(-10, 10))
            log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
            std = log_std.exp()
            return TanhNormal(mean, std)
        else:
            raise RuntimeError("Unsupported action type encountered in forward pass.")

    def _sample_discrete_action(
            self,
            logits: torch.Tensor,
            num_classes: int,
            deterministic: bool,
            compute_log_prob: bool,
            gumbel_tau: float,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Helper to sample from a single discrete action distribution.
        Args:
            logits (torch.Tensor): Logits tensor of shape (batch_size, num_actions).
            num_classes (int): Number of classes (actions) in the distribution.
            deterministic (bool): If True, sample deterministically (argmax).
            compute_log_prob (bool): If True, compute and return the log-probability.
            gumbel_tau (float): Temperature for Gumbel-Softmax.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
            - action_indices: (batch_size, 1) tensor of chosen action indices.
            - log_prob: (batch_size, 1) tensor of log-probabilities, or None.
            - one_hot_actions: (batch_size, num_classes) tensor of one-hot encoded actions.
        """
        log_probs_all_actions = F.log_softmax(logits, dim=-1)
        
        action_idx: torch.Tensor
        one_hot_action: torch.Tensor
        log_prob: Optional[torch.Tensor] = None

        if deterministic:
            action_idx = torch.argmax(logits, dim=-1) # (B, )
            one_hot_action = F.one_hot(action_idx, num_classes=num_classes).float()
            if compute_log_prob:
                # Log probability of the action chosen by argmax
                log_prob = torch.gather(log_probs_all_actions, -1, action_idx.unsqueeze(-1))
        else:
            # Gumbel-Softmax for reparameterized sampling (straight-through estimator)
            one_hot_action = F.gumbel_softmax(
                logits, tau=gumbel_tau, hard=True, dim=-1
            ) # (B, A)
            action_idx = one_hot_action.argmax(dim=-1) # (B, )
            if compute_log_prob:
                # Log probability of the Gumbel-Softmax sample
                # This is log P(action | logits) where action is the one-hot sample.
                log_prob = (one_hot_action * log_probs_all_actions).sum(dim=-1, keepdim=True)
        
        return action_idx.unsqueeze(-1), log_prob, one_hot_action

    def sample(
        self,
        state: torch.Tensor,
        *,
        deterministic: bool = False,
        compute_log_prob: bool = True,
        gumbel_tau: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Samples an action from the policy head, optionally computing its log probability.

        Args:
            state (torch.Tensor): The input state tensor.
            deterministic (bool, optional): If True, sample deterministically (argmax for discrete,
                                          mean for continuous). Defaults to False.
            compute_log_prob (bool, optional): If True, compute and return the log-probability
                                             of the sampled action. Defaults to True.
            gumbel_tau (float, optional): Temperature for Gumbel-Softmax (discrete only).
                                        If None, uses `default_gumbel_tau`.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
            - action: The sampled action.
                - Discrete: (batch_size, 1) tensor of action indices.
                - MultiDiscrete: (batch_size, num_components) tensor of action indices.
                - Box: (batch_size, action_dim) tensor of actions scaled to environment bounds.
            - log_prob: The log-probability of the action (summed across components/dimensions),
                        or None if `compute_log_prob` is False. Shape: (batch_size, 1).
            - dist_info: Additional distribution-specific information.
                - Discrete/MultiDiscrete: One-hot (or concatenated one-hot) representation
                                          of the action. Shape: (batch_size, num_categories_total).
                - Box: None.
        """
        current_gumbel_tau = (
            gumbel_tau 
            if gumbel_tau is not None 
            else self.default_gumbel_tau
        )

        if self.action_type == "Discrete":
            logits = self.forward(state) # Get logits: (B, A)
            action, log_prob, one_hot_action = self._sample_discrete_action(
                logits=logits,
                num_classes=self.action_space.n,
                deterministic=deterministic,
                compute_log_prob=compute_log_prob,
                gumbel_tau=current_gumbel_tau,
            )
            return action, log_prob, one_hot_action

        elif self.action_type == "MultiDiscrete":
            all_logits = self.forward(state) # # List of N tensors: (B, A_i)
            
            actions_list: List[torch.Tensor] = []
            one_hot_actions_list: List[torch.Tensor] = []
            log_probs_list: List[torch.Tensor] = [] # Only populated if compute_log_prob is True
            
            for i in range(self.num_components):
                logits_i = all_logits[i]
                action_i, log_prob_i, one_hot_action_i = self._sample_discrete_action(
                    logits=logits_i,
                    num_classes=self.action_space.nvec[i],
                    deterministic=deterministic,
                    compute_log_prob=compute_log_prob,
                    gumbel_tau=current_gumbel_tau,
                )
                actions_list.append(action_i)
                one_hot_actions_list.append(one_hot_action_i)
                if log_prob_i is not None: # Will be None if compute_log_prob is False
                    log_probs_list.append(log_prob_i)
            
            # Combine actions and one-hot encodings
            action = torch.cat(actions_list, dim=-1) # (B, num_components)
            combined_one_hot = torch.cat(one_hot_actions_list, dim=-1) # (B, sum(A_i))

            total_log_prob: Optional[torch.Tensor] = None
            if compute_log_prob and log_probs_list:
                # Stack to (B, num_components, 1) then sum over components
                total_log_prob = torch.stack(log_probs_list, dim=1).sum(dim=1) # (B, 1)
            
            return action, total_log_prob, combined_one_hot

        elif self.action_type == "Box":
            dist: TanhNormal = self.forward(state)

            raw_action: torch.Tensor # Action in range (-1, 1)
            if deterministic:
                raw_action = dist.mean
            else:
                raw_action = dist.rsample()
            
            log_prob: Optional[torch.Tensor] = None
            if compute_log_prob:
                # TanhNormal.log_prob returns (B, D), sum over D for total log_prob
                log_prob = dist.log_prob(raw_action).sum(dim=-1, keepdim=True)
            
            # Rescale action from (-1, 1) to environment's bounds
            env_action = raw_action * self.action_scale + self.action_bias

            return env_action, log_prob, None
       
       