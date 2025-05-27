import math
import torch
import torch.nn as nn
import torch.nn.functional as F


from gymnasium import spaces
from typing import Tuple, Union, Optional
from torch.distributions import (
    Categorical
)
from networks.modules.distributions import TanhNormal

def init_layer(m: nn.Module, gain: float = 1.0) -> None:
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ActModule(nn.Module):
    """
    Universal action head for continuous and discrete action space.
    Suitable for algorithms like PPO or Discrete SAC (non-reparameterized policy).
    """
    def __init__(
        self,
        in_dim: int,
        action_space: spaces.Space,
        *,
        log_std_bounds: Tuple[float, float] = (-5.0, 2.0),
        state_dependent_std: bool = False,
    ):
        """
        Initialize the Act Module

        Args:
            in_dim (int): Size of the *input feature vector* (last hidden layer of the policy).
            action_space (spaces.Space): Action space of the environment. 
                (can be Discrete, MultiDiscrete or Box)
            log_std_bounds (Tuple[float, float], optional): Clamp for continuous log-σ. Defaults to (-5.0, 2.0).
            state_dependent_std (bool, optional): Whether to use state-dependent std. Defaults to True.
        """
        super().__init__()
        self.action_space = action_space
        self.log_std_min, self.log_std_max = log_std_bounds

        if isinstance(action_space, spaces.Discrete):
            self.action_type = "Discrete"
            self.logits = nn.Linear(in_dim, action_space.n)
            init_layer(self.logits, gain=0.01)
        elif isinstance(action_space, spaces.MultiDiscrete):
            self.action_type = "MultiDiscrete"
            self.logit_heads = nn.ModuleList()
            for n_choices in action_space.nvec:
                layer = nn.Linear(in_dim, n_choices)
                init_layer(layer, gain=0.01)
                self.logit_heads.append(layer)
        elif isinstance(action_space, spaces.Box):
            self.action_type = "Box"
            
            action_dim = math.prod(action_space.shape)
            self.mean = nn.Linear(in_dim, action_dim)
            init_layer(self.mean, gain=0.01) # Small gain for mean output
            if state_dependent_std:
                self.log_std = nn.Linear(in_dim, action_dim)
                init_layer(self.log_std, gain=1.0) # Larger gain for log_std, then squashed
            else:                                     # single learned parameter per dim
                self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))
                # self.log_std = nn.Parameter(torch.zeros(action_dim))

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
                "Supported: Discrete, MultiDiscrete, Box."
            )

    def _distribution(self, state_features: torch.Tensor) -> Union[Categorical,
        Tuple[Categorical, ...], TanhNormal]:
        """
        Get the action distribution for the given input features

        Args:
            state_features (torch.Tensor): Input tensor.
        """
        if self.action_type == "Discrete":
            logits = self.logits(state_features)
            return Categorical(logits=logits)
        elif self.action_type == "MultiDiscrete":
            dists = []
            for head in self.logit_heads:
                logits = head(state_features)
                dists.append(Categorical(logits=logits))
            return tuple(dists)
        elif self.action_type == "Box":
            # continuous
            mean = self.mean(state_features)
            # smooth log-σ in (log_std_min, log_std_max)
            if isinstance(self.log_std, nn.Linear):
                # clamp before tanh to avoid saturation/back-prop-dead
                pre_log_std = self.log_std(state_features).clamp(-10, 10)
                log_std = torch.tanh(pre_log_std)
                log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
            else:
                log_std = self.log_std.expand_as(mean)
                log_std = log_std.clamp(self.log_std_min, self.log_std_max)

            std = log_std.exp()
            return TanhNormal(mean, std)
        else:
            raise RuntimeError("Unsupported action type encountered in forward pass.")
    
    def sample(self,
        state: torch.Tensor,
        *,
        deterministic: bool = False
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Draw action(s) from the policy head.
            * `deterministic=True` → mode / argmax (for evaluation).

        Args:
            state (torch.Tensor): Input tensor.
            deterministic (bool, optional): Whether to use deterministic sampling. Defaults to False.

        Returns:
            action (torch.Tensor): Action tensor. # (B, A) or (B, num_components)
            log_prob (torch.Tensor): Log-probability of the action. # (B, 1)
        """
        distribution = self._distribution(state)
        
        # --- action sampling ---
        if deterministic:
            if self.action_type == "MultiDiscrete":
                action = torch.stack([dist.mode for dist in distribution], dim=-1)
            elif self.action_type == "Discrete":
                action = distribution.mode.unsqueeze(-1) # (B, 1)
            else: # Box
                action = distribution.mean # TanhNormal.mode is tanh(mean)
        else:
            if self.action_type == "MultiDiscrete":
                 action = torch.stack([dist.sample() for dist in distribution], dim=-1)
            elif self.action_type == "Discrete":
                action = distribution.sample().unsqueeze(-1) # (B, 1)
            else: # Box
                action = distribution.sample()
        
        # --- log-probability ---
        log_prob: Optional[torch.Tensor] = None
        if not deterministic:
            if self.action_type == "MultiDiscrete":
                # action shape: (B, num_components)
                component_actions = action.unbind(dim=-1) # List of (B, ) tensors
                log_prob = torch.stack(
                    [dist.log_prob(act_comp) for dist, act_comp in zip(distribution, component_actions)],
                    dim=-1
                ).sum(dim=-1, keepdim=True) # Sum log_probs for independent components
            elif self.action_type == "Discrete":
                # Categorical expects (B, ) tensor of action indices 
                log_prob = distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
            else: # Box
                # TanhNormal.log_prob(action) returns (B, 1) after summing internally
                log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        
        # --- post-processing ---
        if self.action_type == "Box":
            # Action from TanhNormal is in (-1, 1). Scale it.
            action = action * self.action_scale + self.action_bias

        return action, log_prob
    
    def evaluate(
            self, 
            state: torch.Tensor, 
            action: torch.Tensor
            ) -> Tuple[torch.Tensor, 
                       torch.Tensor, 
                       Union[Categorical, Tuple[Categorical, ...], TanhNormal]]:
        """
        Evaluate the log-probability of an action.
        For PPO.

        Args:
            state (torch.Tensor): Input tensor.
            action (torch.Tensor): Action tensor.

        Returns:
            log_prob (torch.Tensor): Log-probability of the action.
            entropy (torch.Tensor): Entropy of the action distribution.
            distribution (torch.distributions.Distribution): Action distribution.
        """
        
        if self.action_type == "Box":
            # Unscale action from environment bounds to (-1, 1)
            unscaled_action = (action - self.action_bias) / self.action_scale
        else:
            unscaled_action = action

        distribution = self._distribution(state)
        log_prob: torch.Tensor
        entropy: torch.Tensor

        if self.action_type == "MultiDiscrete":
            component_actions = unscaled_action.unbind(dim=-1) # List of (B, ) tensors
            log_probs_list = [
                dist.log_prob(act_comp) for dist, act_comp in zip(distribution, component_actions)
            ]
            log_prob = torch.stack(log_probs_list, dim=-1).sum(dim=-1, keepdim=True)
            entropies_list = [dist.entropy() for dist in distribution]
            entropy = torch.stack(entropies_list, dim=-1).sum(dim=-1, keepdim=True)
        elif self.action_type == "Discrete":
            # Squeeze action if it's (B,1) for Categorical's log_prob and then unsqueeze log_prob
            log_prob = distribution.log_prob(unscaled_action.squeeze(-1)).unsqueeze(-1)
            entropy = distribution.entropy().unsqueeze(-1) # Ensure (B,1)
        else: # Box
            log_prob = distribution.log_prob(unscaled_action) # TanhNormal takes action in (-1,1)
            entropy = distribution.entropy() # TanhNormal.entropy() returns (B,1)
     
        return log_prob, entropy, distribution
    
    def discrete_outputs(self, state_features: torch.Tensor
     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return logits, log‑probabilities, and probabilities for categorical heads.
        These tensors needed for the analytic expectation in discrete SAC.

        Args:
            Batched observation features, shape (B, feat_dim).

        Returns
        -------
        logits     : torch.Tensor  # (B, A)      or (B, N, A_i)
        log_probs  : torch.Tensor  # same shape
        probs      : torch.Tensor  # same shape (grad‑carrying πθ)

        Notes
        -----
        * No action is sampled; gradients will flow cleanly through
          `logits → softmax → log_softmax`.
        * For `MultiDiscrete`, the second axis `N` indexes each categorical factor.
        """
        if self.action_type == "Box":
            raise RuntimeError("discrete_outputs_direct() is only valid for "
                           "Discrete / MultiDiscrete action spaces.")

        # ---- Single Categorical ----
        if self.action_type == "Discrete":
            logits = self.logits(state_features)                    # (B, A)
            log_probs = F.log_softmax(logits, -1)
            probs = F.softmax(logits, dim=-1)
            return logits, log_probs, probs
        else:
            # ------------- Multi-Discrete ---------------------------------- #
            # (B, num_components, num_choices_per_component)
            all_logits_list = [head(state_features) for head in self.logit_heads]
            logits_stacked = torch.stack(all_logits_list, dim=1) # (B, N, A_i)
            log_probs = F.log_softmax(logits_stacked, dim=-1)
            probs = F.softmax(logits_stacked, dim=-1)
            return logits_stacked, log_probs, probs
    