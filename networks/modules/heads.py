"""NOTE: Just thought process of possible action heads with and without reparameterization."""
import math
import torch
import torch.nn as nn
from torch.distributions import Categorical, RelaxedOneHotCategorical

from typing import Sequence, Tuple
from gymnasium import spaces
from networks.modules.distributions import TanhNormal


class DiscreteHead(nn.Module):
    """Single Categorical or Gumbel-Softmax head for discrete action spaces."""
    def __init__(self, in_dim: int, num_actions: int, use_gumbel: bool, tau: float):
        super().__init__()
        self.use_gumbel = use_gumbel
        self.tau = tau
        self.logits = nn.Linear(in_dim, num_actions)

    def dist(self, h: torch.Tensor) -> torch.distributions.Distribution:
        """Return the action distribution for the given input."""
        if self.use_gumbel:
            return RelaxedOneHotCategorical(self.tau, logits=self.logits(h))
        return Categorical(logits=self.logits(h))

    def mode(self, h: torch.Tensor) -> torch.Tensor:
        """Return the index of the most likely action."""
        return self.logits(h).argmax(-1, keepdim=True)

class MultiDiscreteHead(nn.Module):
    """Multiple independent Categorical or Gumbel-Softmax heads."""
    def __init__(self, in_dim: int, num_actions: Sequence[int], use_gumbel: bool, tau: float):
        super().__init__()
        self.use_gumbel = use_gumbel
        self.tau = tau
        self.heads = nn.ModuleList([nn.Linear(in_dim, n) for n in num_actions])

    def dist(self, h: torch.Tensor) -> Tuple[torch.distributions.Distribution, ...]:
        """Return a tuple of action distributions for each head."""
        dist_class = RelaxedOneHotCategorical if self.use_gumbel else Categorical
        return tuple(dist_class(self.tau, logits=head(h)) for head in self.heads)

    def mode(self, h: torch.Tensor) -> torch.Tensor:
        """Return the indices of the most likely actions for each head."""
        return torch.stack([head(h).argmax(-1, keepdim=True) for head in self.heads], dim=-1)


class BoxHead(nn.Module):
    """Diagonal Gaussian head with tanh squashing for continuous action spaces."""
    def __init__(self, in_dim: int, shape: Tuple[int, ...], log_std_bounds: Tuple[float, float]):
        super().__init__()
        act_dim = math.prod(shape)
        self.mu = nn.Linear(in_dim, act_dim)
        self.log_std = nn.Linear(in_dim, act_dim)
        self.log_std_bounds = log_std_bounds

    def dist(self, h: torch.Tensor) -> TanhNormal:
        """Return a TanhNormal distribution for the given input."""
        mu = self.mu(h)
        # smooth log-σ in (log_std_min, log_std_max)
        log_std = torch.tanh(self.log_std(x))
        log_std = self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (log_std + 1)
        std = log_std.exp()
        return TanhNormal(mu, std)

    def mode(self, h: torch.Tensor) -> torch.Tensor:
        """Return the mean action (tanh-transformed)."""
        return torch.tanh(self.mu(h))


### Head Factory ###
def make_head(
    in_dim: int,
    space: spaces.Space,
    *,
    use_gumbel: bool = False,
    gumbel_tau: float = 1.0,
    log_std_bounds: Tuple[float, float] = (-20.0, 2.0),
) -> nn.Module:
    """Create an action head based on the action space."""
    if isinstance(space, spaces.Discrete):
        return DiscreteHead(in_dim, space.n, use_gumbel, gumbel_tau)
    if isinstance(space, spaces.MultiDiscrete):
        return MultiDiscreteHead(in_dim, space.nvec, use_gumbel, gumbel_tau)
    if isinstance(space, spaces.Box):
        return BoxHead(in_dim, space.shape, log_std_bounds)
    raise NotImplementedError(f"Unsupported action space: {space}")
