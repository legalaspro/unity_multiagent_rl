import torch
import math
import torch.nn as nn
from torch.distributions import (
    Normal,
    TransformedDistribution,
    TanhTransform,
)

class TanhNormal(TransformedDistribution):
    """Diagonal Gaussian followed by tanh squashing.

    It behaves like the distribution used in the original SAC paper:
    * supports `rsample()` (re-parameterised),
    * provides a stable `log_prob()` **summed over the last dim**,
    * exposes a convenient `.mode()` method (`tanh(µ)`).

    Parameters
    ----------
    loc     : torch.Tensor
        Mean of the underlying Normal.
    scale   : torch.Tensor
        Std-dev of the underlying Normal (must be positive).
    eps     : float, optional
        Clamp tolerance when evaluating `log_prob()`; default ``1e-6``.
    """
    
    has_rsample: bool = True

    def __init__(self, loc, scale, eps: float = 1e-6, cache_size: int = 1):
        self.eps = eps
        base = Normal(loc, scale)
        super().__init__(base, [TanhTransform(cache_size=cache_size)])
    
    # Note: In PyTorch 1.7+, TanhTransform itself has a .mean property
    # that computes E[tanh(X)] via Gauss-Hermite quadrature if base_dist is Normal.
    # However, SAC typically uses tanh(mean(X)) as the deterministic action.
    @property
    def mode(self): 
        """Most probable action after squashing (tanh(μ))."""
        return torch.tanh(self.base_dist.mean)
    
    @property
    def mean(self):
        return self.mode

    def log_prob(self, value):
        """
        Same as the parent implementation, but clamps inputs to
        ``(-1+eps, 1-eps)`` *and* returns a single scalar per sample
        (sum over the last dimension).
        """
        # clamp to avoid atanh(±1)
        clipped = torch.clamp(value, -1 + self.eps, 1 - self.eps)
        lp = super().log_prob(clipped) # shape (..., dim)
        return lp.sum(-1, keepdim=True) # shape (..., 1)
    
    def entropy(self):
        # This is an approximation. Exact entropy of TanhNormal is complex.
        # For PPO, often the entropy of the base Normal distribution is used.
        return self.base_dist.entropy().sum(-1, keepdim=True)