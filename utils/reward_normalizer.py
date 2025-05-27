from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Dict


# ────────────────────────────────────────────────────────────────
# Simple helpers – shared by all classes
# ────────────────────────────────────────────────────────────────
def _as_flat_array(x: Any):
    """Return (flat ndarray, is_scalar flag, original_shape)."""
    if np.isscalar(x):
        return np.asarray([x], dtype=np.float64), True, ()
    arr = np.asarray(x, dtype=np.float64)
    return arr.ravel(), False, arr.shape


def _reshape(flat: np.ndarray, is_scalar: bool, shape):
    return float(flat[0]) if is_scalar else flat.reshape(shape)


# ────────────────────────────────────────────────────────────────
# 1. Minimal base class
# ────────────────────────────────────────────────────────────────
class _BaseNormalizer(ABC):
    """Holds the public API; subclass supplies `_update_stats()`."""

    def __init__(self, clip: float = 10.0, eps: float = 1e-8) -> None:
        self.clip, self.eps = clip, eps
        self.train()          # default: stats update ON
        self.reset()

    # ---- public API -----------------------------------------------------
    def normalize(self, x, update: bool = True):
        """
        Scale *x* using the current running mean/var.

        Parameters
        ----------
        x       : scalar, list or np.ndarray of any shape.
        update  : override whether this call updates the stats
                  (default True, but ignored if `eval()` was called).
        """
        if not self._updating or not update:
            # fast path – just use stored stats
            return self._normalise_only(x)

        # update path
        flat, is_scalar, shape = _as_flat_array(x)
        self._update_stats(flat)                    # subclass hook
        return _reshape(self._apply(flat), is_scalar, shape)

    __call__ = normalize     # convenience

    def train(self): self._updating = True
    def eval(self):  self._updating = False

    def reset(self):
        self._count = 0.0    # Welford only; harmless for EMA
        self._mean  = 0.0
        self._var   = 1.0

    # ---- (de)serialisation ---------------------------------------------
    def state_dict(self) -> Dict[str, float]:
        return dict(count=self._count, mean=self._mean, var=self._var,
                    clip=self.clip, eps=self.eps)

    def load_state_dict(self, state: Dict[str, float]) -> None:
        self._count = state["count"]
        self._mean  = state["mean"]
        self._var   = state["var"]
        self.clip   = state["clip"]
        self.eps    = state["eps"]

    # ---- helpers used by subclasses ------------------------------------
    def _apply(self, flat: np.ndarray):
        """Return normalised version of *flat* (1‑D array)."""
        out = (flat - self._mean) / (np.sqrt(self._var) + self.eps)
        return np.clip(out, -self.clip, self.clip) if self.clip else out

    def _normalise_only(self, x):
        flat, is_scalar, shape = _as_flat_array(x)
        return _reshape(self._apply(flat), is_scalar, shape)

    @abstractmethod
    def _update_stats(self, batch: np.ndarray) -> None:
        """Update `_mean` and `_var` given a 1‑D NumPy array."""
        ...


# ────────────────────────────────────────────────────────────────
# 2. Concrete implementations
# ────────────────────────────────────────────────────────────────
class StandardNormalizer(_BaseNormalizer):
    """Unbiased Welford running mean / variance."""

    def _update_stats(self, batch: np.ndarray) -> None:
        n_b   = batch.size
        m_b   = batch.mean()
        v_b   = batch.var()

        delta = m_b - self._mean
        tot_n = self._count + n_b

        self._mean += delta * n_b / tot_n
        m2_tot = self._var * self._count + v_b * n_b + delta**2 * self._count * n_b / tot_n
        self._var  = m2_tot / tot_n
        self._count = tot_n


class EMANormalizer(_BaseNormalizer):
    """Cheaper exponential‑moving‑average alternative (slight bias)."""

    def __init__(self, decay: float = 0.9999, **kwargs):
        self.decay = float(decay)
        self._one_m = 1.0 - self.decay
        super().__init__(**kwargs)

    def _update_stats(self, batch: np.ndarray) -> None:
        m_b = batch.mean()
        delta = m_b - self._mean
        self._mean += self._one_m * delta

        target_var = delta * delta + batch.var()
        self._var  = self.decay * self._var + self._one_m * target_var

    # make `decay` checkpointable
    def state_dict(self):
        s = super().state_dict()
        s["decay"] = self.decay
        return s

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.decay = state.get("decay", self.decay)
        self._one_m = 1.0 - self.decay

# -───────────────────────────────────────────────────────────────
# 3. Helper function for runners
# -───────────────────────────────────────────────────────────────

def normalise_shared_reward(rew: np.ndarray, norm):
    """
    rew : (n_env, n_agents, 1) – identical values along the agent axis
    norm: StandardNormalizer or EMANormalizer
    """
    r_env  = rew[:, 0, 0]
    r_norm = norm.normalize(r_env)[:, None, None] #  # (n_env,1,1) add broadcast dims
    return np.broadcast_to(r_norm, rew.shape) # view, no copy