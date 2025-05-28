import copy, torch
from abc import ABC, abstractmethod
from typing import Iterable

class MultiAgentModule(ABC):
    """
    Minimal base-class that gives a multi-agent algorithm the same ergonomic
    helpers as nn.Module (`to`, `eval`, …) 
    Gives a multi-agent algorithm module-like helpers:
        • .to(), .cpu(), .cuda()
        • .set_train_mode(), .set_eval_mode()
        • .snapshot()
    """

    # -------- subordinate modules ----------------------------------------
    @property
    @abstractmethod
    def _modules(self) -> Iterable[torch.nn.Module]:
        """
        Return *all* PyTorch modules that should be moved / toggled.
        All torch modules (actors, critics, targets …)
        """
    @property
    def _actor_modules(self) -> Iterable[torch.nn.Module]:
        """
        Yield only the *actors* (modules required for acting in the env).
        By default we return every module; override in subclasses if you
        want a slimmer eval snapshot.
        """
        return self._modules

    # -------- device helpers ---------------------------------------------
    def to(self, device: str | torch.device):
        device = torch.device(device)
        for m in self._modules:
            if m is not None:
                m.to(device)
        self.device = device
        return self

    def cpu(self):   
        return self.to("cpu")
    def cuda(self):  
        return self.to("cuda")

    # -------- mode helpers (train / eval) --------------------------------
    def set_train_mode(self):
        for m in self._modules:
            if m is not None:
                m.train(True)
        return self

    def set_eval_mode(self): 
        for m in self._modules:
            if m is not None:
                m.train(False)
        return self
    
    eval_mode = set_eval_mode

    # -------- snapshot for evaluator -------------------------------------
    def snapshot(self, device="cpu"):
        """Deep-copy on the given device, set to eval mode."""
        return copy.deepcopy(self).to(device).set_eval_mode()
    
    def policy_snapshot(self, device="cpu"):
        """
        Fast eval copy: deep-copy only the ACTORS.
        Critics, targets, optimisers are dropped → much smaller / faster.
        """
        snap = copy.deepcopy(self)           # full copy first
        keep = set(self._actor_modules)      # actors we keep

        # Null-out everything that isn't an actor module
        for name, obj in vars(snap).items():
            if isinstance(obj, torch.nn.Module) and obj not in keep:
                setattr(snap, name, None)
        return snap.to(device).set_eval_mode()