import torch

class RandomPolicy:
    """
    Random policy.
    """
    def __init__(self, action_spaces):
        self.action_spaces = action_spaces
        self.device = torch.device("cpu")          # keeps interface uniform
        self.num_agents = len(action_spaces)

    @torch.no_grad()
    def act(self, _obs, deterministic=True):
        return [torch.as_tensor([sp.sample()]) for sp in self.action_spaces]

    # stubs to keep snapshot() calls happy
    def policy_snapshot(self, *_):   
        return self
    def snapshot(self, *_):          
        return self
    def to(self, *_):                
        return self
    def cpu(self):   
        return self
    def cuda(self):  
        return self
    def set_eval_mode(self):         
        return self
    def set_train_mode(self):       
        return self