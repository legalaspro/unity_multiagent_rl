import torch
import torch.nn as nn
from networks.utlis.weight_init import init_weights

class TwinQNet(nn.Module):
    """
    Twin Q Network for continuous and discrete action space. 
    Outputs the q value given global states and actions.
    """
    def __init__(self, state_size, action_size, hidden_sizes=[64, 64], gain=3e-3, device=torch.device("cpu")):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_sizes (tuple): Sizes of hidden layers
            gain (float): Final layer weight initialization
            device (torch.device): Device to use for training
        """
        super(TwinQNet, self).__init__()

        self._critic1 = self._build_critic_network(state_size, action_size, hidden_sizes)
        self._critic2 = self._build_critic_network(state_size, action_size, hidden_sizes)

        self.apply(lambda m: init_weights(m, gain=nn.init.calculate_gain('relu')))
        init_weights(self._critic1[-1], gain=gain, final_layer=True)
        init_weights(self._critic2[-1], gain=gain, final_layer=True)

        self.to(device)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> 2 Q-values."""
        x = torch.cat((state, action), dim=-1)
        q1 = self._critic1(x)
        q2 = self._critic2(x)
        return q1, q2 # Output is 2 Q-values

    def _build_critic_network(self, state_size, action_size, hidden_sizes):
        """Helper method to build a critic network with given architecture"""
        layers = []
        in_dim = state_size + action_size
        for i in range(len(hidden_sizes)):
            layers += [
                nn.Linear(in_dim, hidden_sizes[i]),
                nn.ReLU()
            ]
            in_dim = hidden_sizes[i]
        layers += [nn.Linear(in_dim, 1)]
        return nn.Sequential(*layers)
    
