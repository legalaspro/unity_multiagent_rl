import torch
import torch.nn as nn
from networks.utlis.weight_init import init_weights

from gymnasium import spaces

class SingleQNet(nn.Module):
    """
    Single Q Network for continuous and discrete action space. 
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
        super(SingleQNet, self).__init__()
       
        # Build the network - concatenate state and action at the first layer
        layers = []
        in_dim = state_size + action_size
        for i in range(len(hidden_sizes)):
            layers += [
                nn.Linear(in_dim, hidden_sizes[i]),
                nn.ReLU(),
                nn.LayerNorm(hidden_sizes[i])
            ]
            in_dim = hidden_sizes[i]
        layers += [nn.Linear(in_dim, 1)]
        self.fc = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(lambda m: init_weights(m, gain=nn.init.calculate_gain('relu')))
        init_weights(self.fc[-1], gain=gain, final_layer=True)

        self.to(device)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values"""
        # Concatenate state and action at the first layer
        x = torch.cat((state, action), dim=-1)
        q_values = self.fc(x)
        return q_values # Output is Q-value 
