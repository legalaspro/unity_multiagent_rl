

import torch
import torch.nn as nn

from networks.utlis.weight_init import init_weights

class VNet(nn.Module):
    """
    Value Network, outputs the value given global states.
    """
    def __init__(
            self, 
            state_size,
            hidden_sizes=[64,64],
            gain=3e-3,
            device=torch.device("cpu")):
        """
        Initialize parameters and build model.

        Args:
            state_size (int): Dimension of the input.
            hidden_size (array): Hidden size of the network.
            device (torch.device): Device to use for training.
        """
        super().__init__()


        # Build the network
        layers = []
        in_dim = state_size
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
        self.apply(lambda module: init_weights(
            module, gain=nn.init.calculate_gain('relu')))
        init_weights(self.fc[-1], gain=gain, final_layer=True)
        
        self.to(device)
        
    def forward(self, x):
        """Forward pass of the critic network.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            value (torch.Tensor): Value.
        """
        value = self.fc(x)

        return value
    
