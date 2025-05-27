import math
import torch
import torch.nn as nn
from networks.utlis.weight_init import init_weights


class DeterministicPolicy(nn.Module):
    """Deterministic policy network for continuous action space."""

    def __init__(self, state_size, action_space, hidden_sizes=[64, 64], gain=3e-3, 
                 device=torch.device("cpu")):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_space (gymnasium.spaces.Space): Action space of the environment.
            hidden_sizes (List[int]): Sizes of hidden layers.
            gain (float): Final layer weight initialization.
            device (torch.device): Device to use for training.
        """
        super().__init__()
        self.device = device
        self.action_space = action_space
        action_dim = math.prod(action_space.shape)

        # Store action bounds
        low, high = map(
            lambda x: torch.as_tensor(x, dtype=torch.float32).reshape(1, -1), # Ensure shape (1, action_dim)
            (action_space.low, action_space.high)
        )
        self.register_buffer("action_scale", (high - low) / 2)
        self.register_buffer("action_bias", (high + low) / 2)
        
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
        layers += [nn.Linear(in_dim, action_dim), nn.Tanh()]
        self.fc = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(lambda module: init_weights(module, gain=nn.init.calculate_gain('relu')))
        init_weights(self.fc[-2], gain=gain, final_layer=True)

        self.to(device)
        
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions"""
        x = self.fc(state)  # Output is in range [-1, 1]
        
        # Scale from [-1, 1] to [action_low, action_high]
        return x * self.action_scale + self.action_bias

            

