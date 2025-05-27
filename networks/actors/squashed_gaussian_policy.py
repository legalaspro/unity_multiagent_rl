import torch
import torch.nn as nn

from networks.utlis.weight_init import init_weights

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class SquashedGaussianPolicy(nn.Module):
    """Squashed Gaussian Policy Network for continuous action space."""
    
    def __init__(self, state_size, action_size, hidden_sizes=[64, 64],
                action_low=-1.0, action_high=1.0,  device=torch.device("cpu")):
        """
        Initialize parameters and build model.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_sizes (tuple): Sizes of hidden layers
            device (torch.device): Device to use for training
        """
        super(SquashedGaussianPolicy, self).__init__()

        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

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
        layers += [nn.Linear(in_dim, action_size * 2)]
        self.fc = nn.Sequential(*layers)

        self.apply(lambda m: init_weights(m, gain=nn.init.calculate_gain('relu')))
        # For Gaussian policies, use xavier/glorot initialization for better exploration
        nn.init.xavier_uniform_(self.fc[-1].weight, gain=1.0)
        nn.init.zeros_(self.fc[-1].bias)

        self.to(device)
    
    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.fc(state)
        mean, log_std = x.chunk(2, dim=-1)
        return mean, log_std
    
    def sample(self, state, compute_log_prob=True, deterministic=False):
        """Sample an action from the policy."""
        mean, log_std = self.forward(state)

        if deterministic:
             return torch.tanh(mean) * self.action_scale + self.action_bias, None
        
        # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX) # [1e-8, 1e2]
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = log_std.exp()

        base_distribution = torch.distributions.Normal(mean, std)
        # Make tanh to bound between [-1, 1] and scale to [action_low, action_high]
        tanh_transform = torch.distributions.transforms.TanhTransform(cache_size=1)
        scale_transform = torch.distributions.transforms.AffineTransform(loc=self.action_bias, scale=self.action_scale)
        squashed_dist = torch.distributions.TransformedDistribution(base_distribution, [tanh_transform, scale_transform])

        # Get squashed action
        action = squashed_dist.rsample()

        if not compute_log_prob:
            return action, None
        
        log_prob = squashed_dist.log_prob(action).sum(-1, keepdim=True) # [batch, 1]
        # this equal to 
        # log_prob = base_distribution.log_prob(arctanh_action)
        # log_prob -=  2*(np.log(2) - arctanh_action - F.softplus(-2 * arctanh_action))
        # log_prob -= torch.log(self.action_scale) 
        # log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob