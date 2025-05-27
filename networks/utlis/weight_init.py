import torch
import torch.nn as nn


def init_weights(module, gain=1.0, final_layer=False):
    """Initialize network weights using standard PyTorch initialization
    
    Args:
        module: PyTorch module to initialize
        gain: Gain parameter for initialization
        final_layer: Whether this is the final 
    """
    if isinstance(module, nn.Linear):
        if final_layer:  # Output layer
            # Use uniform initialization for the final layer
            nn.init.uniform_(module.weight, -gain, gain)
            nn.init.uniform_(module.bias, -gain, gain)
        else:  # Hidden layers
            # Use Orthogonal fan-in initialization for ReLU layers
            nn.init.orthogonal_(module.weight, gain=gain)
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        # Initialize LayerNorm weights to 1 and biases to 0
        if module.weight is not None: # LayerNorm might not have affine params
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
