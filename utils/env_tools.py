import numpy as np

def get_shape_from_obs_space(obs_space) -> tuple:
    """Get shape from observation space.
    Args:
        obs_space: (gymnasium.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space) -> int:
    """Get shape from action space.
    Args:
        act_space: (gymnasium.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def get_action_dim_for_critic_input(act_space):
    """Get action dimension for critic input.
    Args:
        act_space: (gymnasium.spaces) action space
    Returns:
        act_dim: (int) action dimension
    """
    if act_space.__class__.__name__ == "Discrete":
        act_dim = act_space.n
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_dim = int(np.sum(act_space.nvec))
    elif act_space.__class__.__name__ == "Box":
        act_dim = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_dim = act_space.shape[0]
    return act_dim