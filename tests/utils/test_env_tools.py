import pytest
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary

# Functions to be tested
from utils.env_tools import (
    get_shape_from_obs_space,
    get_shape_from_act_space,
    get_action_dim_for_critic_input
)

# --- Tests for get_shape_from_obs_space ---

def test_get_shape_from_obs_space_box():
    obs_space = Box(low=0, high=1, shape=(3, 4), dtype=np.float32)
    assert get_shape_from_obs_space(obs_space) == (3, 4)

    obs_space_flat = Box(low=0, high=1, shape=(5,), dtype=np.float32)
    assert get_shape_from_obs_space(obs_space_flat) == (5,)

def test_get_shape_from_obs_space_list():
    # The function's current implementation for list is:
    # elif obs_space.__class__.__name__ == "list": obs_shape = obs_space
    # This means it expects the list itself to be the shape.
    obs_shape_list = [(3,), (4,4)] # This seems like an unusual input type for "shape"
    assert get_shape_from_obs_space(obs_shape_list) == obs_shape_list

    obs_shape_tuple_in_list = (5,2) # A raw tuple representing a shape
    # If you pass a tuple directly, it won't hit the "list" branch.
    # To test the list branch as written, you must pass a list.
    # A list of numbers representing a shape e.g. [5,2]
    assert get_shape_from_obs_space([5,2]) == [5,2]


def test_get_shape_from_obs_space_unsupported():
    obs_space = Discrete(5) # Not a Box or list
    with pytest.raises(NotImplementedError):
        get_shape_from_obs_space(obs_space)

# --- Tests for get_shape_from_act_space ---
# This function returns an int (usually first dimension or 1 for Discrete)

def test_get_shape_from_act_space_discrete():
    act_space = Discrete(5)
    assert get_shape_from_act_space(act_space) == 1

def test_get_shape_from_act_space_multidiscrete():
    act_space = MultiDiscrete([2, 3, 4]) # shape is (3,)
    assert get_shape_from_act_space(act_space) == 3 # Returns shape[0]

def test_get_shape_from_act_space_box():
    act_space = Box(low=-1, high=1, shape=(3, 4), dtype=np.float32)
    assert get_shape_from_act_space(act_space) == 3 # Returns shape[0]

    act_space_flat = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
    assert get_shape_from_act_space(act_space_flat) == 5

def test_get_shape_from_act_space_multibinary():
    act_space = MultiBinary(5) # shape is (5,)
    assert get_shape_from_act_space(act_space) == 5 # Returns shape[0]

# --- Tests for get_action_dim_for_critic_input ---
# This function returns an int representing the dimension for critic input

def test_get_action_dim_for_critic_input_discrete():
    act_space = Discrete(5)
    assert get_action_dim_for_critic_input(act_space) == 5 # Returns .n (for one-hot usually)

def test_get_action_dim_for_critic_input_multidiscrete():
    act_space = MultiDiscrete([2, 3, 4]) # nvec = [2,3,4]
    assert get_action_dim_for_critic_input(act_space) == 9 # Sum of nvec

def test_get_action_dim_for_critic_input_box():
    act_space = Box(low=-1, high=1, shape=(3, 4), dtype=np.float32)
    assert get_action_dim_for_critic_input(act_space) == 3 # Returns shape[0]

    act_space_flat = Box(low=-1, high=1, shape=(5,), dtype=np.float32)
    assert get_action_dim_for_critic_input(act_space_flat) == 5

def test_get_action_dim_for_critic_input_multibinary():
    act_space = MultiBinary(5) # shape is (5,)
    assert get_action_dim_for_critic_input(act_space) == 5 # Returns shape[0]

# Note: `get_flattened_obs_dim`, `get_action_scale_bias`, `RecordEpisodeStatistics`, 
# and `create_env` were not found in the provided `utils/env_tools.py` content,
# so tests for them are not implemented here.
# If they exist elsewhere or are added, tests would be needed.
# The current tests cover all functions found in the provided `utils/env_tools.py`.
