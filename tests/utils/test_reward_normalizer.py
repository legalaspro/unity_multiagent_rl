import pytest
import numpy as np

# Assuming classes are directly in utils.reward_normalizer
from utils.reward_normalizer import StandardNormalizer, EMANormalizer, normalise_shared_reward

# --- Test Parameters ---
CLIP_VALUE = 5.0
EPS_VALUE = 1e-8
DECAY_VALUE = 0.99

# --- Tests for StandardNormalizer ---

def test_standard_normalizer_initialization():
    norm = StandardNormalizer(clip=CLIP_VALUE, eps=EPS_VALUE)
    assert norm.clip == CLIP_VALUE
    assert norm.eps == EPS_VALUE
    assert norm._count == 0.0
    assert norm._mean == 0.0
    assert norm._var == 1.0
    assert norm._updating is True # Should default to training mode

def test_standard_normalizer_update_and_normalize():
    norm = StandardNormalizer(clip=CLIP_VALUE, eps=EPS_VALUE)
    
    # Test with scalar
    val1 = 10.0
    norm_val1 = norm.normalize(val1)
    assert norm._count == 1
    assert np.isclose(norm._mean, val1)
    assert np.isclose(norm._var, 0.0) # Variance of single point is 0
    # Normalized: (10 - 10) / (sqrt(0) + eps) approx 0
    assert np.isclose(norm_val1, 0.0, atol=1e-7)

    # Test with list
    val2_list = [10.0, 20.0] # Batch mean = 15, Batch var = ((10-15)^2 + (20-15)^2)/2 = (25+25)/2 = 25
    norm_val2 = norm.normalize(val2_list) 
    # After val1 (count=1, mean=10, var=0)
    # New batch: n_b=2, m_b=15, v_b=25
    # delta = 15 - 10 = 5
    # tot_n = 1 + 2 = 3
    # new_mean = 10 + 5 * 2 / 3 = 10 + 10/3 = 40/3 ~= 13.333
    # m2_tot = 0*1 + 25*2 + 5^2 * 1 * 2 / 3 = 50 + 50/3 = 200/3
    # new_var = (200/3) / 3 = 200/9 ~= 22.222
    assert norm._count == 3
    assert np.isclose(norm._mean, 40.0/3.0)
    assert np.isclose(norm._var, 200.0/9.0)
    
    # Normalized val2_list: ((10 - 40/3) / sqrt(200/9+eps)), ((20 - 40/3) / sqrt(200/9+eps))
    # ((-10/3) / (14.14/3)), ((20/3) / (14.14/3))
    # (-0.707), (1.414) approx with var=200/9
    expected_norm_val2_0 = (10.0 - (40.0/3.0)) / (np.sqrt(200.0/9.0) + EPS_VALUE)
    expected_norm_val2_1 = (20.0 - (40.0/3.0)) / (np.sqrt(200.0/9.0) + EPS_VALUE)
    assert np.isclose(norm_val2[0], expected_norm_val2_0)
    assert np.isclose(norm_val2[1], expected_norm_val2_1)

    # Test with numpy array
    val3_arr = np.array([10.0, 12.0, 14.0, 16.0, 18.0]) # Mean=14, Var=8
    norm.normalize(val3_arr) # Update internal stats
    # Previous: count=3, mean=13.333, var=22.222
    # New batch: n_b=5, m_b=14, v_b=8
    # ... (Welford updates)

def test_standard_normalizer_clip():
    norm = StandardNormalizer(clip=1.0)
    # Make mean 0, var 1 for easy testing of clip
    norm._mean = 0.0
    norm._var = 1.0
    norm._count = 10 # Dummy count > 0
    
    val_unclipped = 0.5
    assert np.isclose(norm.normalize(val_unclipped, update=False), 0.5)
    
    val_positive_clip = 10.0 # (10-0)/1 = 10, should be clipped to 1.0
    assert np.isclose(norm.normalize(val_positive_clip, update=False), 1.0)

    val_negative_clip = -10.0 # (-10-0)/1 = -10, should be clipped to -1.0
    assert np.isclose(norm.normalize(val_negative_clip, update=False), -1.0)

def test_standard_normalizer_train_eval_mode():
    norm = StandardNormalizer()
    norm.normalize(10.0) # mean=10, var=0, count=1
    
    norm.eval() # Switch to eval mode
    assert norm._updating is False
    mean_before = norm._mean
    var_before = norm._var
    count_before = norm._count
    
    norm.normalize(20.0) # This should not update stats
    assert norm._mean == mean_before
    assert norm._var == var_before
    assert norm._count == count_before
    
    norm.train() # Switch back to train mode
    assert norm._updating is True
    norm.normalize(20.0) # This should update stats
    assert norm._count > count_before


def test_standard_normalizer_reset():
    norm = StandardNormalizer()
    norm.normalize(10.0)
    norm.normalize(20.0)
    norm.reset()
    assert norm._count == 0.0
    assert norm._mean == 0.0
    assert norm._var == 1.0

def test_standard_normalizer_state_dict():
    norm = StandardNormalizer(clip=7.0, eps=1e-7)
    norm.normalize(np.array([1.0, 2.0, 3.0]))
    state = norm.state_dict()

    assert state["clip"] == 7.0
    assert state["eps"] == 1e-7
    assert state["count"] == norm._count
    assert state["mean"] == norm._mean
    assert state["var"] == norm._var

    new_norm = StandardNormalizer()
    new_norm.load_state_dict(state)
    assert new_norm.clip == 7.0
    assert new_norm.eps == 1e-7
    assert new_norm._count == norm._count
    assert new_norm._mean == norm._mean
    assert new_norm._var == norm._var


# --- Tests for EMANormalizer ---

def test_ema_normalizer_initialization():
    norm = EMANormalizer(decay=DECAY_VALUE, clip=CLIP_VALUE, eps=EPS_VALUE)
    assert norm.decay == DECAY_VALUE
    assert norm.clip == CLIP_VALUE
    assert norm.eps == EPS_VALUE
    assert np.isclose(norm._mean, 0.0)
    assert np.isclose(norm._var, 1.0)
    assert norm._updating is True

def test_ema_normalizer_update_and_normalize():
    norm = EMANormalizer(decay=0.9, clip=CLIP_VALUE) # Smaller decay for faster change
    
    # Initial state: mean=0, var=1
    val1 = 10.0
    norm.normalize(val1) # Update with scalar
    # m_b = 10, delta = 10 - 0 = 10
    # _mean = 0 + (1-0.9)*10 = 1
    # target_var = 10*10 + 0 (var of scalar) = 100
    # _var = 0.9*1 + (1-0.9)*100 = 0.9 + 10 = 10.9
    assert np.isclose(norm._mean, 1.0)
    assert np.isclose(norm._var, 10.9)

    val2_list = [10.0, 20.0] # m_b = 15, v_b = 25
    norm.normalize(val2_list)
    # Prev: mean=1, var=10.9
    # m_b = 15, delta = 15 - 1 = 14
    # _mean = 1 + (0.1)*14 = 1 + 1.4 = 2.4
    # target_var = 14*14 + 25 = 196 + 25 = 221
    # _var = 0.9*10.9 + (0.1)*221 = 9.81 + 22.1 = 31.91
    assert np.isclose(norm._mean, 2.4)
    assert np.isclose(norm._var, 31.91)


def test_ema_normalizer_clip():
    norm = EMANormalizer(clip=1.0)
    norm._mean = 0.0
    norm._var = 1.0 
    
    assert np.isclose(norm.normalize(0.5, update=False), 0.5)
    assert np.isclose(norm.normalize(10.0, update=False), 1.0)
    assert np.isclose(norm.normalize(-10.0, update=False), -1.0)


def test_ema_normalizer_train_eval_mode():
    norm = EMANormalizer(decay=0.9)
    norm.normalize(10.0) 
    mean_before = norm._mean
    var_before = norm._var
    
    norm.eval()
    norm.normalize(20.0)
    assert norm._mean == mean_before
    assert norm._var == var_before
    
    norm.train()
    norm.normalize(20.0)
    assert norm._mean != mean_before


def test_ema_normalizer_reset():
    norm = EMANormalizer()
    norm.normalize(10.0)
    norm.normalize(20.0)
    norm.reset()
    assert np.isclose(norm._mean, 0.0)
    assert np.isclose(norm._var, 1.0)

def test_ema_normalizer_state_dict():
    norm = EMANormalizer(decay=0.95, clip=8.0, eps=1e-6)
    norm.normalize(np.array([2.0, 4.0]))
    state = norm.state_dict()

    assert state["decay"] == 0.95
    assert state["clip"] == 8.0
    assert state["eps"] == 1e-6
    assert state["mean"] == norm._mean
    assert state["var"] == norm._var
    # 'count' is in base state dict, though not used by EMA update
    assert "count" in state 

    new_norm = EMANormalizer()
    new_norm.load_state_dict(state)
    assert new_norm.decay == 0.95
    assert new_norm.clip == 8.0
    assert new_norm.eps == 1e-6
    assert new_norm._mean == norm._mean
    assert new_norm._var == norm._var


# --- Tests for normalise_shared_reward ---

def test_normalise_shared_reward():
    rewards = np.array([
        [[10.0], [10.0]], # Env 0, Agent 0 gets 10, Agent 1 gets 10
        [[20.0], [20.0]], # Env 1, Agent 0 gets 20, Agent 1 gets 20
        [[30.0], [30.0]]  # Env 2, Agent 0 gets 30, Agent 1 gets 30
    ], dtype=np.float64) # Shape (3, 2, 1) -> (n_env, n_agents, 1)

    norm = StandardNormalizer() # Test with StandardNormalizer
    
    # First pass, update stats
    # r_env = [10, 20, 30], mean=20, var=((10-20)^2 + (20-20)^2 + (30-20)^2)/3 = (100+0+100)/3 = 200/3
    norm_rewards = normalise_shared_reward(rewards, norm)
    
    assert norm_rewards.shape == rewards.shape
    
    # Expected normalized r_env values:
    # (10-20)/sqrt(200/3) = -10 / 8.16 = -1.224
    # (20-20)/sqrt(200/3) = 0
    # (30-20)/sqrt(200/3) = 10 / 8.16 = 1.224
    expected_r_env_norm_0 = (10.0 - 20.0) / (np.sqrt(200.0/3.0) + norm.eps)
    expected_r_env_norm_1 = (20.0 - 20.0) / (np.sqrt(200.0/3.0) + norm.eps)
    expected_r_env_norm_2 = (30.0 - 20.0) / (np.sqrt(200.0/3.0) + norm.eps)

    assert np.allclose(norm_rewards[0, :, 0], expected_r_env_norm_0)
    assert np.allclose(norm_rewards[1, :, 0], expected_r_env_norm_1)
    assert np.allclose(norm_rewards[2, :, 0], expected_r_env_norm_2)

    # Second pass, stats should be more stable, norm should not update if eval mode
    norm.eval()
    mean_before = norm._mean
    norm_rewards_eval = normalise_shared_reward(np.array([[[15.0],[15.0]]]), norm)
    assert norm._mean == mean_before # Stats shouldn't change
    expected_r_env_norm_eval = (15.0 - mean_before) / (np.sqrt(norm._var) + norm.eps)
    assert np.allclose(norm_rewards_eval[0,:,0], expected_r_env_norm_eval)
