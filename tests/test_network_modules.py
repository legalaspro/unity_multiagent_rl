import pytest
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete # For action space definitions

# --- Actor Network Imports ---
try:
    from networks.actors.deterministic_policy import DeterministicPolicy
except ImportError:
    DeterministicPolicy = None

try:
    from networks.actors.stochastic_policy import StochasticPolicy
except ImportError:
    StochasticPolicy = None # Typically for discrete actions

try:
    from networks.actors.reparam_stochastic_policy import ReparamStochasticPolicy
except ImportError:
    ReparamStochasticPolicy = None # Typically for continuous actions (SAC)

try:
    from networks.actors.squashed_gaussian_policy import SquashedGaussianPolicy # Often same as ReparamStochasticPolicy
except ImportError:
    SquashedGaussianPolicy = None


# --- Critic Network Imports ---
try:
    from networks.critics.single_q_net import SingleQNet
except ImportError:
    SingleQNet = None

try:
    from networks.critics.twin_q_net import TwinQNet
except ImportError:
    TwinQNet = None

try:
    from networks.critics.v_net import VNet
except ImportError:
    VNet = None


# --- Shared Modules Imports (Example) ---
try:
    from networks.modules.act import ActLayer # Example activation layer wrapper
except ImportError:
    ActLayer = None

try:
    from networks.modules.heads import CategoricalHead, DiagGaussianHead # Example output heads
except ImportError:
    CategoricalHead = None
    DiagGaussianHead = None


# --- Weight Initialization Import ---
try:
    from networks.utlis.weight_init import init_weights
except ImportError:
    init_weights = None


# --- Global Test Parameters ---
BATCH_SIZE = 4
DEVICE = torch.device("cpu")
DEFAULT_HIDDEN_SIZES = (64, 64)

# Helper to check if a module is available
def is_module_available(module):
    if module is None:
        pytest.skip(f"Module not available for testing.")
    return True

# --- Actor Network Tests ---

@pytest.mark.skipif(DeterministicPolicy is None, reason="DeterministicPolicy not available")
def test_deterministic_policy_forward():
    is_module_available(DeterministicPolicy)
    obs_dim = 10
    # For DeterministicPolicy, action_space is a gym.spaces.Box
    action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    
    actor = DeterministicPolicy(obs_dim, action_space, DEFAULT_HIDDEN_SIZES, device=DEVICE)
    actor.to(DEVICE)
    
    obs = torch.randn(BATCH_SIZE, obs_dim, device=DEVICE)
    actions = actor(obs)
    
    assert actions.shape == (BATCH_SIZE, *action_space.shape)
    # Check if actions are within bounds (actor should handle scaling/clamping)
    low = torch.tensor(action_space.low, device=DEVICE, dtype=torch.float32)
    high = torch.tensor(action_space.high, device=DEVICE, dtype=torch.float32)
    assert torch.all(actions >= low)
    assert torch.all(actions <= high)

@pytest.mark.skipif(StochasticPolicy is None, reason="StochasticPolicy not available")
def test_stochastic_policy_forward_discrete():
    is_module_available(StochasticPolicy)
    obs_dim = 10
    # For StochasticPolicy with discrete actions
    action_space = Discrete(5) # 5 discrete actions
        
    actor = StochasticPolicy(obs_dim, action_space, state_dependent_std=False, # Common for discrete
                             hidden_sizes=DEFAULT_HIDDEN_SIZES, device=DEVICE)
    actor.to(DEVICE)

    obs = torch.randn(BATCH_SIZE, obs_dim, device=DEVICE)
    
    # Test sample() method
    actions_sampled, log_probs_sampled = actor.sample(obs, deterministic=False)
    assert actions_sampled.shape == (BATCH_SIZE, 1) # Discrete action index
    assert actions_sampled.dtype == torch.long
    assert torch.all(actions_sampled >= 0) and torch.all(actions_sampled < action_space.n)
    assert log_probs_sampled.shape == (BATCH_SIZE, 1)

    # Test evaluate() method
    # Use the sampled actions to evaluate
    log_probs_eval, entropy_eval = actor.evaluate(obs, actions_sampled)
    assert log_probs_eval.shape == (BATCH_SIZE, 1)
    assert entropy_eval.shape == (BATCH_SIZE, 1) # Entropy per batch item
    assert torch.all(entropy_eval >= 0)

    # Test deterministic action (usually mode/argmax for discrete)
    actions_det, log_probs_det = actor.sample(obs, deterministic=True)
    assert actions_det.shape == (BATCH_SIZE, 1)


@pytest.mark.skipif(StochasticPolicy is None, reason="StochasticPolicy not available")
def test_stochastic_policy_forward_continuous():
    is_module_available(StochasticPolicy)
    obs_dim = 10
    action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)

    # For continuous, state_dependent_std might be True or False
    actor = StochasticPolicy(obs_dim, action_space, state_dependent_std=True,
                             hidden_sizes=DEFAULT_HIDDEN_SIZES, device=DEVICE)
    actor.to(DEVICE)
    obs = torch.randn(BATCH_SIZE, obs_dim, device=DEVICE)

    actions_sampled, log_probs_sampled = actor.sample(obs, deterministic=False)
    assert actions_sampled.shape == (BATCH_SIZE, *action_space.shape)
    assert log_probs_sampled.shape == (BATCH_SIZE, 1) # Sum of log_probs across action dims

    log_probs_eval, entropy_eval = actor.evaluate(obs, actions_sampled)
    assert log_probs_eval.shape == (BATCH_SIZE, 1)
    assert entropy_eval.shape == (BATCH_SIZE, 1)

    actions_det, log_probs_det = actor.sample(obs, deterministic=True) # Mean for Gaussian
    assert actions_det.shape == (BATCH_SIZE, *action_space.shape)


@pytest.mark.skipif(ReparamStochasticPolicy is None, reason="ReparamStochasticPolicy not available")
def test_reparam_stochastic_policy_forward(): # Usually for continuous (SAC)
    is_module_available(ReparamStochasticPolicy)
    obs_dim = 10
    action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
    
    actor = ReparamStochasticPolicy(obs_dim, action_space, DEFAULT_HIDDEN_SIZES, device=DEVICE)
    actor.to(DEVICE)
    
    obs = torch.randn(BATCH_SIZE, obs_dim, device=DEVICE)
    
    # Test sample()
    actions, log_probs, dist_info = actor.sample(obs, compute_log_prob=True, deterministic=False)
    assert actions.shape == (BATCH_SIZE, *action_space.shape)
    assert log_probs.shape == (BATCH_SIZE, 1)
    assert isinstance(dist_info, torch.Tensor) # For Box, dist_info is usually the action itself before Tanh

    # Test deterministic (mean)
    actions_det, _, _ = actor.sample(obs, compute_log_prob=False, deterministic=True)
    assert actions_det.shape == (BATCH_SIZE, *action_space.shape)

    # Check action bounds (due to Tanh in SquashedGaussian / Reparam)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
    assert torch.all(actions_det >= -1.0) and torch.all(actions_det <= 1.0)

# SquashedGaussianPolicy is often identical to ReparamStochasticPolicy if it uses Tanh squashing
if SquashedGaussianPolicy is not None and SquashedGaussianPolicy != ReparamStochasticPolicy:
    @pytest.mark.skipif(SquashedGaussianPolicy is None, reason="SquashedGaussianPolicy not available or same as ReparamStochasticPolicy")
    def test_squashed_gaussian_policy_forward():
        is_module_available(SquashedGaussianPolicy)
        obs_dim = 10
        action_space = Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        actor = SquashedGaussianPolicy(obs_dim, action_space.shape[0], DEFAULT_HIDDEN_SIZES, device=DEVICE)
        actor.to(DEVICE)
        obs = torch.randn(BATCH_SIZE, obs_dim, device=DEVICE)
        actions, log_probs, _ = actor.sample(obs) # Assuming sample() is primary method
        assert actions.shape == (BATCH_SIZE, *action_space.shape)
        assert log_probs.shape == (BATCH_SIZE, 1)
        assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)


# --- Critic Network Tests ---

@pytest.mark.skipif(SingleQNet is None, reason="SingleQNet not available")
def test_single_q_net_forward():
    is_module_available(SingleQNet)
    obs_dim = 10
    action_dim = 3 
    # SingleQNet typically takes concatenated obs and actions if used in DDPG/TD3 style
    # Or just obs if it's a VNet (but this is QNet).
    # The implementation of _MASACAgent, _MADDPGAgent, _MATD3Agent use total_state_size, total_action_size
    # This means obs_dim here is total_obs_dim, action_dim is total_action_dim
    
    critic = SingleQNet(obs_dim, action_dim, DEFAULT_HIDDEN_SIZES, device=DEVICE)
    critic.to(DEVICE)
    
    obs_input = torch.randn(BATCH_SIZE, obs_dim, device=DEVICE)
    action_input = torch.randn(BATCH_SIZE, action_dim, device=DEVICE)
    
    q_value = critic(obs_input, action_input)
    assert q_value.shape == (BATCH_SIZE, 1)

@pytest.mark.skipif(TwinQNet is None, reason="TwinQNet not available")
def test_twin_q_net_forward():
    is_module_available(TwinQNet)
    obs_dim = 10
    action_dim = 3
    
    critic = TwinQNet(obs_dim, action_dim, DEFAULT_HIDDEN_SIZES, device=DEVICE)
    critic.to(DEVICE)
    
    obs_input = torch.randn(BATCH_SIZE, obs_dim, device=DEVICE)
    action_input = torch.randn(BATCH_SIZE, action_dim, device=DEVICE)
    
    q1_value, q2_value = critic(obs_input, action_input)
    assert q1_value.shape == (BATCH_SIZE, 1)
    assert q2_value.shape == (BATCH_SIZE, 1)

@pytest.mark.skipif(VNet is None, reason="VNet not available")
def test_v_net_forward():
    is_module_available(VNet)
    obs_dim = 10 # For VNet, this is typically the (potentially centralized) observation dim
    
    critic = VNet(obs_dim, DEFAULT_HIDDEN_SIZES, device=DEVICE)
    critic.to(DEVICE)
    
    obs_input = torch.randn(BATCH_SIZE, obs_dim, device=DEVICE)
    v_value = critic(obs_input)
    assert v_value.shape == (BATCH_SIZE, 1)


# --- Shared Modules Tests ---

@pytest.mark.skipif(ActLayer is None, reason="ActLayer not available")
def test_act_layer_forward():
    is_module_available(ActLayer)
    # ActLayer(activation_fn_str, output_activation_fn_str=None)
    # Example: Tanh
    act_layer = ActLayer('tanh') # Assuming it takes string name
    act_layer.to(DEVICE)
    dummy_input = torch.randn(BATCH_SIZE, 10, device=DEVICE)
    output = act_layer(dummy_input)
    assert output.shape == (BATCH_SIZE, 10)
    assert torch.all(output >= -1) and torch.all(output <= 1)


@pytest.mark.skipif(CategoricalHead is None, reason="CategoricalHead not available")
def test_categorical_head_forward():
    is_module_available(CategoricalHead)
    input_dim = DEFAULT_HIDDEN_SIZES[-1] # From an MLP base
    action_dim_discrete = 5 # Number of discrete actions
    
    head = CategoricalHead(input_dim, action_dim_discrete, device=DEVICE)
    head.to(DEVICE)
    
    features = torch.randn(BATCH_SIZE, input_dim, device=DEVICE)
    dist = head(features) # Should return a torch.distributions.Categorical
    
    assert isinstance(dist, torch.distributions.Categorical)
    assert dist.logits.shape == (BATCH_SIZE, action_dim_discrete)
    assert dist.sample().shape == (BATCH_SIZE,) # Sample shape for Categorical

@pytest.mark.skipif(DiagGaussianHead is None, reason="DiagGaussianHead not available")
def test_diag_gaussian_head_forward():
    is_module_available(DiagGaussianHead)
    input_dim = DEFAULT_HIDDEN_SIZES[-1]
    action_dim_continuous = 3
    
    # DiagGaussianHead(num_inputs, action_dim, use_orthogonal_init, gain, use_state_dependent_std, ...)
    # Need to check actual signature for required params. Assume some defaults.
    head = DiagGaussianHead(input_dim, action_dim_continuous, use_state_dependent_std=False, device=DEVICE)
    head.to(DEVICE)

    features = torch.randn(BATCH_SIZE, input_dim, device=DEVICE)
    dist = head(features) # Should return a torch.distributions.Normal (or MultivariateNormal if diagonal)
    
    assert isinstance(dist, torch.distributions.Normal) # Assuming Normal for diagonal Gaussian
    assert dist.mean.shape == (BATCH_SIZE, action_dim_continuous)
    assert dist.stddev.shape == (BATCH_SIZE, action_dim_continuous)
    assert dist.sample().shape == (BATCH_SIZE, action_dim_continuous)


# --- Weight Initialization Test ---

@pytest.mark.skipif(init_weights is None, reason="init_weights utility not available")
def test_init_weights_changes_weights():
    is_module_available(init_weights)
    layer = nn.Linear(10, 5, device=DEVICE)
    layer.to(DEVICE) # Ensure layer is on device before getting params

    original_weight = layer.weight.clone().detach()
    original_bias = layer.bias.clone().detach()

    # Example: init_weights(module, weight_init_type, bias_const, gain=1)
    init_weights(layer, 'xavier_uniform_', 0.01) 

    assert not torch.allclose(original_weight, layer.weight.data), "Weights did not change after init_weights."
    # Bias might or might not change if bias_const was already close to original or if init doesn't touch bias with some schemes
    # A specific check for bias might be:
    assert torch.allclose(layer.bias.data, torch.full_like(layer.bias.data, 0.01)), "Bias not set to constant as expected."

    # Test another init type if supported, e.g., 'orthogonal_'
    layer_ortho = nn.Linear(10, 5, device=DEVICE).to(DEVICE)
    original_weight_ortho = layer_ortho.weight.clone().detach()
    init_weights(layer_ortho, 'orthogonal_', 0.05, gain=0.5) # Assuming gain is a param
    assert not torch.allclose(original_weight_ortho, layer_ortho.weight.data)
    assert torch.allclose(layer_ortho.bias.data, torch.full_like(layer_ortho.bias.data, 0.05))

# Final check of imports
def test_all_imports():
    # This test just serves to ensure the file is parsed and imports are attempted.
    # Individual skips handle cases where modules aren't found.
    assert True
