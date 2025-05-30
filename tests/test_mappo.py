import pytest
import torch
import numpy as np
from argparse import Namespace
from gymnasium.spaces import Box, Discrete

from algos.mappo import MAPPO
from algos.agent._mappo_agent import _MAPPOAgent
from buffers.rollout_storage import RolloutStorage
from networks.actors.stochastic_policy import StochasticPolicy
from networks.critics.v_net import VNet


@pytest.fixture
def mappo_config_dict_discrete():
    """Basic configuration dictionary for MAPPO tests with Discrete actions."""
    config = {
        "num_agents": 2,
        "obs_space_shape_defs": [(10,), (10,)], # Homogeneous obs spaces for shared policy test later
        "action_type": 'Discrete',
        "discrete_action_n_per_agent": [5, 5], # Homogeneous action spaces
        "continuous_action_shape_defs": [(2,), (2,)], # Placeholder

        "device": 'cpu',
        "actor_lr": 3e-4, # Renamed from lr for clarity
        "critic_lr": 3e-4,
        "hidden_sizes": (64, 64), # Tuple for network layers
        "state_dependent_std": False, # For StochasticPolicy

        "n_rollout_threads": 2,
        "n_steps": 32, # Smaller for faster tests
        "ppo_epoch": 4, # PPO epochs
        "num_mini_batch": 2, # n_steps * n_rollout_threads / num_mini_batch = batch_size for PPO
        
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_param": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "use_max_grad_norm": True,
        
        "use_gae": True,
        "use_popart": False,
        "use_huber_loss": False, # For value loss
        "use_clipped_value_loss": True, # Common in PPO
        "use_value_active_masks": True,
        "use_policy_active_masks": True, # Usually refers to done masks for actor updates
        "use_feature_normalization": False, # For obs normalization
        
        "use_centralized_V": True,
        "share_actor": False, # Test separate actors
        "share_critic": True, # Test shared critic (common for MAPPO)

        "seed": 123,
        "total_steps": 2000000,
        "use_linear_lr_decay": False,
        "recurrent_N": 1, 
        "use_role_id": False, 
        "teams": None,
        "use_recurrent_policy": False, # For MLP tests
    }
    return config

# Helper to create spaces from config
def get_spaces_from_config_params(num_agents, obs_space_shapes, action_type, discrete_ns, continuous_shapes):
    obs_spaces = [Box(low=-np.inf, high=np.inf, shape=s, dtype=np.float32) for s in obs_space_shapes]
    if action_type == 'Discrete':
        act_spaces = [Discrete(n) for n in discrete_ns]
    else: # Continuous
        act_spaces = [Box(low=-1, high=1, shape=s, dtype=np.float32) for s in continuous_shapes]
    return obs_spaces, act_spaces

@pytest.fixture
def mappo_instance_discrete(mappo_config_dict_discrete):
    args = Namespace(**mappo_config_dict_discrete)
    obs_spaces, action_spaces = get_spaces_from_config_params(
        args.num_agents, args.obs_space_shape_defs, args.action_type,
        args.discrete_action_n_per_agent, args.continuous_action_shape_defs
    )
    mappo_algo = MAPPO(args, obs_spaces, action_spaces, device=torch.device(args.device))
    return mappo_algo

# --- Initialization Tests ---
def test_mappo_initialization_discrete_unshared_actor_shared_critic(mappo_instance_discrete, mappo_config_dict_discrete):
    args = Namespace(**mappo_config_dict_discrete)
    mappo = mappo_instance_discrete

    assert not args.share_actor
    assert args.share_critic
    assert args.use_centralized_V # Implied by MAPPO structure

    assert mappo.num_agents == args.num_agents
    assert mappo.device == torch.device(args.device)

    # Check agents (if not shared actor)
    assert len(mappo.agents) == args.num_agents
    for i, agent in enumerate(mappo.agents):
        assert isinstance(agent, _MAPPOAgent)
        assert agent.idx == i
        assert isinstance(agent.actor, StochasticPolicy)
        assert isinstance(agent.actor_optimizer, torch.optim.Adam)
        assert agent.actor_optimizer.defaults['lr'] == args.actor_lr
        # Critic is shared, so agent should not have its own critic if share_critic=True
        assert not hasattr(agent, 'critic') 
        assert not hasattr(agent, 'critic_optimizer')

    # Check shared critic
    assert isinstance(mappo.shared_critic, VNet)
    assert isinstance(mappo.shared_critic_optimizer, torch.optim.Adam)
    assert mappo.shared_critic_optimizer.defaults['lr'] == args.critic_lr
    
    # Check that no shared_actor object exists if share_actor is False
    assert not hasattr(mappo, 'shared_actor')
    assert not hasattr(mappo, 'shared_actor_optimizer')


# --- Act and Get Values Tests ---
def test_mappo_act_and_values_discrete_unshared_actor_shared_critic(mappo_instance_discrete, mappo_config_dict_discrete):
    args = Namespace(**mappo_config_dict_discrete)
    mappo = mappo_instance_discrete
    
    # Prepare dummy inputs for act and get_values
    # For act with unshared policy: list of obs tensors
    # obs_for_act = [torch.randn(args.n_rollout_threads, *obs_shape).to(mappo.device) for obs_shape in args.obs_space_shape_defs]
    
    # The MAPPO.act expects a single tensor if shared_policy=False, assumes it's been processed.
    # Let's make a batch for one step: (n_rollout_threads * num_agents, obs_dim) if agents have same obs_dim
    # Or if obs_dim differs, then MAPPO.act must take a list.
    # The code: `for i, agent in enumerate(self.agents): action, log_prob = agent.act(obs[i], ...)`
    # This implies obs to MAPPO.act should be a list if not shared_policy.
    # Let's create obs as a list of tensors [ (n_rollout_threads, obs_dim_agent_i) ]
    obs_list_for_act = []
    for i in range(args.num_agents):
        obs_list_for_act.append(torch.randn(args.n_rollout_threads, args.obs_space_shape_defs[i][0], device=mappo.device))

    # Test act method (non-deterministic)
    actions, log_probs = mappo.act(obs_list_for_act, deterministic=False)
    assert len(actions) == args.num_agents
    assert len(log_probs) == args.num_agents
    for i in range(args.num_agents):
        assert isinstance(actions[i], torch.Tensor)
        assert actions[i].shape == (args.n_rollout_threads, 1) # Discrete action is a single index
        assert actions[i].dtype == torch.long
        assert torch.all(actions[i] >= 0) and torch.all(actions[i] < args.discrete_action_n_per_agent[i])
        
        assert isinstance(log_probs[i], torch.Tensor)
        assert log_probs[i].shape == (args.n_rollout_threads, 1)

    # Test act method (deterministic)
    # For discrete, deterministic usually means taking argmax, log_prob might be 0 or of max.
    actions_det = mappo.act(obs_list_for_act, deterministic=True) # log_probs not returned by API if deterministic
    assert len(actions_det) == args.num_agents
    for i in range(args.num_agents):
        assert isinstance(actions_det[i], torch.Tensor)
        assert actions_det[i].shape == (args.n_rollout_threads, 1)


    # Test get_values method
    # `get_values` expects `obs` as (N, obs_dim) where N = n_rollout_threads * num_agents (if obs same for all)
    # Or it takes (total_agents_in_batch, individual_obs_dim) and then permutes.
    # Looking at `get_values` code: `N, obs_dim = obs.shape`. It expects a single tensor.
    # This `obs` is likely the per-agent observations stacked: (n_rollout_threads * num_agents, obs_dim_agent_0)
    # This assumes homogeneous agent observation spaces for simple stacking.
    # If obs spaces are different, `get_values` current logic is problematic. Our config has same obs_shape.
    
    # Stack obs from obs_list_for_act:
    # Each element is (n_rollout_threads, obs_dim). Stack along new dim then reshape.
    # Or, interleave them. RolloutStorage flattens to (n_steps * n_rollout_threads, num_agents, obs_dim)
    # then for get_values it might pass obs[:, agent_idx, :] if it were per agent.
    # But `get_values` permutes based on agent index within the flattened batch.
    # Let's form `obs_for_values` as if it's one step from `n_rollout_threads` environments,
    # with agents' observations concatenated.
    # Shape: (n_rollout_threads * num_agents, obs_dim_per_agent)
    # Since obs_shape_defs are same (10,), obs_dim_per_agent = 10.
    single_obs_dim = args.obs_space_shape_defs[0][0]
    stacked_obs_for_values = torch.cat(obs_list_for_act, dim=0) # (n_rollout_threads * num_agents, single_obs_dim)
    # This matches the (N, obs_dim) expectation where N is total number of agent observations in the batch.
    
    values = mappo.get_values(stacked_obs_for_values)
    assert isinstance(values, torch.Tensor)
    # `get_values` returns (N, 1) where N = n_rollout_threads * num_agents
    assert values.shape == (args.n_rollout_threads * args.num_agents, 1)


# --- Train (Update) Method Tests ---
@pytest.fixture
def populated_rollout_storage(mappo_instance_discrete, mappo_config_dict_discrete):
    args = Namespace(**mappo_config_dict_discrete)
    mappo = mappo_instance_discrete
    
    obs_spaces_list, action_spaces_list = get_spaces_from_config_params(
        args.num_agents, args.obs_space_shape_defs, args.action_type,
        args.discrete_action_n_per_agent, args.continuous_action_shape_defs
    )

    # MAPPO does not create its own RolloutStorage, it's passed to train.
    # But for testing MAPPO.train, we need a RolloutStorage instance.
    # The MAPPO class itself doesn't store it as self.storage.
    # The training script would manage the storage.
    # For this test, we create one.
    storage = RolloutStorage(
        args, # RolloutStorage expects args directly
        args.n_steps,
        obs_spaces_list,
        action_spaces_list, # Pass the actual space objects
        device=mappo.device
    )
    assert storage.obs.shape == (args.n_steps + 1, args.n_rollout_threads, args.num_agents, *args.obs_space_shape_defs[0])

    # Populate storage with dummy data for n_steps
    # Initial recurrent_hs (if used, MLP uses None effectively)
    # MAPPO's RolloutStorage interface expects numpy arrays for insert.
    # And it expects obs and share_obs separately. Share_obs is the centralized state.
    
    # For MLP, recurrent_hs can be zeros or None. Let's use zeros for storage shape.
    # RolloutStorage itself doesn't have explicit recurrent_hs unless it's for actor.
    # PPO agent handles recurrent state if policy is recurrent.
    # The `RolloutStorage` stores `obs`, `rewards`, `actions`, `action_log_probs`, `values`, `masks`, `truncated`.
    # `share_obs` is stored in `RolloutStorage.obs` if `use_centralized_V=False` effectively makes obs=share_obs.
    # If `use_centralized_V=True`, `RolloutStorage.share_obs` should be populated.
    # The `RolloutStorage` code shows `self.obs` and no separate `self.share_obs`.
    # The `_build_global_obs` method in RolloutStorage creates global_obs from self.obs.
    # So, we just need to insert per-agent obs into `storage.obs`.

    # Initial step obs
    current_obs_list = []
    for i in range(args.num_agents):
        current_obs_list.append(np.random.rand(args.n_rollout_threads, *args.obs_space_shape_defs[i]).astype(np.float32))
    storage.obs[0] = np.array(current_obs_list).transpose(1,0,2) # Transpose to (threads, agents, dim)

    for step in range(args.n_steps):
        # Convert current_obs_list (list of np arrays) to list of tensors for mappo.act
        obs_tensors_for_act = [torch.from_numpy(o).to(mappo.device) for o in current_obs_list]
        
        actions_list_torch, log_probs_list_torch = mappo.act(obs_tensors_for_act, deterministic=False)
        
        # For get_values, stack agent observations for each thread
        # Each obs_tensors_for_act[i] is (n_rollout_threads, obs_dim_i)
        # We need (n_rollout_threads * num_agents, obs_dim_i) assuming homogeneous obs
        # Or, if get_values can take the list and process:
        # The `MAPPO.get_values` takes a single tensor (N, obs_dim) where N is total number of agent instances.
        # So, we stack obs from all threads and agents.
        # obs_for_value_call shape: (n_rollout_threads, num_agents, obs_dim) -> reshape to (n_rollout_threads*num_agents, obs_dim)
        # obs_tensors_for_act is list of (n_rollout_threads, obs_dim_agent_i)
        # Assuming homogeneous obs_dim for stacking:
        stacked_obs_for_value = torch.stack(obs_tensors_for_act, dim=1) # (n_rollout_threads, num_agents, obs_dim)
        reshaped_obs_for_value = stacked_obs_for_value.reshape(args.n_rollout_threads * args.num_agents, -1)
        values_torch = mappo.get_values(reshaped_obs_for_value) # (n_rollout_threads * num_agents, 1)
        # Reshape values back to (n_rollout_threads, num_agents, 1) for storage
        values_for_storage_torch = values_torch.reshape(args.n_rollout_threads, args.num_agents, 1)

        # Convert actions and log_probs to numpy for storage
        actions_np = np.array([a.cpu().numpy() for a in actions_list_torch]).transpose(1,0,2) # (threads, agents, 1)
        log_probs_np = np.array([lp.cpu().numpy() for lp in log_probs_list_torch]).transpose(1,0,2) # (threads, agents, 1)
        values_np = values_for_storage_torch.cpu().numpy() # (threads, agents, 1)

        # Dummy rewards, masks, truncates
        rewards_np = np.random.rand(args.n_rollout_threads, args.num_agents, 1).astype(np.float32)
        # masks are 1 if not done, 0 if done. Start with all 1s.
        masks_np = np.ones((args.n_rollout_threads, args.num_agents, 1), dtype=np.float32)
        truncated_np = np.zeros((args.n_rollout_threads, args.num_agents, 1), dtype=bool)
        if step == args.n_steps - 1: # Example: last step, some are done
            masks_np[0, :, 0] = 0.0 # First thread, all agents done
            truncated_np[0, :, 0] = True # Assume truncated

        # Insert expects obs to be list of np arrays (per agent)
        # actions, log_probs, values, rewards, masks, truncated also as np arrays matching storage dims
        storage.insert(current_obs_list, actions_np, log_probs_np, values_np, rewards_np, masks_np, truncated_np)
        
        # Update current_obs_list for next iteration (dummy step)
        current_obs_list = [np.random.rand(args.n_rollout_threads, *shape).astype(np.float32) for shape in args.obs_space_shape_defs]
        # The last obs (obs[n_steps]) is set by insert based on this new current_obs_list

    # Compute returns and advantages
    # Need last_values for compute_returns_and_advantages
    # This is V(s_T) where s_T is storage.obs[-1]
    last_obs_for_value_call_list = [torch.from_numpy(storage.obs[-1, :, i]).to(mappo.device) for i in range(args.num_agents)]
    stacked_last_obs_for_value = torch.stack(last_obs_for_value_call_list, dim=1) # (threads, agents, dim)
    reshaped_last_obs_for_value = stacked_last_obs_for_value.reshape(args.n_rollout_threads * args.num_agents, -1)
    
    with torch.no_grad():
        last_values_torch = mappo.get_values(reshaped_last_obs_for_value) # (threads*agents, 1)
    last_values_np = last_values_torch.reshape(args.n_rollout_threads, args.num_agents, 1).cpu().numpy()
    
    last_masks_np = storage.masks[-1] # Masks corresponding to storage.obs[-1] (s_T)
    last_truncated_np = storage.truncated[-1]

    storage.compute_returns_and_advantages(last_values_np, args.gamma, args.gae_lambda, args.use_gae, args.use_proper_time_limits) # RolloutStorage uses args for some of these
    
    return storage


def test_mappo_train_discrete_unshared_actor_shared_critic(mappo_instance_discrete, populated_rollout_storage, mappo_config_dict_discrete):
    args = Namespace(**mappo_config_dict_discrete)
    mappo = mappo_instance_discrete
    storage = populated_rollout_storage

    # Store initial parameters for comparison
    initial_actor_params = []
    if not args.share_actor:
        for agent in mappo.agents:
            initial_actor_params.append([p.clone().detach() for p in agent.actor.parameters()])
    # else: # TODO: Handle shared actor parameter check

    initial_critic_params = []
    if args.share_critic:
        initial_critic_params.append([p.clone().detach() for p in mappo.shared_critic.parameters()])
    # else: # TODO: Handle unshared agent critic parameter check

    # Call the main train method
    train_infos = mappo.train(storage)

    assert isinstance(train_infos, dict) 
    # train_infos is defaultdict(lambda: defaultdict(float)), keys are agent_idx or 0 for shared components
    
    # Check losses (these are averaged over epochs and minibatches)
    # For shared critic and unshared actor:
    # train_infos[0] will have critic_loss (if shared_critic is handled under agent 0 key by default)
    # train_infos[i] for each agent `i` will have actor_loss, entropy_loss etc.
    # Based on MAPPO.train structure:
    # If shared_policy, policy losses are in train_info[0].
    # If not shared_policy, agent.train returns its losses, stored in train_info[i].
    # If shared_critic, critic losses are in train_info[0].
    # If not shared_critic (done by agent.train), critic losses are in train_info[i].
    # Current config: not shared_actor, shared_critic.
    # So, critic_loss in train_infos[0]. Actor losses in train_infos[i].

    assert "critic_loss" in train_infos[0] 
    assert isinstance(train_infos[0]["critic_loss"], float)
    if args.use_max_grad_norm:
         assert "critic_grad_norm" in train_infos[0]

    for i in range(args.num_agents):
        assert "actor_loss" in train_infos[i]
        assert "entropy_loss" in train_infos[i]
        assert isinstance(train_infos[i]["actor_loss"], float)
        assert isinstance(train_infos[i]["entropy_loss"], float)
        if args.use_max_grad_norm:
            assert "actor_grad_norm" in train_infos[i]


    # Check parameter updates
    if not args.share_actor:
        for i, agent in enumerate(mappo.agents):
            for p_initial, p_updated in zip(initial_actor_params[i], agent.actor.parameters()):
                assert not torch.allclose(p_initial, p_updated.data), f"Agent {i} actor params did not change."
    # else: TODO shared actor check

    if args.share_critic:
        for p_initial, p_updated in zip(initial_critic_params[0], mappo.shared_critic.parameters()):
            assert not torch.allclose(p_initial, p_updated.data), "Shared critic params did not change."
    # else: TODO unshared critic check


# Basic check that imports work
def test_imports():
    assert MAPPO is not None
    assert _MAPPOAgent is not None
    assert RolloutStorage is not None

# TODO:
# - Test with continuous actions.
# - Test with shared actor.
# - Test with unshared critic.
# - Test with recurrent policy.
# - Test specific PPO loss calculations with known inputs if possible (might be too complex).
# - Test edge cases in RolloutStorage population (e.g., all steps done early).
# - Test learning rate decay.
# - Test PopArt if used.
# - Test save/load functionality.
# - The `MAPPO.get_values` method's current way of handling `obs` (expecting a single stacked tensor for all agents)
#   and then building `critic_input` needs to be carefully aligned with how `RolloutStorage` provides data.
#   The `populated_rollout_storage` fixture attempts to do this correctly for testing `train`.
#   When `get_values` is called during rollout generation (one env step at a time), the `obs` input
#   would be `(n_rollout_threads, num_agents, obs_dim_agent)` if passed directly, or it might be pre-processed.
#   The current `get_values` takes `(N, obs_dim)` and assumes N is total agents in batch.
#   The `RolloutStorage.insert` takes `values` shaped `(n_rollout_threads, num_agents, 1)`.
#   So, `get_values` must produce this. The current test for `get_values` reshapes its output. This seems consistent.The initial tests for MAPPO algorithm (initialization, act/get_values, and train methods) have been implemented in `tests/test_mappo.py`.

**Key aspects covered:**

1.  **Configuration (`mappo_config_dict_discrete`):**
    *   A fixture provides a detailed configuration for MAPPO, currently set for discrete actions, non-shared actor, and shared critic. It includes PPO-specific parameters, network sizes, learning rates, etc.
    *   A helper `get_spaces_from_config_params` creates `gymnasium.spaces` objects from the configuration.

2.  **Initialization Test (`test_mappo_initialization_discrete_unshared_actor_shared_critic`):**
    *   Verifies correct instantiation of `MAPPO` based on the configuration.
    *   Checks for the correct number of `_MAPPOAgent` instances when the actor is not shared.
    *   Ensures each agent has a `StochasticPolicy` (actor) and an Adam optimizer.
    *   Confirms that agents do *not* have individual critics when `share_critic=True`.
    *   Verifies the presence and correct type (`VNet`) of the `shared_critic` and its optimizer on the `MAPPO` instance.

3.  **`act` and `get_values` Test (`test_mappo_act_and_values_discrete_unshared_actor_shared_critic`):**
    *   Tests `MAPPO.act()` for both deterministic and stochastic action selection:
        *   Input observations are lists of tensors (one per agent, with a batch dim for `n_rollout_threads`).
        *   Verifies output actions and log probabilities have correct shapes, types, and that discrete actions are within valid range.
    *   Tests `MAPPO.get_values()`:
        *   Input observations are stacked into a single tensor `(n_rollout_threads * num_agents, obs_dim_agent)` as expected by the current `get_values` implementation (which then internally reconstructs the centralized critic input).
        *   Verifies output values have the correct shape `(n_rollout_threads * num_agents, 1)`.

4.  **`train` (Update) Method Test (`test_mappo_train_discrete_unshared_actor_shared_critic`):**
    *   Uses a fixture `populated_rollout_storage` to set up an instance of the actual `RolloutStorage`.
    *   This fixture populates the storage by simulating `n_steps` of data generation:
        *   Calls `mappo.act()` to get actions and log probabilities.
        *   Calls `mappo.get_values()` to get value predictions.
        *   Inserts these, along with dummy rewards and masks, into the `RolloutStorage`.
        *   Calls `storage.compute_returns_and_advantages()` after all steps are inserted, using `mappo.get_values()` for the value of the final next state.
    *   Calls `mappo.train(storage)`.
    *   Verifies that `train_infos` (a dictionary) is returned.
    *   Checks for presence and correct type (float) of averaged losses (`critic_loss`, `actor_loss`, `entropy_loss`) in `train_infos`. The structure of `train_infos` is checked based on shared/unshared components.
    *   Confirms that parameters of the actor networks (per-agent) and the shared critic network change after the `train()` call, indicating that backpropagation and optimizer steps occurred.

This set of tests provides a good baseline for verifying the core mechanics of the MAPPO implementation, particularly its interaction with the `RolloutStorage` and the PPO update process for both shared and unshared components (starting with unshared actor, shared critic). Future tests can expand on different configurations (continuous actions, fully shared/unshared networks, recurrent policies) and more specific numerical checks.
