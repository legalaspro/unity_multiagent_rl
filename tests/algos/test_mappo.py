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


        "n_steps": 32, # Smaller for faster tests
        "ppo_epoch": 4, # PPO epochs
        "num_mini_batch": 2, # Number of mini-batches for PPO

        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_param": 0.2,
        "value_loss_coef": 0.5,
        "entropy_coef": 0.01,
        "max_grad_norm": 0.5,
        "use_max_grad_norm": True,

        "use_gae": True,
        "gae_lambda": 0.95,
        "use_proper_time_limits": True,
        "use_popart": False,
        "use_huber_loss": False, # For value loss
        "use_clipped_value_loss": True, # Common in PPO
        "use_value_active_masks": True,
        "use_policy_active_masks": True, # Usually refers to done masks for actor updates
        "use_feature_normalization": False, # For obs normalization

        "use_centralized_V": True,
        "shared_policy": False, # Test separate actors
        "shared_critic": True, # Test shared critic (common for MAPPO)

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

    assert not args.shared_policy
    assert args.shared_critic
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
        # Critic is shared, so agent should not have its own critic if shared_critic=True
        assert not hasattr(agent, 'critic')
        assert not hasattr(agent, 'critic_optimizer')

    # Check shared critic
    assert isinstance(mappo.shared_critic, VNet)
    assert isinstance(mappo.shared_critic_optimizer, torch.optim.Adam)
    assert mappo.shared_critic_optimizer.defaults['lr'] == args.critic_lr

    # Check that no shared_actor object exists if shared_policy is False
    assert not hasattr(mappo, 'shared_actor')
    assert not hasattr(mappo, 'shared_actor_optimizer')


# --- Act and Get Values Tests ---
def test_mappo_act_and_values_discrete_unshared_actor_shared_critic(mappo_instance_discrete, mappo_config_dict_discrete):
    args = Namespace(**mappo_config_dict_discrete)
    mappo = mappo_instance_discrete

    # Prepare dummy inputs for act and get_values
    # For act with unshared policy: list of obs tensors
    # The MAPPO.act expects a list of observations, one per agent
    # Each observation should be a single tensor (batch_size, obs_dim)
    # For testing, we'll use batch_size = 1
    obs_list_for_act = []
    for i in range(args.num_agents):
        obs_list_for_act.append(torch.randn(1, args.obs_space_shape_defs[i][0], device=mappo.device))

    # Test act method (non-deterministic)
    actions, log_probs = mappo.act(obs_list_for_act, deterministic=False)
    assert len(actions) == args.num_agents
    assert len(log_probs) == args.num_agents
    for i in range(args.num_agents):
        assert isinstance(actions[i], torch.Tensor)
        assert actions[i].shape == (1, 1) # Discrete action is a single index, batch_size=1
        assert actions[i].dtype == torch.long
        assert torch.all(actions[i] >= 0) and torch.all(actions[i] < args.discrete_action_n_per_agent[i])

        assert isinstance(log_probs[i], torch.Tensor)
        assert log_probs[i].shape == (1, 1)

    # Test act method (deterministic)
    # For discrete, deterministic usually means taking argmax, log_prob might be 0 or of max.
    actions_det = mappo.act(obs_list_for_act, deterministic=True) # log_probs not returned by API if deterministic
    assert len(actions_det) == args.num_agents
    for i in range(args.num_agents):
        assert isinstance(actions_det[i], torch.Tensor)
        assert actions_det[i].shape == (1, 1)


    # Test get_values method
    # get_values expects obs with shape (num_agents, obs_dim) for a single time step
    # Each element in obs_list_for_act is (batch_size, obs_dim) where batch_size=1
    # Stack observations from all agents: (num_agents, obs_dim)
    stacked_obs_for_values = torch.stack([obs.squeeze(0) for obs in obs_list_for_act], dim=0) # (num_agents, obs_dim)

    values = mappo.get_values(stacked_obs_for_values)
    assert isinstance(values, torch.Tensor)
    # get_values returns (num_agents, 1)
    assert values.shape == (args.num_agents, 1)


# --- Train (Update) Method Tests ---
@pytest.fixture
def populated_rollout_storage(mappo_instance_discrete, mappo_config_dict_discrete):
    args = Namespace(**mappo_config_dict_discrete)
    mappo = mappo_instance_discrete

    obs_spaces_list, action_spaces_list = get_spaces_from_config_params(
        args.num_agents, args.obs_space_shape_defs, args.action_type,
        args.discrete_action_n_per_agent, args.continuous_action_shape_defs
    )

    storage = RolloutStorage(
        args,
        args.n_steps,
        obs_spaces_list,
        action_spaces_list,
        device=mappo.device
    )
    assert storage.obs.shape == (args.n_steps + 1, args.num_agents, *args.obs_space_shape_defs[0])

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

    # Initial step obs - RolloutStorage expects shape (agents, obs_dim)
    current_obs_list = []
    for i in range(args.num_agents):
        current_obs_list.append(np.random.rand(*args.obs_space_shape_defs[i]).astype(np.float32))
    # Stack to (agents, obs_dim)
    storage.obs[0] = np.stack(current_obs_list, axis=0)

    for step in range(args.n_steps):
        # Convert current_obs_list to tensors for mappo.act (add batch dimension)
        obs_tensors_for_act = [torch.from_numpy(o).unsqueeze(0).to(mappo.device) for o in current_obs_list]

        actions_list_torch, log_probs_list_torch = mappo.act(obs_tensors_for_act, deterministic=False)

        # For get_values, convert to tensors and stack: (num_agents, obs_dim)
        thread_obs = torch.stack([torch.from_numpy(obs) for obs in current_obs_list], dim=0).to(mappo.device)
        thread_values = mappo.get_values(thread_obs) # (num_agents, 1)

        # Convert actions and log_probs to numpy for storage
        actions_np = np.stack([a.squeeze(0).cpu().numpy() for a in actions_list_torch], axis=0) # (agents, 1)
        log_probs_np = np.stack([lp.squeeze(0).cpu().numpy() for lp in log_probs_list_torch], axis=0) # (agents, 1)
        values_np = thread_values.cpu().numpy() # (agents, 1)

        # Dummy rewards, masks, truncates
        rewards_np = np.random.rand(args.num_agents, 1).astype(np.float32)
        masks_np = np.ones((args.num_agents, 1), dtype=np.float32)
        truncated_np = np.zeros((args.num_agents, 1), dtype=bool)
        if step == args.n_steps - 1: # Example: last step, some are done
            masks_np[:, 0] = 0.0 # All agents done
            truncated_np[:, 0] = True # Assume truncated

        # Insert data into storage
        storage.insert(current_obs_list, actions_np, log_probs_np, values_np, rewards_np, masks_np, truncated_np)

        # Update current_obs_list for next iteration
        current_obs_list = [np.random.rand(*shape).astype(np.float32) for shape in args.obs_space_shape_defs]

    # Compute returns and advantages
    # Need last_values for compute_returns_and_advantages
    # storage.obs[-1] has shape (num_agents, obs_dim)
    last_obs = torch.from_numpy(storage.obs[-1]).to(mappo.device) # (num_agents, obs_dim)

    with torch.no_grad():
        last_values = mappo.get_values(last_obs) # (num_agents, 1)

    last_values_np = last_values.cpu().numpy()

    storage.compute_returns_and_advantages(last_values_np, args.gamma, args.gae_lambda, args.use_gae, args.use_proper_time_limits)

    return storage


def test_mappo_train_discrete_unshared_actor_shared_critic(mappo_instance_discrete, populated_rollout_storage, mappo_config_dict_discrete):
    args = Namespace(**mappo_config_dict_discrete)
    mappo = mappo_instance_discrete
    storage = populated_rollout_storage

    # Store initial parameters for comparison
    initial_actor_params = []
    if not args.shared_policy:
        for agent in mappo.agents:
            initial_actor_params.append([p.clone().detach() for p in agent.actor.parameters()])
    # else: # TODO: Handle shared actor parameter check

    initial_critic_params = []
    if args.shared_critic:
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
    if not args.shared_policy:
        for i, agent in enumerate(mappo.agents):
            for p_initial, p_updated in zip(initial_actor_params[i], agent.actor.parameters()):
                assert not torch.allclose(p_initial, p_updated.data), f"Agent {i} actor params did not change."
    # else: TODO shared actor check

    if args.shared_critic:
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