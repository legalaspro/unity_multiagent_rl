import pytest
import torch
import numpy as np
from argparse import Namespace
from gymnasium.spaces import Box

from algos.matd3 import MATD3
from algos.agent._matd3_agent import _MATD3Agent
from buffers.replay_buffer import ReplayBuffer # MATD3 instantiates this
from tests.helpers import MockReplayBuffer # For mocking during update test

from networks.actors.deterministic_policy import DeterministicPolicy
from networks.critics.twin_q_net import TwinQNet


@pytest.fixture
def matd3_config_dict_continuous_decentral_critic(): # Name implies decentralized, but MATD3 critics are centralized
    """Basic configuration dictionary for MATD3 tests."""
    num_agents = 2
    obs_shapes = [(10,), (10,)]
    act_shapes = [(2,), (2,)]   # Continuous actions

    config = {
        "num_agents": num_agents,
        "obs_space_shape_defs": obs_shapes,
        "act_space_shape_defs": act_shapes,
        "action_type": 'Continuous',

        "device": 'cpu',
        "actor_lr": 1e-4,
        "critic_lr": 1e-3,
        "hidden_sizes": (64, 64),

        "gamma": 0.99,
        "tau": 0.005,
        "buffer_size": 10000, # Smaller for tests
        "batch_size": 32,    # Smaller for tests

        "policy_noise": 0.2,
        "target_policy_noise": 0.2, # Same as policy_noise for compatibility
        "noise_clip": 0.5,
        "target_noise_clip": 0.5, # Same as noise_clip for compatibility
        "policy_freq": 2,
        "policy_delay": 2, # Same as policy_freq for compatibility

        "n_step": 1,

        # _MATD3Agent critics take total_state_size and total_action_size, implying centralized info.
        "use_centralized_V": True, # This flag is more conceptual for MATD3's critic structure.
        "shared_critic": False, # Each agent has its own TwinQNet critic, but it's centralized.

        "seed": 123,
        "exploration_noise": 0.1,
        "use_max_grad_norm": True,
        "max_grad_norm": 1.0,
        "state_dependent_std": False, # Not used by DeterministicPolicy
    }
    return config

# Helper to create spaces
def get_spaces_from_config_params_matd3(num_agents, obs_space_shapes, act_space_shapes):
    obs_spaces = [Box(low=-np.inf, high=np.inf, shape=s, dtype=np.float32) for s in obs_space_shapes]
    act_spaces = [Box(low=-1, high=1, shape=s, dtype=np.float32) for s in act_space_shapes]
    return obs_spaces, act_spaces

@pytest.fixture
def matd3_instance(matd3_config_dict_continuous_decentral_critic):
    args = Namespace(**matd3_config_dict_continuous_decentral_critic)
    obs_spaces, action_spaces = get_spaces_from_config_params_matd3(
        args.num_agents, args.obs_space_shape_defs, args.act_space_shape_defs
    )
    # As with MASAC, MATD3.train expects a buffer.
    # MATD3.__init__ does not create self.buffer from the provided code.
    # So, for init test, we don't check self.buffer.
    # For train test, we pass a mock buffer.
    matd3_algo = MATD3(args, obs_spaces, action_spaces, device=torch.device(args.device))
    return matd3_algo

# --- Initialization Tests ---
def test_matd3_initialization(matd3_instance, matd3_config_dict_continuous_decentral_critic):
    args = Namespace(**matd3_config_dict_continuous_decentral_critic)
    matd3 = matd3_instance

    assert not args.shared_critic # Each agent has its own centralized critic

    assert matd3.num_agents == args.num_agents
    assert matd3.device == torch.device(args.device)
    assert matd3.gamma == args.gamma
    assert matd3.tau == args.tau
    assert matd3.exploration_noise == args.exploration_noise
    assert matd3.total_iterations == 1

    assert len(matd3.agents) == args.num_agents
    for i, agent in enumerate(matd3.agents):
        assert isinstance(agent, _MATD3Agent)
        assert agent.idx == i
        assert agent.state_size == args.obs_space_shape_defs[i][0]

        assert isinstance(agent.actor, DeterministicPolicy)
        assert isinstance(agent.actor_target, DeterministicPolicy)
        assert isinstance(agent.actor_optimizer, torch.optim.Adam)
        assert agent.actor_optimizer.defaults['lr'] == args.actor_lr

        assert isinstance(agent.critic, TwinQNet)
        assert isinstance(agent.critic_target, TwinQNet)
        assert isinstance(agent.critic_optimizer, torch.optim.Adam)
        assert agent.critic_optimizer.defaults['lr'] == args.critic_lr

        # Check target networks are initialized same as local
        for target_param, local_param in zip(agent.actor_target.parameters(), agent.actor.parameters()):
            assert torch.allclose(target_param.data, local_param.data)
        for target_param, local_param in zip(agent.critic_target.parameters(), agent.critic.parameters()):
            assert torch.allclose(target_param.data, local_param.data)

    # MATD3 does not create self.buffer in __init__ per provided code snippet.
    # If it were to, we'd test:
    # assert hasattr(matd3, 'buffer') and isinstance(matd3.buffer, ReplayBuffer)

# --- Act Method Tests ---
def test_matd3_act_method(matd3_instance, matd3_config_dict_continuous_decentral_critic):
    matd3 = matd3_instance
    args = Namespace(**matd3_config_dict_continuous_decentral_critic)

    # MATD3.act expects a list of tensors
    obs_tensors = [
        torch.randn(1, shape[0], device=matd3.device) # Batch size 1 for single env step
        for shape in args.obs_space_shape_defs
    ]

    # Test with exploration noise (explore=True is implicit if deterministic=False)
    actions_explore = matd3.act(obs_tensors, deterministic=False)
    assert len(actions_explore) == args.num_agents
    for i, action_tensor in enumerate(actions_explore):
        assert isinstance(action_tensor, torch.Tensor)
        expected_shape = matd3.agents[i].action_space.shape
        assert action_tensor.shape == (1, *expected_shape) # Batch size 1
        # Check bounds (DeterministicPolicy + noise, then clamped by agent.act)
        low = matd3.agents[i].action_low
        high = matd3.agents[i].action_high
        assert torch.all(action_tensor >= low) and torch.all(action_tensor <= high)

    # Test deterministic (explore=False)
    actions_deterministic1 = matd3.act(obs_tensors, deterministic=True)
    assert len(actions_deterministic1) == args.num_agents
    for i, action_tensor in enumerate(actions_deterministic1):
        assert isinstance(action_tensor, torch.Tensor)
        expected_shape = matd3.agents[i].action_space.shape
        assert action_tensor.shape == (1, *expected_shape)
        low = matd3.agents[i].action_low
        high = matd3.agents[i].action_high
        assert torch.all(action_tensor >= low) and torch.all(action_tensor <= high)

    actions_deterministic2 = matd3.act(obs_tensors, deterministic=True)
    for a1, a2 in zip(actions_deterministic1, actions_deterministic2):
        assert torch.allclose(a1, a2, atol=1e-6)


# --- Train (Update) Method Tests ---
@pytest.fixture
def mock_replay_buffer_for_matd3(matd3_config_dict_continuous_decentral_critic, mocker):
    args = Namespace(**matd3_config_dict_continuous_decentral_critic)
    obs_spaces, action_spaces = get_spaces_from_config_params_matd3(
        args.num_agents, args.obs_space_shape_defs, args.act_space_shape_defs
    )
    mock_buffer_instance = MockReplayBuffer(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        state_spaces=obs_spaces,
        action_spaces=action_spaces,
        device=torch.device(args.device),
        n_step=args.n_step,
        gamma=args.gamma,
        num_agents=args.num_agents
    )
    mock_buffer_instance.add_call_count = args.batch_size + 10
    return mock_buffer_instance

def test_matd3_train_decentral_critic(matd3_instance, mock_replay_buffer_for_matd3, matd3_config_dict_continuous_decentral_critic):
    matd3 = matd3_instance
    mock_buffer = mock_replay_buffer_for_matd3
    args = Namespace(**matd3_config_dict_continuous_decentral_critic)
    tau = args.tau
    policy_freq = args.policy_freq

    initial_actor_params = []
    initial_critic_params = [] # List of lists (for twin Q per agent)
    initial_target_actor_params = []
    initial_target_critic_params = []

    for agent in matd3.agents:
        initial_actor_params.append([p.clone().detach() for p in agent.actor.parameters()])
        crit_params = []
        for p_q1, p_q2 in zip(agent.critic._critic1.parameters(), agent.critic._critic2.parameters()):
            crit_params.append(p_q1.clone().detach())
            crit_params.append(p_q2.clone().detach())
        initial_critic_params.append(crit_params)

        initial_target_actor_params.append([p.clone().detach() for p in agent.actor_target.parameters()])
        target_crit_params = []
        for p_q1, p_q2 in zip(agent.critic_target._critic1.parameters(), agent.critic_target._critic2.parameters()):
            target_crit_params.append(p_q1.clone().detach())
            target_crit_params.append(p_q2.clone().detach())
        initial_target_critic_params.append(target_crit_params)

    # --- Test critic-only update (total_iterations % policy_freq != 0) ---
    # Ensure total_iterations starts at 1, so first call updates critic only if policy_freq > 1
    assert matd3.total_iterations == 1
    if policy_freq > 1:
        train_infos_crit_only = matd3.train(mock_buffer)
        assert matd3.total_iterations == 2
        for i, agent in enumerate(matd3.agents):
            agent_info = train_infos_crit_only[i]
            assert "critic_loss" in agent_info
            assert "actor_loss" not in agent_info # Actor loss should not be in info if not updated
            # Check critic params changed, actor params did not
            idx = 0
            for p_q1_initial, p_q1_updated in zip(initial_critic_params[i][idx::2], agent.critic._critic1.parameters()):
                 assert not torch.allclose(p_q1_initial, p_q1_updated.data), f"Agent {i} critic Q1 params did not change on critic-only step."
            idx = 1
            for p_q2_initial, p_q2_updated in zip(initial_critic_params[i][idx::2], agent.critic._critic2.parameters()):
                 assert not torch.allclose(p_q2_initial, p_q2_updated.data), f"Agent {i} critic Q2 params did not change on critic-only step."

            for p_initial, p_updated in zip(initial_actor_params[i], agent.actor.parameters()):
                assert torch.allclose(p_initial, p_updated.data), f"Agent {i} actor params changed on critic-only step."

            # Target critics should have been updated, target actors not yet
            # Check that target network parameters have been soft updated
            # We'll just verify that target params are different from initial and not same as main
            idx = 0
            for p_target_old, p_main_updated in zip(initial_target_critic_params[i][idx::2], agent.critic._critic1.parameters()):
                # Find corresponding parameter in target network by matching shapes
                p_target_new = None
                for param in agent.critic_target._critic1.parameters():
                    if param.shape == p_target_old.shape:
                        p_target_new = param.data
                        break
                assert p_target_new is not None, f"Could not find matching target parameter for Agent {i} critic Q1"
                assert not torch.allclose(p_target_old, p_target_new), f"Agent {i} target critic Q1 params did not change from initial."
                assert not torch.allclose(p_main_updated.data, p_target_new), f"Agent {i} target critic Q1 params became same as main Q1."
            # Similar check for Q2
            idx = 1
            for p_target_old, p_main_updated in zip(initial_target_critic_params[i][idx::2], agent.critic._critic2.parameters()):
                # Find corresponding parameter in target network by matching shapes
                p_target_new = None
                for param in agent.critic_target._critic2.parameters():
                    if param.shape == p_target_old.shape:
                        p_target_new = param.data
                        break
                assert p_target_new is not None, f"Could not find matching target parameter for Agent {i} critic Q2"
                assert not torch.allclose(p_target_old, p_target_new), f"Agent {i} target critic Q2 params did not change from initial."
                assert not torch.allclose(p_main_updated.data, p_target_new), f"Agent {i} target critic Q2 params became same as main Q2."

            # Target actor should not change on critic-only step
            for p_initial_target_actor, p_updated_target_actor in zip(initial_target_actor_params[i], agent.actor_target.parameters()):
                 assert torch.allclose(p_initial_target_actor, p_updated_target_actor.data), f"Agent {i} target actor params changed on critic-only step."


    # --- Test full actor-critic update (total_iterations % policy_freq == 0) ---
    # Advance total_iterations to trigger policy update
    # If policy_freq is 1, the first call already did a policy update.
    # If policy_freq is 2 (current test default), total_iterations is now 2. So this call will update policy.
    # If it was 1, it's now 2.
    if matd3.total_iterations % policy_freq != 0 : # if it's not yet time for policy update
        # Call train enough times to reach the policy update step
        for _ in range(policy_freq - (matd3.total_iterations % policy_freq)):
             intermediate_infos = matd3.train(mock_buffer) # these are critic only updates
             # Check actor params remain unchanged during these intermediate steps
             for i, agent in enumerate(matd3.agents):
                 for p_initial_actor, p_updated_actor in zip(initial_actor_params[i], agent.actor.parameters()):
                     assert torch.allclose(p_initial_actor, p_updated_actor.data), f"Agent {i} actor params changed before policy update iteration."


    # Store params again before the policy update iteration
    actor_params_before_policy_update = []
    critic_params_before_policy_update = [] # List of lists (for twin Q per agent)
    target_actor_params_before_policy_update = []
    target_critic_params_before_policy_update = []
    for agent_idx_loop, agent_obj in enumerate(matd3.agents): # Renamed to avoid conflict
        actor_params_before_policy_update.append([p.clone().detach() for p in agent_obj.actor.parameters()])
        # ... (store other params similarly)

    train_infos_full = matd3.train(mock_buffer) # This call should trigger actor update

    assert matd3.total_iterations % policy_freq == (1 if policy_freq > 0 else 0) # Iteration count after this train call

    for i, agent in enumerate(matd3.agents):
        agent_info = train_infos_full[i]
        assert "critic_loss" in agent_info
        assert "actor_loss" in agent_info # Actor loss should now be present
        assert isinstance(agent_info["critic_loss"], float)
        assert isinstance(agent_info["actor_loss"], float)

        # Check actor params changed from before this specific policy update iteration
        for p_before_policy_upd, p_updated in zip(actor_params_before_policy_update[i], agent.actor.parameters()):
            assert not torch.allclose(p_before_policy_upd, p_updated.data), f"Agent {i} actor params did not change on policy update iteration."

        # Target actor and target critics should have been updated
        # Target actor update:
        for p_target_old, p_main_updated in zip(initial_target_actor_params[i], agent.actor.parameters()): # Compare to original target, main is now updated

            # A better way to check soft update:
            # Find the specific parameter in agent.actor_target.parameters() that corresponds to p_target_old
            # For simplicity, we check if it changed from initial_target_actor_params and is not same as main actor params
            found_actor_target_param = None
            for p_target_new_actor in agent.actor_target.parameters():
                if p_target_new_actor.shape == p_target_old.shape: # Assuming shapes are unique enough or order is preserved
                    found_actor_target_param = p_target_new_actor.data
                    break
            assert found_actor_target_param is not None
            assert not torch.allclose(p_target_old, found_actor_target_param), f"Agent {i} target actor params did not change on policy update."
            assert not torch.allclose(p_main_updated.data, found_actor_target_param), f"Agent {i} target actor params became same as main actor on policy update."


# Basic check that imports work
def test_imports():
    assert MATD3 is not None
    assert _MATD3Agent is not None
    assert ReplayBuffer is not None

# TODO:
# - Refine target network parameter checking for more robustness (e.g. by name or direct application of formula).
# - Test policy_noise and noise_clip effect on target actions in critic update (might need more controlled sample data).
# - Test save/load functionality.
# - The MATD3/MADDPG code for `act_target` in the main class takes `obs` as `[num_agents, batch_size, obs_size]`.
#   This seems inconsistent with `act` which takes `[ (batch_size, obs_size_agent_i) ]` list.
#   The `_MATD3Agent.act_target` expects `(batch_size, obs_size_agent_i)`.
#   The `MATD3.act_target` loop `for agent, observation in zip(self.agents, obs)` implies `obs` should be a list.
#   This needs clarification or a fix in the main code. My tests for `act` pass a list of tensors.
#   The `MockReplayBuffer.sample()` returns `obs` and `next_obs` as lists of tensors `[ (batch_size, obs_dim_agent_i) ]`.
#   This is consistent with what `MATD3.train` then passes to `MATD3.act_target`.
# - The `_MATD3Agent`'s action clamping uses `self.action_low[0]` and `self.action_high[0]`. This is fine if all action dimensions have same bounds.
#   My test config uses `Box(low=-1, high=1, shape=(X,))`, so this is fine.
# - The `_MATD3Agent.train` uses `self.critic.Q1` for actor loss. This is standard.
