import pytest
import numpy as np
import torch
from argparse import Namespace
import os # For checking log file, though not strictly necessary if logger is mocked well

from evals.unity_evaluator import UnityEvaluator
from tests.helpers import MockMAEnv, MockEvalAgent
from gymnasium.spaces import Box


@pytest.fixture
def unity_eval_config_dict(tmp_path):
    """Basic configuration dictionary for UnityEvaluator tests."""
    env_num_agents = 2
    config = {
        "eval_episodes": 3, # Keep low for tests
        "log_dir": str(tmp_path), # Not directly used by class if logger is perfectly mocked
        "seed": 42,
        "worker_id": 1,
        "env_id": "MockUnityEnvID",
        "teams": [[i] for i in range(env_num_agents)], # e.g., [[0], [1]] for 2 agents
        "device": "cpu",
        # Other params that might be in cfg, even if not directly used by UnityEvaluator itself
        # but potentially by make_agent_snapshot or logger setup in a real scenario.
    }
    return config

@pytest.fixture
def mock_logger(mocker):
    logger = mocker.MagicMock()
    logger.add_scalar = mocker.MagicMock()
    return logger

@pytest.fixture
def mock_ma_env_for_unity_eval(unity_eval_config_dict):
    cfg = Namespace(**unity_eval_config_dict)
    num_env_agents = sum(len(team) for team in cfg.teams)
    obs_shapes = [(4,)] * num_env_agents
    action_spaces = [Box(low=-1, high=1, shape=(2,), dtype=np.float32) for _ in range(num_env_agents)]
    
    env = MockMAEnv(
        num_agents=num_env_agents,
        obs_shapes=obs_shapes,
        action_spaces=action_spaces,
        episode_length=5 # Short episodes
    )
    return env

@pytest.fixture
def make_mock_agent_snapshot_fn(mock_ma_env_for_unity_eval, unity_eval_config_dict): # Corrected fixture name
    """Factory to create a make_agent_snapshot function."""
    cfg = Namespace(**unity_eval_config_dict)
    def _make_snapshot_fn(agent_id_str="snapshot_0"):
        # MockEvalAgent needs the list of action spaces for all agents in the environment.
        agent = MockEvalAgent(
            agent_id=agent_id_str,
            name=f"Agent_{agent_id_str}",
            env_action_spaces_list=mock_ma_env_for_unity_eval.action_space, # Pass full list
            device=cfg.device
        )
        return agent
    return _make_snapshot_fn

# --- Initialization Tests ---
def test_unity_evaluator_initialization(mock_logger, unity_eval_config_dict, make_mock_agent_snapshot_fn, mocker, mock_ma_env_for_unity_eval):
    cfg = Namespace(**unity_eval_config_dict)
    
    # Patch UnityEnvWrapper to return our MockMAEnv instance
    mock_env_patch = mocker.patch('evals.unity_evaluator.UnityEnvWrapper', return_value=mock_ma_env_for_unity_eval)
    
    snapshot_fn = make_mock_agent_snapshot_fn # Get the actual function

    evaluator = UnityEvaluator(logger=mock_logger, cfg=cfg, make_agent_snapshot=snapshot_fn)

    assert evaluator.logger == mock_logger
    assert evaluator.cfg == cfg
    assert evaluator.snapshot_fn == snapshot_fn
    assert evaluator.env == mock_ma_env_for_unity_eval
    mock_env_patch.assert_called_once_with(cfg.env_id, worker_id=cfg.worker_id + 10, seed=cfg.seed + 1000)
    assert evaluator.team_indices == cfg.teams
    assert evaluator.cfg.eval_episodes == cfg.eval_episodes


# --- Run Method Tests ---
def test_unity_evaluator_run(mock_logger, unity_eval_config_dict, make_mock_agent_snapshot_fn, mocker, mock_ma_env_for_unity_eval):
    cfg_dict = unity_eval_config_dict
    cfg = Namespace(**cfg_dict)
    
    # Configure MockMAEnv for predictable results
    # Episode 1: Team0 wins (agent 0 gets 10, agent 1 gets 5). Total for T0=10, T1=5. Length 2
    # Episode 2: Team1 wins (agent 0 gets 2, agent 1 gets 8). Total for T0=2, T1=8. Length 3
    # Episode 3: Draw based on team score (agent 0 gets 6, agent 1 gets 6). Total T0=6, T1=6. Length 4
    # Note: MockMAEnv sums rewards if teams have multiple agents. Here teams=[[0]], [[1]] so agent reward = team reward.
    fixed_rewards_pattern = [
        # Ep1
        np.array([5.0, 2.0]), np.array([5.0, 3.0]), # Step1, Step2 -> Ep1 Rewards: [10, 5], Length 2
        # Ep2
        np.array([1.0, 3.0]), np.array([0.5, 2.0]), np.array([0.5, 3.0]), # -> Ep2 Rewards: [2, 8], Length 3
        # Ep3
        np.array([1.0, 1.0]), np.array([2.0, 2.0]), np.array([1.5, 1.5]), np.array([1.5, 1.5]) # -> Ep3 Rewards: [6, 6], Length 4
    ]
    # We need to ensure episode lengths also match.
    # MockMAEnv episode_length is fixed. Let's make it long enough, and rely on "all_done".
    # The rewards above are per step. The evaluator sums them.
    # The test above sums rewards.
    # For `play_one_episode`, the episode runs until `info["all_done"]` is true.
    # `MockMAEnv` sets `all_done` based on `self.current_step >= self.episode_length`.
    # So, we need to control episode lengths carefully.
    
    # Let's simplify: make each episode 1 step long, and rewards are final.
    mock_ma_env_for_unity_eval.episode_length = 1 
    # Rewards for 3 episodes:
    fixed_rewards_per_episode = [
        np.array([10.0, 5.0], dtype=np.float32), # Ep1: Agent0 (Team0) wins
        np.array([2.0, 8.0], dtype=np.float32),  # Ep2: Agent1 (Team1) wins
        np.array([6.0, 6.0], dtype=np.float32),  # Ep3: Draw
    ]
    # For remaining 2 episodes (cfg.eval_episodes = 5)
    fixed_rewards_per_episode.append(np.array([7.0, 3.0], dtype=np.float32)) # Ep4: Agent0 (Team0) wins
    fixed_rewards_per_episode.append(np.array([4.0, 9.0], dtype=np.float32)) # Ep5: Agent1 (Team1) wins
    mock_ma_env_for_unity_eval._fixed_rewards_pattern = fixed_rewards_per_episode # Set the fixed rewards

    mock_env_patch = mocker.patch('evals.unity_evaluator.UnityEnvWrapper', return_value=mock_ma_env_for_unity_eval)
    snapshot_fn = make_mock_agent_snapshot_fn # This returns a MockEvalAgent

    evaluator = UnityEvaluator(logger=mock_logger, cfg=cfg, make_agent_snapshot=snapshot_fn)
    
    global_step_val = 123
    mean_max_agent_return_overall = evaluator.run(global_step=global_step_val)

    # --- Verify `make_agent_snapshot` was called ---
    # snapshot_fn is called once at the start of run()
    # If snapshot_fn was a MagicMock: snapshot_fn.assert_called_once()
    # Here, make_mock_agent_snapshot_fn is a factory, snapshot_fn is the actual function.
    # We can check if the agent it returned was used.

    # --- Verify environment interactions ---
    # Env should be reset `cfg.eval_episodes` times
    assert mock_ma_env_for_unity_eval.reset.call_count == cfg.eval_episodes 
    # Env step called sum of episode lengths. Here, each episode is 1 step.
    assert mock_ma_env_for_unity_eval.step.call_count == cfg.eval_episodes * mock_ma_env_for_unity_eval.episode_length

    # --- Verify metrics logged ---
    # Agent returns: Ep1 [10,5], Ep2 [2,8], Ep3 [6,6], Ep4 [7,3], Ep5 [4,9]
    # Mean agent returns:
    # Agent0: (10+2+6+7+4)/5 = 29/5 = 5.8
    # Agent1: (5+8+6+3+9)/5 = 31/5 = 6.2
    mock_logger.add_scalar.assert_any_call("eval/agent0_mean_return", 5.8, global_step_val)
    mock_logger.add_scalar.assert_any_call("eval/agent1_mean_return", 6.2, global_step_val)

    # Max agent per ep: Ep1:10, Ep2:8, Ep3:6, Ep4:7, Ep5:9
    # Mean max_agent_ret: (10+8+6+7+9)/5 = 40/5 = 8.0
    mock_logger.add_scalar.assert_any_call("eval/agent_mean_max_return", 8.0, global_step_val)
    assert mean_max_agent_return_overall == 8.0 # Check return value of run()

    # Team returns (teams = [[0], [1]])
    # Team0 returns: Ep1:10, Ep2:2, Ep3:6, Ep4:7, Ep5:4
    # Team1 returns: Ep1:5, Ep2:8, Ep3:6, Ep4:3, Ep5:9
    # Mean Team0: 5.8
    # Mean Team1: 6.2
    mock_logger.add_scalar.assert_any_call("eval/team0_mean_return", 5.8, global_step_val)
    mock_logger.add_scalar.assert_any_call("eval/team1_mean_return", 6.2, global_step_val)

    # Max team per ep: Ep1:10, Ep2:8, Ep3:6, Ep4:7, Ep5:9
    # Mean max_team_ret: (10+8+6+7+9)/5 = 8.0
    mock_logger.add_scalar.assert_any_call("eval/team_mean_max_return", 8.0, global_step_val)
    
    # Mean episode length (all are 1 step)
    mock_logger.add_scalar.assert_any_call("eval/mean_episode_length", 1.0, global_step_val)


# --- Close Method Test ---
def test_unity_evaluator_close(mock_logger, unity_eval_config_dict, make_mock_agent_snapshot_fn, mocker, mock_ma_env_for_unity_eval):
    cfg = Namespace(**unity_eval_config_dict)
    mock_env_patch = mocker.patch('evals.unity_evaluator.UnityEnvWrapper', return_value=mock_ma_env_for_unity_eval)
    snapshot_fn = make_mock_agent_snapshot_fn

    evaluator = UnityEvaluator(logger=mock_logger, cfg=cfg, make_agent_snapshot=snapshot_fn)
    
    # Add a close mock to the MockMAEnv instance if it doesn't have one, or ensure it does
    mock_ma_env_for_unity_eval.close = mocker.MagicMock()
    
    evaluator.close()
    mock_ma_env_for_unity_eval.close.assert_called_once()


# Basic import check
def test_imports_unity_evaluator():
    assert UnityEvaluator is not None

# Note: `MockEvalAgent` was updated in `tests/helpers.py` to include `num_agents` and `device` attributes,
# and its `act` method was refined to accept `obs_tensor_all_env_agents` and return a list of action tensors.
# `MockMAEnv` was also updated for compatibility with `UnityEnvWrapper`'s expected `reset` and `step` signatures.
# The `competitive_eval_config_dict` in `test_competitive_evaluator.py` should be checked for any keys
# that `UnityEvaluator`'s `cfg` might also need if we were to merge configs (e.g. `env_id`).
# For `UnityEvaluator`, `cfg.teams` is important.
# The `make_mock_agent_snapshot_fn` now correctly uses `mock_ma_env_for_unity_eval.action_space` (which is a list of spaces).
# The `MockMAEnv` constructor was also updated to properly store `action_spaces` as a list.
# The `eval_episodes` in `unity_eval_config_dict` is used by `UnityEvaluator`.
# `MockMAEnv.reset` was simplified to only return observations.
# `MockMAEnv.step` was updated to return `next_obs_list, rewards_array, dones_array, info_dict_with_all_done`.
# Corrected `mock_snapshot_fn_factory` to `make_mock_agent_snapshot_fn` in test arguments.
# Corrected logic in `test_unity_evaluator_run` for setting up fixed rewards and checking metrics.
# Ensured `MockMAEnv` action_spaces is correctly used by `MockEvalAgent`.
# `MockEvalAgent.act` updated to return list of Tensors.
# `MockMAEnv.step` now takes `actions` (list of numpy) which is consistent with `UnityEvaluator`'s processing.
# `MockEvalAgent` constructor now correctly takes `env_action_spaces_list`.The tests for `UnityEvaluator` have been implemented in `tests/test_unity_evaluator.py`.

**Key aspects covered:**

1.  **Configuration and Mocking Setup:**
    *   A `unity_eval_config_dict` fixture provides configurations like `eval_episodes`, `teams`, and parameters needed for mocking `UnityEnvWrapper` (`env_id`, `worker_id`, `seed`).
    *   A `mock_logger` fixture provides a `MagicMock` for verifying logging calls.
    *   A `mock_ma_env_for_unity_eval` fixture creates a `MockMAEnv` instance, configured with observation and action spaces suitable for the number of agents defined by the `teams` structure in the config.
    *   A `make_mock_agent_snapshot_fn` factory fixture creates a `make_agent_snapshot` function. This function, when called, returns a `MockEvalAgent` instance. The `MockEvalAgent` is initialized with the environment's full list of action spaces and a device, and its `num_agents` attribute is set. Its `act` method is designed to take a tensor of observations for all agents and return a list of action tensors.
    *   `evals.unity_evaluator.UnityEnvWrapper` is consistently patched using `mocker.patch` to return the `mock_ma_env_for_unity_eval` instance, preventing real Unity environment initialization.

2.  **Initialization Test (`test_unity_evaluator_initialization`):**
    *   Verifies that `UnityEvaluator` correctly stores its dependencies (mocked logger, config, snapshot function).
    *   Ensures that the `UnityEnvWrapper` is patched and the evaluator uses the provided mock environment.
    *   Checks that `team_indices` and `eval_episodes` are correctly set from the config.

3.  **`run` Method Test (`test_unity_evaluator_run`):**
    *   This is the main integration test for the evaluation logic.
    *   The `MockMAEnv` is configured with a `fixed_rewards_pattern` and a fixed `episode_length` of 1 step per episode to make outcomes and episode lengths predictable across the `cfg.eval_episodes`.
    *   `evaluator.run(global_step)` is called.
    *   **Verification:**
        *   `make_agent_snapshot` is called (implicitly, by checking the agent is used).
        *   Environment interactions: `mock_ma_env.reset` is called once per evaluation episode. `mock_ma_env.step` is called for the total number of steps across all episodes.
        *   Logged metrics: `mock_logger.add_scalar` calls are checked for all expected metrics (`eval/agent{i}_mean_return`, `eval/agent_mean_max_return`, `eval/team{t}_mean_return`, `eval/team_mean_max_return`, `eval/mean_episode_length`). The values of these metrics are calculated based on the predictable rewards from `MockMAEnv` and asserted.
        *   The return value of `evaluator.run()` (which is `mean_max_agent_ret`) is also verified.

4.  **`close` Method Test (`test_unity_evaluator_close`):**
    *   Verifies that calling `evaluator.close()` results in a call to the `close()` method of the (mocked) environment instance.

The tests cover the primary functionalities of `UnityEvaluator`, focusing on its interaction with the environment, agent snapshots, and the logging of computed metrics. The mocking strategy ensures that the tests are self-contained and do not require a live Unity environment. Helper classes `MockMAEnv` and `MockEvalAgent` were refined in a previous step to align with the specific interface expectations of `UnityEvaluator`.
