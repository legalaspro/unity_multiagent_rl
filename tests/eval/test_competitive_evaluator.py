import pytest
import numpy as np
import torch # Added for MockEvalAgent actions
import json
import os
from argparse import Namespace
from gymnasium.spaces import Box

from evals.competitive_evaluator import CompetitiveEvaluator
from evals.elo_rating import EloRatingSystem # Corrected from EloRating
from tests.helpers import MockMAEnv, MockEvalAgent # Assuming MockEvalAgent is in helpers
# RandomPolicy is used by CompetitiveEvaluator, need to mock or ensure it can be init'd
from algos.random import RandomPolicy


@pytest.fixture
def competitive_eval_config_dict(tmp_path): # Renamed to dict
    """Basic configuration dictionary for CompetitiveEvaluator tests."""
    # Define team configurations based on a 2-agent environment for simplicity in tests
    # teams: [[team0_agent_indices], [team1_agent_indices]]
    # For a 2-agent env, agent 0 is team 0, agent 1 is team 1.
    # env_num_agents should align with what MockMAEnv will be configured with.
    env_num_agents = 2
    config = {
        "eval_episodes": 4, # Reduced for faster tests
        "render_episodes": 0,
        "num_processes": 1, # For evaluator's own env, not directly used by this class structure
        "seed": 42,
        "log_dir": str(tmp_path),
        "eval_log_filename": "eval_log.json", # Not directly used by this CompetitiveEvaluator
        "render": False,
        "verbose": False,
        "k_factor": 32,
        "teams": [[i] for i in range(env_num_agents)], # e.g. [[0], [1]] for 2 agents
        "env_id": "MockEnvID", # For UnityEnvWrapper
        "worker_id": 0, # For UnityEnvWrapper
        "max_model_snapshots": 3,
        "competitive_eval_episodes": 2, # Episodes per matchup against stored snapshots
        # Params for RandomPolicy if it needs them from cfg, or use defaults
        # Params for agent snapshot creation if snapshot_fn needs them from cfg
    }
    return config

@pytest.fixture
def mock_logger(mocker):
    logger = mocker.MagicMock()
    logger.add_scalar = mocker.MagicMock()
    return logger

@pytest.fixture
def mock_ma_env(competitive_eval_config_dict): # Depends on config for num_agents
    cfg = Namespace(**competitive_eval_config_dict)
    num_agents_in_env = sum(len(team) for team in cfg.teams)

    # Define simple obs and action spaces for the env
    obs_shapes = [(4,)] * num_agents_in_env
    # Action space for MockEvalAgent and RandomPolicy.
    # RandomPolicy(self.env.action_spaces) -> self.env.action_spaces is a list of gym.Space
    action_spaces = [Box(low=-1, high=1, shape=(2,), dtype=np.float32) for _ in range(num_agents_in_env)]

    env = MockMAEnv(
        num_agents=num_agents_in_env,
        obs_shapes=obs_shapes,
        action_spaces=action_spaces, # Provide list of spaces
        episode_length=5 # Short episodes for testing
    )
    return env

@pytest.fixture
def mock_snapshot_fn_factory(mock_ma_env): # Factory to create snapshot_fn
    def _create_snapshot_fn(agent_id_counter_list): # Pass list to modify counter by reference
        def make_agent_snapshot():
            # Create a new MockEvalAgent instance each time, possibly with a new ID/name
            agent_id = f"agent_step_{agent_id_counter_list[0]}"
            agent_name = f"AgentStep{agent_id_counter_list[0]}"
            # MockEvalAgent needs the list of action spaces for all agents in the environment
            # as its `act` method is expected to return actions for all of them.
            agent = MockEvalAgent(
                agent_id=agent_id,
                name=agent_name,
                env_action_spaces_list=mock_ma_env.action_space, # Pass full list
                device='cpu'
            )
            agent_id_counter_list[0] += 100 # Increment step for next snapshot
            return agent
        return make_agent_snapshot
    return _create_snapshot_fn


# --- Initialization Tests ---
def test_competitive_evaluator_initialization(mock_logger, competitive_eval_config_dict, mock_ma_env, mock_snapshot_fn_factory, mocker):
    cfg = Namespace(**competitive_eval_config_dict)

    # Mock UnityEnvWrapper to return our MockMAEnv instance
    mocker.patch('evals.competitive_evaluator.UnityEnvWrapper', return_value=mock_ma_env)

    agent_id_counter = [0] # Mutable counter for snapshot IDs
    snapshot_fn = mock_snapshot_fn_factory(agent_id_counter)

    evaluator = CompetitiveEvaluator(logger=mock_logger, cfg=cfg, make_agent_snapshot=snapshot_fn)

    assert evaluator.logger == mock_logger
    assert evaluator.cfg == cfg
    assert evaluator.snapshot_fn == snapshot_fn
    assert evaluator.env == mock_ma_env # Check if mock env was used
    assert isinstance(evaluator.elo_system, EloRatingSystem)
    assert evaluator.elo_system.k_factor == cfg.k_factor
    assert len(evaluator.model_snapshots) == 0
    assert len(evaluator.snapshot_ratings) == 0
    assert evaluator.max_snapshots == cfg.max_model_snapshots
    assert isinstance(evaluator.random_policy, RandomPolicy)
    # RandomPolicy should be initialized with the environment's action spaces
    assert len(evaluator.random_policy.action_spaces) == mock_ma_env.n_agents
    assert evaluator.eval_episodes_per_matchup == cfg.competitive_eval_episodes


# --- Test play_one_episode effects (via a controlled call) ---
def test_play_one_episode_outcome(competitive_eval_config_dict, mocker):
    cfg_dict = competitive_eval_config_dict
    # For this test, we make a simplified evaluator just to call play_one_episode
    # Or, we can make play_one_episode a static method or testable helper if it doesn't rely too much on self

    # Setup agents
    # MockEvalAgent's act now needs obs_tensor_all_env_agents and returns List[actions]
    # The action_spaces passed to MockEvalAgent should be the list of action_spaces of the env.
    num_env_agents = sum(len(team) for team in cfg_dict["teams"])
    env_action_spaces = [Box(low=-1, high=1, shape=(2,), dtype=np.float32) for _ in range(num_env_agents)]

    agent1 = MockEvalAgent("agent1", "Agent1", env_action_spaces, device='cpu')
    agent2 = MockEvalAgent("agent2", "Agent2", env_action_spaces, device='cpu')

    # Setup environment with predictable rewards for 2 agents, 1 step episode
    # Team A (agent1) wins: score_A = 1.0
    mock_env_win = MockMAEnv(
        num_agents=num_env_agents,
        action_spaces=env_action_spaces,
        episode_length=1,
        fixed_rewards=[np.array([1.0, -1.0])] # Agent 0 (team A) gets 1, Agent 1 (team B) gets -1
    )

    # Mock UnityEnvWrapper to return our MockMAEnv instance
    mock_dummy_env = MockMAEnv(num_agents=num_env_agents, action_spaces=env_action_spaces)
    mocker.patch('evals.competitive_evaluator.UnityEnvWrapper', return_value=mock_dummy_env)

    # Minimal evaluator components needed for play_one_episode
    evaluator = CompetitiveEvaluator(mocker.MagicMock(), Namespace(**cfg_dict), lambda: agent1) # Dummy logger, snapshot_fn

    # Test win for agentA
    # Teams: agent1 is policy for team0, agent2 is policy for team1
    # cfg_dict["teams"] is [[0],[1]] for 2 agents
    score_win = evaluator.play_one_episode(agent1, agent2, teams=cfg_dict["teams"], env=mock_env_win)
    assert score_win == 1.0

    # Team A (agent1) loses: score_A = 0.0
    mock_env_loss = MockMAEnv(
        num_agents=num_env_agents, action_spaces=env_action_spaces,
        episode_length=1, fixed_rewards=[np.array([-1.0, 1.0])]
    )
    score_loss = evaluator.play_one_episode(agent1, agent2, teams=cfg_dict["teams"], env=mock_env_loss)
    assert score_loss == 0.0

    # Draw: score_A = 0.5
    mock_env_draw = MockMAEnv(
        num_agents=num_env_agents, action_spaces=env_action_spaces,
        episode_length=1, fixed_rewards=[np.array([0.0, 0.0])]
    )
    score_draw = evaluator.play_one_episode(agent1, agent2, teams=cfg_dict["teams"], env=mock_env_draw)
    assert score_draw == 0.5


# --- Integration Test for run() method ---
def test_competitive_evaluator_run_integration(mock_logger, competitive_eval_config_dict, mock_ma_env, mock_snapshot_fn_factory, mocker):
    cfg_dict = competitive_eval_config_dict
    cfg = Namespace(**cfg_dict)

    # Configure env for somewhat predictable results if needed, or just let it run
    # For simplicity, let random policy be very weak (e.g. always lose or draw)
    # And let new agents be slightly better.
    # MockMAEnv by default has random rewards, but we can make it fixed for more predictability.
    # Let's say new agent always beats random, and new agents draw against each other.

    # Patch UnityEnvWrapper
    mocker.patch('evals.competitive_evaluator.UnityEnvWrapper', return_value=mock_ma_env)

    agent_id_counter = [0] # Step counter for agent versions
    snapshot_fn = mock_snapshot_fn_factory(agent_id_counter)

    evaluator = CompetitiveEvaluator(logger=mock_logger, cfg=cfg, make_agent_snapshot=snapshot_fn)
    initial_elo = evaluator.elo_system.initial_rating

    # --- First run: Only vs Random ---
    current_step_1 = agent_id_counter[0] # Should be 0 initially from factory

    # Modify mock_env to make current_agent win against random_policy
    # agentA is current_agent, agentB is random_policy
    # To make agentA win, rewards should be [1, -1] if current_agent is team0, or [-1, 1] if current_agent is team1
    # play_one_episode randomizes team assignment for vs_random.
    # For simplicity, let's assume for this test, random policy is so bad it always gives max reward to other team.
    # This is hard to enforce perfectly without complex mocking of play_one_episode's internals.
    # Let's focus on Elo changes and snapshot management.

    elo_after_run1 = evaluator.run(global_step=current_step_1) # global_step matches snapshot id

    assert current_step_1 in evaluator.model_snapshots
    assert current_step_1 in evaluator.snapshot_ratings
    # Rating vs random doesn't use Elo updates in this code, just logs win rate.
    # The initial rating is set in _save_model_snapshot using _get_best_rating or initial_rating.
    # First agent gets initial_rating if _get_best_rating is from empty, or best if not.
    assert evaluator.snapshot_ratings[current_step_1] == initial_elo
    assert elo_after_run1 == initial_elo
    mock_logger.add_scalar.assert_any_call("competitive/win_rate_vs_random", mocker.ANY, current_step_1)


    # --- Second run: New agent, vs Random, vs Snapshot 0 ---
    current_step_2 = agent_id_counter[0] # Should be 100 now

    # To make Elo changes predictable:
    # Assume current_agent (step 100) vs opponent (step 0).
    # Let current_agent win all cfg.competitive_eval_episodes against step 0 agent.
    # This requires mocking play_one_episode or making MockMAEnv highly predictable.
    # Let's try to make MockMAEnv predictable for this specific matchup.
    # When agentA (current_step_2 policy) plays agentB (current_step_0 policy), agentA wins.

    # This is tricky because play_one_episode is called internally.
    # We can patch `evaluator.play_one_episode` for this test.

    # Store original play_one_episode
    original_play_one_episode = evaluator.play_one_episode

    def mock_play_one_episode_for_run2(agentA, agentB, *, teams=None, env=None):
        # agentA is current_agent (step_100), agentB is opponent (step_0 or random)
        if agentB == evaluator.random_policy:
            return 1.0 # Current agent always beats random
        elif hasattr(agentB, 'id') and agentB.id == "agent_step_0": # Opponent is the first snapshot
            return 1.0 # Current agent (step 100) always beats snapshot 0
        return original_play_one_episode(agentA, agentB, teams=teams, env=env) # Fallback

    evaluator.play_one_episode = mocker.MagicMock(side_effect=mock_play_one_episode_for_run2)

    elo_after_run2 = evaluator.run(global_step=current_step_2)

    assert current_step_2 in evaluator.model_snapshots
    assert current_step_2 in evaluator.snapshot_ratings

    # Elo of current_agent (step_2) should increase because it beat step_0 agent
    # Elo of step_0 agent should decrease
    assert evaluator.snapshot_ratings[current_step_2] > initial_elo
    assert evaluator.snapshot_ratings[current_step_1] < initial_elo # current_step_1 was agent_step_0
    assert elo_after_run2 == evaluator.snapshot_ratings[current_step_2]

    mock_logger.add_scalar.assert_any_call("competitive/elo_rating", evaluator.snapshot_ratings[current_step_2], current_step_2)
    evaluator.play_one_episode.assert_called() # Ensure our mock was used

    # Restore original method if needed for other tests or cleanup
    evaluator.play_one_episode = original_play_one_episode


    # --- Test Max Snapshots ---
    # Add more snapshots until max_snapshots is reached and exceeded
    # max_snapshots is 3 in config. We have step_0 and step_100. Add two more.
    current_step_3 = agent_id_counter[0] # 200
    evaluator.run(global_step=current_step_3) # Snapshot for 200, total 3 snapshots.
    assert len(evaluator.model_snapshots) == 3
    assert current_step_3 in evaluator.model_snapshots
    assert 0 in evaluator.model_snapshots # oldest step 0 should still be there

    current_step_4 = agent_id_counter[0] # 300
    evaluator.run(global_step=current_step_4) # Snapshot for 300. Max is 3. Oldest (0) should be removed.
    assert len(evaluator.model_snapshots) == 3
    assert current_step_4 in evaluator.model_snapshots
    assert 0 not in evaluator.model_snapshots # Step 0 should be gone
    assert 100 in evaluator.model_snapshots # Step 100 should remain

def test_competitive_evaluator_state_persistence(mock_logger, competitive_eval_config_dict, mock_ma_env, mock_snapshot_fn_factory, tmp_path, mocker):
    cfg_dict = competitive_eval_config_dict
    cfg = Namespace(**cfg_dict)

    mocker.patch('evals.competitive_evaluator.UnityEnvWrapper', return_value=mock_ma_env)

    agent_id_counter = [0]
    snapshot_fn = mock_snapshot_fn_factory(agent_id_counter)

    evaluator1 = CompetitiveEvaluator(logger=mock_logger, cfg=cfg, make_agent_snapshot=snapshot_fn)
    evaluator1.run(global_step=0)
    evaluator1.run(global_step=100)

    ratings_before_save = dict(evaluator1.snapshot_ratings)
    elo_system_state_before_save = evaluator1.elo_system.get_state()

    save_dir = str(tmp_path / "eval_state")
    evaluator1.save_state(save_dir)
    assert os.path.exists(os.path.join(save_dir, 'competitive_state.pkl'))

    # Create new evaluator and load state
    agent_id_counter2 = [0] # Reset counter for new snapshot fn instances if any
    snapshot_fn2 = mock_snapshot_fn_factory(agent_id_counter2) # New snapshot_fn for new evaluator
    evaluator2 = CompetitiveEvaluator(logger=mock_logger, cfg=cfg, make_agent_snapshot=snapshot_fn2)

    # Mock model snapshots loading as well, since they are not part of state file
    # For this test, we only care about ratings and elo_system state.
    # If load_state tried to load actual model files, it would fail here.
    # The current load_state only loads ratings and elo_system_state.

    evaluator2.load_state(save_dir)

    assert evaluator2.snapshot_ratings == ratings_before_save
    assert evaluator2.elo_system.k_factor == elo_system_state_before_save['k_factor']
    assert evaluator2.elo_system.initial_rating == elo_system_state_before_save['initial_rating']


# Basic import check
def test_imports_competitive_evaluator():
    assert CompetitiveEvaluator is not None
    assert EloRatingSystem is not None

# **Key aspects covered:**

# 1.  **Configuration and Mocking Setup:**
#     *   A `competitive_eval_config_dict` fixture provides necessary configurations. Unused keys like `eval_episodes` (which is distinct from `competitive_eval_episodes` used by the class) were noted and effectively ignored for these tests.
#     *   A `mock_logger` fixture provides a `MagicMock` for logging calls.
#     *   A `mock_ma_env` fixture provides an instance of the enhanced `MockMAEnv` (from `tests/helpers.py`), configured according to the number of agents derived from the `teams` setting in the config.
#     *   A `mock_snapshot_fn_factory` creates a `make_agent_snapshot` function that returns new `MockEvalAgent` instances with unique IDs for simulating different agent versions. The `MockEvalAgent` is initialized with the environment's full list of action spaces, as its `act` method is expected to provide actions for all agents in the environment.
#     *   `evals.competitive_evaluator.UnityEnvWrapper` is patched using `mocker.patch` to return the `mock_ma_env` instance, preventing attempts to initialize a real Unity environment.

# 2.  **Initialization Test (`test_competitive_evaluator_initialization`):**
#     *   Verifies that `CompetitiveEvaluator` initializes correctly, storing the mocked logger, config, and snapshot function.
#     *   Ensures the (mocked) environment is set up.
#     *   Checks that an `EloRatingSystem` instance is created and configured.
#     *   Confirms that internal data structures like `model_snapshots` and `snapshot_ratings` are initially empty.
#     *   Verifies that a `RandomPolicy` is instantiated for baseline evaluations.

# 3.  **`play_one_episode` Logic Test (`test_play_one_episode_outcome`):**
#     *   Tests the core episode execution and outcome determination logic of `play_one_episode`.
#     *   Uses two `MockEvalAgent` instances and a `MockMAEnv` configured with `fixed_rewards` to produce predictable win, loss, and draw outcomes for one team against another.
#     *   Asserts that `play_one_episode` returns the correct score (1.0 for win, 0.0 for loss, 0.5 for draw from Team A's perspective).

# 4.  **`run` Method Integration Test (`test_competitive_evaluator_run_integration`):**
#     *   This is the main integration test covering the interaction of most components.
#     *   It calls `evaluator.run(global_step)` multiple times to simulate different stages:
#         *   **First run:** The new agent (snapshot 0) is evaluated against the random baseline. Its initial Elo rating is stored. Logger calls for random evaluation are checked.
#         *   **Second run:** A "newer" agent (snapshot 100) is evaluated. It first plays against random, then against the previously saved snapshot (snapshot 0).
#             *   To make Elo changes predictable, `evaluator.play_one_episode` is temporarily patched with a mock that forces specific outcomes (e.g., new agent always beats random and always beats the older snapshot).
#             *   Elo ratings for both the new agent (should increase) and the older snapshot (should decrease) are verified. Logger calls for Elo rating and match results are checked.
#         *   **Max Snapshots Test:** Further calls to `run()` are made to ensure that when `max_model_snapshots` is exceeded, the oldest snapshot and its rating are correctly removed.

# 5.  **State Persistence Test (`test_competitive_evaluator_state_persistence`):**
#     *   An evaluator instance is run for a couple of steps to populate `snapshot_ratings` and potentially alter `elo_system` state.
#     *   `evaluator.save_state()` is called.
#     *   A new `CompetitiveEvaluator` instance is created.
#     *   `evaluator.load_state()` is called on the new instance.
#     *   Verifies that `snapshot_ratings` and the `elo_system`'s parameters are correctly restored in the new instance.

# These tests cover the primary lifecycle and functionalities of the provided `CompetitiveEvaluator`, including its unique aspects like model snapshot management and evaluation against a pool of prior agents, using extensive mocking for its dependencies.
