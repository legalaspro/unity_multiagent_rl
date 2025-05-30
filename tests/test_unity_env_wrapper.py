import pytest
import numpy as np
import torch
from gymnasium.spaces import Box, Discrete, Tuple

try:
    from envs.unity_env_wrapper import UnityEnvWrapper
except ImportError:
    UnityEnvWrapper = None

# Using the mocks defined in the same file in the previous step for brevity.
# In a real setup, these would be imported from tests.mocks or similar.

class MockUnityEnvironmentException(Exception):
    pass

class MockBrainParameters:
    def __init__(self, brain_name, vector_observation_space_size, num_stacked_vector_observations,
                 camera_resolutions, number_visual_observations,
                 vector_action_space_size, vector_action_descriptions,
                 vector_action_space_type):
        self.brain_name = brain_name
        self.vector_observation_space_size = vector_observation_space_size
        self.num_stacked_vector_observations = num_stacked_vector_observations
        self.camera_resolutions = camera_resolutions
        self.number_visual_observations = len(camera_resolutions) if camera_resolutions else 0 # Corrected
        self.vector_action_space_size = vector_action_space_size
        self.vector_action_descriptions = vector_action_descriptions
        self.vector_action_space_type = vector_action_space_type

class MockBrainInfo:
    def __init__(self, visual_obs, vector_obs, rewards, agents, local_done, max_reached, local_rewards=None):
        self.visual_observations = visual_obs
        self.vector_observations = vector_obs
        self.rewards = rewards
        self.agents = agents # List of unique agent IDs for this brain's current step
        self.local_done = local_done
        self.max_reached = max_reached
        self.local_rewards = local_rewards if local_rewards is not None else rewards

class MockCommunicator:
    def __init__(self, brain_params_list, num_agents_per_brain_list, worker_id=0):
        self.brain_params_list = brain_params_list
        self.num_agents_per_brain_list = num_agents_per_brain_list # Renamed for clarity
        self.worker_id = worker_id
        self._is_closed = False
        self.train_mode = True

        self.brains = {bp.brain_name: bp for bp in brain_params_list}
        
        self.agent_id_to_brain_name = {}
        self.current_agent_idx = 0
        self.brain_agent_ids = {} # brain_name -> list of agent_ids for that brain at current step

        # Create all unique agent_ids that this communicator will ever report
        self.all_managed_agent_ids = []
        for i, (bp, num_agents) in enumerate(zip(brain_params_list, num_agents_per_brain_list)):
            brain_specific_ids = []
            for j in range(num_agents):
                agent_id = f"brain{i}_agent{j}"
                brain_specific_ids.append(agent_id)
                self.all_managed_agent_ids.append(agent_id)
                self.agent_id_to_brain_name[agent_id] = bp.brain_name
            self.brain_agent_ids[bp.brain_name] = brain_specific_ids # Store the fixed set of agent IDs per brain

    def _get_brain_info_for_brain(self, brain_param, num_agents_in_brain, is_reset=False):
        vec_obs_size = brain_param.vector_observation_space_size * brain_param.num_stacked_vector_observations
        vec_obs = np.random.randn(num_agents_in_brain, vec_obs_size).astype(np.float32) if vec_obs_size > 0 else np.array([]).reshape(num_agents_in_brain,0)
        
        vis_obs_list = []
        if brain_param.number_visual_observations > 0:
            for res in brain_param.camera_resolutions:
                vis_obs_shape = (num_agents_in_brain, res['height'], res['width'], res['num_channels'])
                vis_obs = np.random.randint(0, 256, size=vis_obs_shape, dtype=np.uint8) # uint8 for visual
                vis_obs_list.append(vis_obs)
        
        rewards = [0.0 if is_reset else np.random.rand() for _ in range(num_agents_in_brain)]
        local_done = [False if is_reset else np.random.rand() > 0.9 for _ in range(num_agents_in_brain)]
        max_reached = local_done[:] # Simplification
        
        # Use the fixed agent IDs for this brain
        current_brain_agent_ids = self.brain_agent_ids[brain_param.brain_name]

        return MockBrainInfo(
            visual_obs=vis_obs_list, vector_obs=vec_obs, rewards=rewards,
            agents=current_brain_agent_ids, local_done=local_done, max_reached=max_reached
        )

    def initialize(self, inputs):
        all_brain_info = {}
        for i, bp in enumerate(self.brain_params_list):
            num_agents = self.num_agents_per_brain_list[i]
            all_brain_info[bp.brain_name] = self._get_brain_info_for_brain(bp, num_agents, is_reset=True)
        return all_brain_info

    def exchange(self, inputs):
        if self._is_closed: raise MockUnityEnvironmentException("Communicator is closed.")
        all_brain_info = {}
        for i, bp in enumerate(self.brain_params_list):
            num_agents = self.num_agents_per_brain_list[i]
            all_brain_info[bp.brain_name] = self._get_brain_info_for_brain(bp, num_agents, is_reset=False)
        return all_brain_info

    def close(self):
        self._is_closed = True

    @property
    def number_external_brains(self):
        return len(self.brain_params_list)
    
    @property
    def is_closed(self): # For test assertion
        return self._is_closed


# --- Pytest Fixtures ---
@pytest.fixture
def mock_communicator_params_vec_cont(): # Single brain, 2 agents, vector obs, continuous actions
    bp = MockBrainParameters("VecContBrain", 5, 1, [], 0, [3], [""]*3, 0)
    return {"brain_params_list": [bp], "num_agents_per_brain_list": [2], "use_visual_wrapper": False}

@pytest.fixture
def mock_communicator_params_vis_disc(): # Single brain, 1 agent, visual obs, discrete actions
    bp = MockBrainParameters("VisDiscBrain", 0, 0, [{'width': 84, 'height': 84, 'num_channels': 3}], 1, [5], [""]*5, 1)
    return {"brain_params_list": [bp], "num_agents_per_brain_list": [1], "use_visual_wrapper": True}

@pytest.fixture
def mock_communicator_params_multi_brain_mixed(): # Multi-brain, mixed types
    bp0 = MockBrainParameters("MBrainC", 3, 1, [], 0, [2], [""]*2, 0) # 1 agent, Cont Act
    bp1 = MockBrainParameters("MBrainD", 0, 0, [{'width':64,'height':64,'num_channels':1}],1, [3,2], [""]*2,1) # 2 agents, Disc Act (branched)
    return {"brain_params_list": [bp0, bp1], "num_agents_per_brain_list": [1, 2], "use_visual_wrapper": True}


@pytest.fixture
def mock_communicator(request):
    params_dict = request.param
    communicator = MockCommunicator(
        brain_params_list=params_dict["brain_params_list"],
        num_agents_per_brain_list=params_dict["num_agents_per_brain_list"]
    )
    yield communicator # Test will use this
    if not communicator.is_closed: # Ensure cleanup
        communicator.close()

@pytest.fixture
def unity_env_wrapper(mock_communicator, request): # request can be used to get params for wrapper if needed
    params_dict = request.param # This will be the same dict as passed to mock_communicator
    env = UnityEnvWrapper(
        communicator=mock_communicator, # Pass the mock communicator directly
        worker_id=0,
        use_visual=params_dict.get("use_visual_wrapper", False), # Set based on test param
        uint8_visual=False, # Default for tests, can be overridden
        allow_multiple_obs=True, # Usually True for flexibility with brains
        flatten_branched=params_dict.get("flatten_branched_wrapper", False) # Set based on test param
    )
    yield env
    env.close()

# --- Test Cases ---
@pytest.mark.skipif(UnityEnvWrapper is None, reason="UnityEnvWrapper not available")
@pytest.mark.parametrize(
    "mock_communicator, unity_env_wrapper", 
    [("mock_communicator_params_vec_cont", "mock_communicator_params_vec_cont"), 
     ("mock_communicator_params_vis_disc", "mock_communicator_params_vis_disc"),
     ("mock_communicator_params_multi_brain_mixed", "mock_communicator_params_multi_brain_mixed")],
    indirect=["mock_communicator", "unity_env_wrapper"] # Tells pytest these are fixture names
)
def test_wrapper_initialization(unity_env_wrapper, mock_communicator):
    env = unity_env_wrapper
    comm = mock_communicator

    assert env.num_agents == sum(comm.num_agents_per_brain_list)
    assert len(env.observation_space) == env.num_agents
    assert len(env.action_space) == env.num_agents
    assert env.brain_names == [bp.brain_name for bp in comm.brain_params_list]
    assert env.external_brain_names == env.brain_names # Assuming all brains are external for mock

    agent_idx_offset = 0
    for i, brain_param in enumerate(comm.brain_params_list):
        num_agents_in_brain = comm.num_agents_per_brain_list[i]
        for j in range(num_agents_in_brain):
            global_agent_idx = agent_idx_offset + j
            
            # Observation Space Check
            expected_obs_space = None
            if brain_param.number_visual_observations > 0 and env.use_visual:
                # Assuming allow_multiple_obs=True and first visual obs is primary if no vector
                # And uint8_visual=False, so HWC * stack. Stack is 1 for visual in wrapper by default.
                # Wrapper stacks visual obs: (H, W, C * num_visual_obs * num_stacked_vec_obs??) - this is complex.
                # Simpler: wrapper default is (H,W,C) for one visual. If stack, it applies.
                # If allow_multiple_obs is False, it prioritizes.
                # For now, let's assume a simple case where wrapper extracts one primary obs.
                # If visual, shape is (H, W, C) * num_stacked_vector_observations (which is usually 1 for visual)
                # The num_stacked_vector_observations is misnamed if it applies to visual.
                # The wrapper concatenates visual observations along channel axis.
                total_channels = 0
                for res in brain_param.camera_resolutions:
                    total_channels += res['num_channels']
                
                if total_channels > 0 :
                    # If vector obs also present and allow_multiple_obs, obs_space is Tuple
                    if brain_param.vector_observation_space_size > 0 and env.allow_multiple_obs:
                         assert isinstance(env.observation_space[global_agent_idx], Tuple)
                         # Further checks for tuple components can be added.
                    else: # Only visual (or visual prioritized)
                        res = brain_param.camera_resolutions[0] # Taking first one for shape check
                        expected_shape = (res['height'], res['width'], total_channels)
                        assert isinstance(env.observation_space[global_agent_idx], Box)
                        assert env.observation_space[global_agent_idx].shape == expected_shape
                        assert env.observation_space[global_agent_idx].dtype == np.uint8 if env.uint8_visual else np.float32
            
            elif brain_param.vector_observation_space_size > 0:
                expected_shape = (brain_param.vector_observation_space_size * brain_param.num_stacked_vector_observations,)
                assert isinstance(env.observation_space[global_agent_idx], Box)
                assert env.observation_space[global_agent_idx].shape == expected_shape
                assert env.observation_space[global_agent_idx].dtype == np.float32
            
            # Action Space Check
            if brain_param.vector_action_space_type == 0: # Continuous
                assert isinstance(env.action_space[global_agent_idx], Box)
                assert env.action_space[global_agent_idx].shape == (brain_param.vector_action_space_size[0],)
            else: # Discrete
                if env.flatten_branched or len(brain_param.vector_action_space_size) == 1:
                    assert isinstance(env.action_space[global_agent_idx], Discrete)
                    assert env.action_space[global_agent_idx].n == brain_param.vector_action_space_size[0] # Assuming single branch for this check
                else: # Branched
                    assert isinstance(env.action_space[global_agent_idx], Tuple)
                    assert len(env.action_space[global_agent_idx].spaces) == len(brain_param.vector_action_space_size)
                    for k, branch_size in enumerate(brain_param.vector_action_space_size):
                        assert isinstance(env.action_space[global_agent_idx].spaces[k], Discrete)
                        assert env.action_space[global_agent_idx].spaces[k].n == branch_size
        agent_idx_offset += num_agents_in_brain


@pytest.mark.skipif(UnityEnvWrapper is None, reason="UnityEnvWrapper not available")
@pytest.mark.parametrize(
    "mock_communicator, unity_env_wrapper",
    [("mock_communicator_params_vec_cont", "mock_communicator_params_vec_cont"),
     ("mock_communicator_params_vis_disc", {"brain_params_list": mock_communicator_params_vis_disc()["brain_params_list"], 
                                            "num_agents_per_brain_list": mock_communicator_params_vis_disc()["num_agents_per_brain_list"],
                                            "use_visual_wrapper": True}) # Pass use_visual to wrapper fixture
    ],
    indirect=["mock_communicator", "unity_env_wrapper"]
)
def test_wrapper_reset(unity_env_wrapper, mock_communicator):
    env = unity_env_wrapper
    obs_list = env.reset()

    assert isinstance(obs_list, list)
    assert len(obs_list) == env.num_agents
    for i in range(env.num_agents):
        assert isinstance(obs_list[i], np.ndarray)
        if isinstance(env.observation_space[i], Box):
            assert obs_list[i].shape == env.observation_space[i].shape
            assert obs_list[i].dtype == env.observation_space[i].dtype
        elif isinstance(env.observation_space[i], Tuple): # For allow_multiple_obs with mixed types
            assert isinstance(obs_list[i], tuple) # Wrapper might return tuple of obs components
            # Further checks for tuple components can be added here
    
    # Test train_mode
    env.reset(train_mode=False)
    assert not env.communicator.train_mode # Assuming communicator stores this state for checking

    env.reset(train_mode=True)
    assert env.communicator.train_mode


@pytest.mark.skipif(UnityEnvWrapper is None, reason="UnityEnvWrapper not available")
@pytest.mark.parametrize(
    "mock_communicator, unity_env_wrapper",
    [("mock_communicator_params_vec_cont", "mock_communicator_params_vec_cont"),
     ("mock_communicator_params_multi_brain_mixed", {"brain_params_list": mock_communicator_params_multi_brain_mixed()["brain_params_list"], 
                                                     "num_agents_per_brain_list": mock_communicator_params_multi_brain_mixed()["num_agents_per_brain_list"],
                                                     "use_visual_wrapper": True, "flatten_branched_wrapper": False})
    ],
    indirect=["mock_communicator", "unity_env_wrapper"]
)
def test_wrapper_step(unity_env_wrapper, mock_communicator):
    env = unity_env_wrapper
    env.reset() # Initial reset

    # Generate dummy actions
    actions = []
    for i in range(env.num_agents):
        actions.append(env.action_space[i].sample())
    
    # For multi-discrete (branched) actions, sample() returns a tuple.
    # The wrapper's step input expects a list of numbers for each agent if flattened,
    # or potentially a list of tuples if not flattened.
    # The BrainInfo processing in wrapper handles this.
    # For testing, we provide what action_space.sample() gives.

    next_obs, rewards, dones, info = env.step(actions)

    assert isinstance(next_obs, list)
    assert len(next_obs) == env.num_agents
    for i in range(env.num_agents):
        assert isinstance(next_obs[i], np.ndarray)
        if isinstance(env.observation_space[i], Box):
             assert next_obs[i].shape == env.observation_space[i].shape
        # Add checks for Tuple obs if allow_multiple_obs

    assert isinstance(rewards, np.ndarray)
    assert rewards.shape == (env.num_agents,)
    assert rewards.dtype == np.float32

    assert isinstance(dones, np.ndarray)
    assert dones.shape == (env.num_agents,)
    assert dones.dtype == bool

    assert isinstance(info, dict)
    assert 'all_done' in info
    assert isinstance(info['all_done'], bool)

    # Test if all_done is True when mock communicator makes all agents done
    # This requires configuring MockCommunicator's local_done behavior for a specific step.
    # For simplicity, we can assume if any agent is done, all_done might be true, or if ALL are done.
    # The wrapper sets info['all_done'] = all(dones_list).
    # So, if mock_communicator.exchange makes all local_done True for all brains:
    
    # Temporarily rig MockCommunicator to make all agents done
    original_exchange = mock_communicator.exchange
    def make_all_done_exchange(inputs):
        all_brain_info = original_exchange(inputs)
        for brain_name in all_brain_info:
            num_agents_in_brain = len(all_brain_info[brain_name].agents)
            all_brain_info[brain_name].local_done = [True] * num_agents_in_brain
        return all_brain_info
    mock_communicator.exchange = make_all_done_exchange
    
    _, _, dones_all_true, info_all_true = env.step(actions)
    assert info_all_true['all_done'] is True
    assert np.all(dones_all_true) is True
    mock_communicator.exchange = original_exchange # Restore


@pytest.mark.skipif(UnityEnvWrapper is None, reason="UnityEnvWrapper not available")
@pytest.mark.parametrize(
    "mock_communicator, unity_env_wrapper",
    [("mock_communicator_params_vec_cont", "mock_communicator_params_vec_cont")],
    indirect=["mock_communicator", "unity_env_wrapper"]
)
def test_wrapper_close(unity_env_wrapper, mock_communicator):
    env = unity_env_wrapper
    env.close()
    assert mock_communicator.is_closed # Check property on mock

@pytest.mark.skipif(UnityEnvWrapper is None, reason="UnityEnvWrapper not available")
@pytest.mark.parametrize(
    "mock_communicator, unity_env_wrapper",
    [("mock_communicator_params_vec_cont", "mock_communicator_params_vec_cont")],
    indirect=["mock_communicator", "unity_env_wrapper"]
)
def test_wrapper_error_handling(unity_env_wrapper, mock_communicator):
    env = unity_env_wrapper

    # Test step before reset (should be handled by internal _assert_is_waiting_for_step)
    # This might not raise an error but set a flag or log, depending on wrapper design.
    # The current wrapper seems to allow step if _ma_agents_info is populated (done by _reset_internal)
    # A direct call to step() without reset() would likely fail if _ma_agents_info is empty.
    # However, the structure `all_brain_info = self.communicator.initialize(self._get_unity_input())` happens in __init__.
    # And `self._reset_internal(train_mode)` is called by `reset()`.
    # `_get_all_brain_info()` is called by `reset()`, which calls `initialize`.
    # `step()` calls `_get_all_brain_info(self.actions_for_brain)`, which calls `exchange`.
    # It seems the wrapper is always "initialized" with a first call to communicator.initialize()
    # So, step() before reset() might not error out but use initial (post-init) state.
    # This behavior might be acceptable.

    # Test after close
    env.close()
    with pytest.raises(MockUnityEnvironmentException): # Assuming communicator.exchange raises this
        env.step([env.action_space[i].sample() for i in range(env.num_agents)])
    
    with pytest.raises(MockUnityEnvironmentException): # Assuming communicator.initialize raises this
        env.reset()


def test_imports(): # Keep this simple test
    assert UnityEnvWrapper is not None

# Further notes:
# - The visual observation shape logic in `test_wrapper_initialization` is simplified.
#   A real test would need to be more precise based on how `UnityEnvWrapper`
#   handles `uint8_visual`, `allow_multiple_obs`, and stacking for visual data.
#   My MockBrainParameters and MockBrainInfo return visual data as HWC uint8.
#   The wrapper converts to float32 unless uint8_visual is True.
#   If uint8_visual=True, space is Box(0,255, (H,W,C*stack), uint8).
#   If uint8_visual=False, space is Box(0,1, (H,W,C*stack), float32) after normalization.
#   My mock uses float32 for Box space by default.
# - Branched discrete actions (`flatten_branched=False`) create a Tuple action space.
#   The test for this in `test_wrapper_initialization` is included.
#   When stepping with branched actions, `actions` for that agent would be a tuple of ints.
# - `allow_multiple_obs=True` with mixed visual/vector for a brain results in Tuple obs space.
#   This is also covered conceptually in `test_wrapper_initialization`.
# - The parameterization of `unity_env_wrapper` fixture is a bit tricky with `request.param`
#   if `mock_communicator` also uses `request.param`. A common pattern is to pass the
#   config dict to both. I've updated `unity_env_wrapper` to take `request.param` which will be the
#   same dict as `mock_communicator` receives, so it can access `use_visual_wrapper` etc.
# - MockBrainParameters `number_visual_observations` corrected to be `len(camera_resolutions)`.
# - MockCommunicator agent ID generation simplified to ensure unique IDs per brain instance.
# - MockCommunicator `initialize` and `exchange` now use `_get_brain_info_for_brain` for consistency.
# - `MockCommunicator.is_closed` property added for `test_wrapper_close`.
# - `MockCommunicator.exchange` now raises MockUnityEnvironmentException if closed.
# - Corrected `MockBrainInfo.agents` to use the fixed set of agent IDs for that brain.
# - Test for `train_mode` in `test_wrapper_reset`.
# - Enhanced `test_wrapper_step` to check `info['all_done']` correctly.
# - Added `test_wrapper_error_handling`.The tests for `UnityEnvWrapper` have been implemented in `tests/test_unity_env_wrapper.py`.

**Key aspects of the implementation:**

1.  **Mock Objects (`MockCommunicator`, `MockBrainParameters`, `MockBrainInfo`):**
    *   These were defined within the test file to simulate the behavior of the Unity `ml-agents` communication layer.
    *   `MockBrainParameters` allows defining various brain configurations (vector/visual obs, discrete/continuous actions, stacked observations, camera resolutions).
    *   `MockBrainInfo` represents the data packet coming from a brain for its agents.
    *   `MockCommunicator` manages a list of mock brains and their agents. Its `initialize()` and `exchange()` methods return structured `AllBrainInfo` (a dictionary mapping brain names to `MockBrainInfo`) with randomly generated data conforming to the `MockBrainParameters`. It also tracks its `closed` state and `train_mode`.

2.  **Pytest Fixtures:**
    *   **Communicator Parameter Fixtures (`mock_communicator_params_*`):** Three fixtures were defined to provide configurations for the `MockCommunicator`:
        *   `mock_communicator_params_vec_cont`: Single brain, 2 agents, vector observations, continuous actions.
        *   `mock_communicator_params_vis_disc`: Single brain, 1 agent, visual observations, discrete actions.
        *   `mock_communicator_params_multi_brain_mixed`: Two brains with a mix of agent counts and observation/action types (including branched discrete for one brain).
        *   These fixtures return dictionaries that include `brain_params_list`, `num_agents_per_brain_list`, and flags like `use_visual_wrapper` to control `UnityEnvWrapper` settings.
    *   **`mock_communicator(request)`:** A parameterized fixture that instantiates `MockCommunicator` using the parameters from one of the `_params_` fixtures.
    *   **`unity_env_wrapper(mock_communicator, request)`:** A parameterized fixture that instantiates `UnityEnvWrapper`. It receives the `mock_communicator` instance and the same parameter dictionary (via `request.param`) to configure wrapper settings like `use_visual` and `flatten_branched` consistently with the communicator's setup.

3.  **Test Cases:**
    *   **`test_wrapper_initialization`:**
        *   Parameterized to run with all three communicator configurations.
        *   Verifies that `env.observation_space` and `env.action_space` are lists of the correct `gymnasium.spaces` types (`Box`, `Discrete`, `Tuple` for branched discrete or mixed multiple observations) with appropriate shapes and dtypes.
        *   Checks `env.num_agents`, `env.brain_names`.
        *   The visual observation shape check accounts for channel concatenation by the wrapper.
        *   The discrete action space check considers `flatten_branched` for branched actions.
    *   **`test_wrapper_reset`:**
        *   Parameterized for vector and visual observation scenarios.
        *   Asserts that `env.reset()` returns a list of numpy arrays matching the length, shapes, and dtypes defined by `env.observation_space`.
        *   Tests that the `train_mode` parameter of `reset()` is correctly communicated (mock `Communicator` stores this state).
    *   **`test_wrapper_step`:**
        *   Parameterized for continuous and mixed (including branched discrete) action scenarios.
        *   Generates dummy actions compatible with `env.action_space`.
        *   Calls `env.step(actions)` and verifies the shapes and dtypes of `next_obs` (list of numpy arrays), `rewards` (numpy array), and `dones` (numpy array).
        *   Checks that `info` is a dictionary containing an `all_done` boolean key.
        *   Includes a sub-test where the `MockCommunicator` is temporarily rigged to make all agents done, ensuring `info['all_done']` and the `dones` array become `True`.
    *   **`test_wrapper_close`:**
        *   Verifies that calling `env.close()` results in `env.communicator.close()` being called (by checking the `is_closed` property on the mock communicator).
    *   **`test_wrapper_error_handling`:**
        *   Tests that calling `step()` or `reset()` on a closed environment raises a `MockUnityEnvironmentException` (as thrown by the mock communicator).

The tests aim to cover the primary logic of `UnityEnvWrapper` in translating data from the (mocked) Unity communication protocol to a Gymnasium-like multi-agent environment interface. The parameterization allows testing various configurations with a single set of test functions.
