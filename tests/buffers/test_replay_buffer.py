import pytest
import numpy as np
import torch
from gymnasium import spaces

from buffers.replay_buffer import ReplayBuffer

# Default parameters for creating a ReplayBuffer instance in tests
DEFAULT_BUFFER_SIZE = 100
DEFAULT_BATCH_SIZE = 10
DEFAULT_N_STEP = 1
DEFAULT_GAMMA = 0.99
DEFAULT_DEVICE = torch.device("cpu")

# --- Fixtures ---

@pytest.fixture
def discrete_action_space_defs():
    return [spaces.Discrete(5)] # Single agent, 5 discrete actions

@pytest.fixture
def continuous_action_space_defs():
    # Single agent, 2 continuous actions
    return [spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)]

@pytest.fixture
def multi_agent_action_space_defs():
    return [
        spaces.Discrete(5), # Agent 0, 5 discrete actions
        spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32) # Agent 1, 2 continuous actions
    ]

@pytest.fixture
def state_space_defs(): # Renamed to be generic
    # Single agent, state dim 10
    return [spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)]

@pytest.fixture
def multi_agent_state_space_defs():
    return [
        spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32), # Agent 0
        spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)   # Agent 1
    ]

@pytest.fixture
def replay_buffer_single_agent_discrete(state_space_defs, discrete_action_space_defs):
    """Replay buffer for a single agent with discrete actions and n_step=1."""
    return ReplayBuffer(
        buffer_size=DEFAULT_BUFFER_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        state_spaces=state_space_defs,
        action_spaces=discrete_action_space_defs,
        device=DEFAULT_DEVICE,
        n_step=DEFAULT_N_STEP,
        gamma=DEFAULT_GAMMA
    )

@pytest.fixture
def replay_buffer_single_agent_continuous(state_space_defs, continuous_action_space_defs):
    """Replay buffer for a single agent with continuous actions and n_step=1."""
    return ReplayBuffer(
        buffer_size=DEFAULT_BUFFER_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        state_spaces=state_space_defs,
        action_spaces=continuous_action_space_defs,
        device=DEFAULT_DEVICE,
        n_step=DEFAULT_N_STEP,
        gamma=DEFAULT_GAMMA
    )

@pytest.fixture
def replay_buffer_multi_agent(multi_agent_state_space_defs, multi_agent_action_space_defs):
    """Replay buffer for multiple agents with mixed action types and n_step=1."""
    return ReplayBuffer(
        buffer_size=DEFAULT_BUFFER_SIZE,
        batch_size=DEFAULT_BATCH_SIZE,
        state_spaces=multi_agent_state_space_defs,
        action_spaces=multi_agent_action_space_defs,
        device=DEFAULT_DEVICE,
        n_step=DEFAULT_N_STEP,
        gamma=DEFAULT_GAMMA
    )

# --- Helper Functions ---
def generate_dummy_experience(state_space_list, action_space_list):
    """Generates a single experience tuple (lists for each component) for all agents."""
    num_agents = len(state_space_list)

    states_li = []
    actions_li = []
    next_states_li = []

    for i in range(num_agents):
        states_li.append(state_space_list[i].sample().astype(np.float32))
        next_states_li.append(state_space_list[i].sample().astype(np.float32))

        action_space_i = action_space_list[i]
        # Store actions in the way ReplayBuffer expects them (raw, before one-hot)
        if isinstance(action_space_i, spaces.Discrete):
            actions_li.append(action_space_i.sample()) # This is an int
        elif isinstance(action_space_i, spaces.Box):
            actions_li.append(action_space_i.sample().astype(np.float32))
        elif isinstance(action_space_i, spaces.MultiDiscrete):
             actions_li.append(action_space_i.sample().astype(np.int64))
        else:
            raise ValueError(f"Unsupported action space type for dummy data generation: {type(action_space_i)}")

    rewards_li = np.random.rand(num_agents).astype(np.float32)
    # `add` expects list of bools for dones
    dones_li = [bool(np.random.randint(0, 2)) for _ in range(num_agents)]

    return states_li, actions_li, rewards_li, next_states_li, dones_li

# --- Test Cases ---

def test_replay_buffer_initialization(replay_buffer_single_agent_discrete):
    rb = replay_buffer_single_agent_discrete
    assert rb.buffer_size == DEFAULT_BUFFER_SIZE
    assert rb.batch_size == DEFAULT_BATCH_SIZE
    assert rb.n_step == DEFAULT_N_STEP
    assert rb.num_agents == 1

    state_shape = rb.state_shapes[0]  # ReplayBuffer stores shapes, not spaces
    action_shape = rb.action_shapes[0]  # For Discrete(N), this is 1 (int)

    assert len(rb.states_buffer) == 1
    assert rb.states_buffer[0].shape == (DEFAULT_BUFFER_SIZE, *state_shape)
    # For Discrete, action_shape is 1 (int), so actions_buffer[0] is (DEFAULT_BUFFER_SIZE, 1)
    assert rb.actions_buffer[0].shape == (DEFAULT_BUFFER_SIZE, action_shape)
    assert rb.rewards_buffer[0].shape == (DEFAULT_BUFFER_SIZE, 1)
    assert rb.next_states_buffer[0].shape == (DEFAULT_BUFFER_SIZE, *state_shape)
    assert rb.dones_buffer[0].shape == (DEFAULT_BUFFER_SIZE, 1)

    assert rb.pos == 0
    assert rb.size == 0
    assert len(rb) == 0

def test_replay_buffer_initialization_multi_agent(replay_buffer_multi_agent):
    rb = replay_buffer_multi_agent
    assert rb.buffer_size == DEFAULT_BUFFER_SIZE
    assert rb.num_agents == 2
    assert len(rb.states_buffer) == 2

    state_shape_0 = rb.state_shapes[0]  # ReplayBuffer stores shapes, not spaces
    action_shape_0 = rb.action_shapes[0]  # Discrete -> 1 (int)
    state_shape_1 = rb.state_shapes[1]  # ReplayBuffer stores shapes, not spaces
    action_shape_1 = rb.action_shapes[1]  # Box -> 2 (int, first dimension)

    assert rb.states_buffer[0].shape == (DEFAULT_BUFFER_SIZE, *state_shape_0)
    assert rb.states_buffer[1].shape == (DEFAULT_BUFFER_SIZE, *state_shape_1)

    assert rb.actions_buffer[0].shape == (DEFAULT_BUFFER_SIZE, action_shape_0)  # Discrete
    assert rb.actions_buffer[1].shape == (DEFAULT_BUFFER_SIZE, action_shape_1)  # Box

    assert rb.actions_buffer[0].dtype == np.int64 # Agent 0 is Discrete
    assert rb.actions_buffer[1].dtype == np.float32 # Agent 1 is Box
    assert rb.pos == 0
    assert rb.size == 0

def test_add_single_transition_single_agent_discrete(replay_buffer_single_agent_discrete, state_space_defs, discrete_action_space_defs):
    rb = replay_buffer_single_agent_discrete
    states, actions, rewards, next_states, dones = generate_dummy_experience(
        state_space_defs, discrete_action_space_defs
    )

    rb.add(states, actions, rewards, next_states, dones)

    assert rb.pos == 1 % DEFAULT_BUFFER_SIZE
    assert rb.size == 1
    assert len(rb) == 1

    assert np.array_equal(rb.states_buffer[0][0], states[0])
    assert rb.actions_buffer[0][0] == actions[0] # actions[0] is an int for Discrete
    assert rb.rewards_buffer[0][0] == pytest.approx(rewards[0])
    assert np.array_equal(rb.next_states_buffer[0][0], next_states[0])
    assert rb.dones_buffer[0][0] == dones[0] # dones[0] is a bool, stored as uint8

def test_add_single_transition_single_agent_continuous(replay_buffer_single_agent_continuous, state_space_defs, continuous_action_space_defs):
    rb = replay_buffer_single_agent_continuous
    states, actions, rewards, next_states, dones = generate_dummy_experience(
        state_space_defs, continuous_action_space_defs
    )

    rb.add(states, actions, rewards, next_states, dones)

    assert rb.pos == 1 % DEFAULT_BUFFER_SIZE
    assert rb.size == 1
    assert np.array_equal(rb.actions_buffer[0][0], actions[0]) # actions[0] is a np.array for Box

def test_add_multiple_transitions_update_pointer_and_size(replay_buffer_single_agent_discrete, state_space_defs, discrete_action_space_defs):
    rb = replay_buffer_single_agent_discrete
    num_adds = 5
    for i in range(num_adds):
        states, actions, rewards, next_states, dones = generate_dummy_experience(
            state_space_defs, discrete_action_space_defs
        )
        rb.add(states, actions, rewards, next_states, dones)
        assert rb.pos == (i + 1) % DEFAULT_BUFFER_SIZE
        assert rb.size == (i + 1)
        assert len(rb) == (i + 1)

def test_add_wrapping_around_capacity(replay_buffer_single_agent_discrete, state_space_defs, discrete_action_space_defs):
    rb = replay_buffer_single_agent_discrete
    for i in range(DEFAULT_BUFFER_SIZE):
        states, actions, rewards, next_states, dones = generate_dummy_experience(
            state_space_defs, discrete_action_space_defs,
        )
        rb.add(states, actions, rewards, next_states, dones)

    assert rb.pos == 0
    assert rb.size == DEFAULT_BUFFER_SIZE

    s_new, a_new, r_new, ns_new, d_new = generate_dummy_experience(
        state_space_defs, discrete_action_space_defs
    )
    rb.add(s_new, a_new, r_new, ns_new, d_new)

    assert rb.pos == 1
    assert rb.size == DEFAULT_BUFFER_SIZE

    assert np.array_equal(rb.states_buffer[0][0], s_new[0])
    assert rb.actions_buffer[0][0] == a_new[0]
    assert rb.rewards_buffer[0][0] == pytest.approx(r_new[0])

def test_sample_from_empty_buffer(replay_buffer_single_agent_discrete):
    rb = replay_buffer_single_agent_discrete
    samples = rb.sample()
    assert samples is None

def test_sample_less_than_batch_size(replay_buffer_single_agent_discrete, state_space_defs, discrete_action_space_defs):
    rb = replay_buffer_single_agent_discrete
    for _ in range(DEFAULT_BATCH_SIZE - 1): # Add 9 experiences, batch_size is 10
        states, actions, rewards, next_states, dones = generate_dummy_experience(
            state_space_defs, discrete_action_space_defs
        )
        rb.add(states, actions, rewards, next_states, dones)

    samples = rb.sample()
    assert samples is None

def test_sample_correct_batch_shape_and_types_single_agent_discrete(replay_buffer_single_agent_discrete, state_space_defs, discrete_action_space_defs):
    rb = replay_buffer_single_agent_discrete
    for _ in range(DEFAULT_BATCH_SIZE + 5):
         states, actions, rewards, next_states, dones = generate_dummy_experience(
            state_space_defs, discrete_action_space_defs
        )
         rb.add(states, actions, rewards, next_states, dones)

    samples = rb.sample()
    assert samples is not None
    s_batch, a_batch, r_batch, ns_batch, d_batch, s_full, ns_full, a_full = samples

    assert len(s_batch) == 1
    assert len(a_batch) == 1 # This is a list of tensors, one per agent

    state_shape_tuple = rb.state_shapes[0]  # ReplayBuffer stores shapes, not spaces
    action_space_n = rb.action_spaces[0].n # For Discrete

    assert s_batch[0].shape == (DEFAULT_BATCH_SIZE, *state_shape_tuple)
    # a_batch[0] for Discrete is one-hot encoded by _preprocess_actions_for_critic
    assert a_batch[0].shape == (DEFAULT_BATCH_SIZE, action_space_n)
    assert r_batch[0].shape == (DEFAULT_BATCH_SIZE, 1)
    assert ns_batch[0].shape == (DEFAULT_BATCH_SIZE, *state_shape_tuple)
    assert d_batch[0].shape == (DEFAULT_BATCH_SIZE, 1)

    assert s_batch[0].dtype == torch.float32
    assert a_batch[0].dtype == torch.float32 # one-hot is float
    assert r_batch[0].dtype == torch.float32
    assert ns_batch[0].dtype == torch.float32
    assert d_batch[0].dtype == torch.float32

    assert s_full.shape == (DEFAULT_BATCH_SIZE, state_shape_tuple[0])
    assert ns_full.shape == (DEFAULT_BATCH_SIZE, state_shape_tuple[0])
    assert a_full.shape == (DEFAULT_BATCH_SIZE, action_space_n)

def test_sample_correct_batch_shape_multi_agent(replay_buffer_multi_agent, multi_agent_state_space_defs, multi_agent_action_space_defs):
    rb = replay_buffer_multi_agent
    num_agents = rb.num_agents
    for _ in range(DEFAULT_BATCH_SIZE + 5):
        s, a, r, ns, d = generate_dummy_experience(multi_agent_state_space_defs, multi_agent_action_space_defs)
        rb.add(s, a, r, ns, d)

    samples = rb.sample()
    assert samples is not None
    s_batch_list, a_batch_list, r_batch_list, ns_batch_list, d_batch_list, s_full, ns_full, a_full = samples

    sum_state_dims = 0
    sum_action_dims_critic = 0

    for i in range(num_agents):
        state_shape_i = rb.state_shapes[i]  # ReplayBuffer stores shapes, not spaces
        assert s_batch_list[i].shape == (DEFAULT_BATCH_SIZE, *state_shape_i)
        sum_state_dims += state_shape_i[0]

        action_space_i = rb.action_spaces[i]
        if isinstance(action_space_i, spaces.Discrete):
            assert a_batch_list[i].shape == (DEFAULT_BATCH_SIZE, action_space_i.n) # one-hot
            sum_action_dims_critic += action_space_i.n
        elif isinstance(action_space_i, spaces.Box):
            action_shape_i = rb.action_shapes[i]  # ReplayBuffer stores shapes, not spaces (int)
            assert a_batch_list[i].shape == (DEFAULT_BATCH_SIZE, action_shape_i)
            sum_action_dims_critic += action_shape_i

        assert r_batch_list[i].shape == (DEFAULT_BATCH_SIZE, 1)
        assert ns_batch_list[i].shape == (DEFAULT_BATCH_SIZE, *state_shape_i)
        assert d_batch_list[i].shape == (DEFAULT_BATCH_SIZE, 1)

    assert s_full.shape == (DEFAULT_BATCH_SIZE, sum_state_dims)
    assert ns_full.shape == (DEFAULT_BATCH_SIZE, sum_state_dims)
    assert a_full.shape == (DEFAULT_BATCH_SIZE, sum_action_dims_critic)

def test_capacity_limit(replay_buffer_single_agent_discrete, state_space_defs, discrete_action_space_defs):
    rb = replay_buffer_single_agent_discrete
    for _ in range(DEFAULT_BUFFER_SIZE + 20):
        states, actions, rewards, next_states, dones = generate_dummy_experience(
            state_space_defs, discrete_action_space_defs
        )
        rb.add(states, actions, rewards, next_states, dones)

    assert rb.size == DEFAULT_BUFFER_SIZE
    assert len(rb) == DEFAULT_BUFFER_SIZE
    assert rb.pos == (DEFAULT_BUFFER_SIZE + 20) % DEFAULT_BUFFER_SIZE

def test_data_integrity_single_agent_discrete(replay_buffer_single_agent_discrete, state_space_defs, discrete_action_space_defs):
    rb = replay_buffer_single_agent_discrete

    added_experiences = []
    for i in range(DEFAULT_BATCH_SIZE + 5): # Fill enough to sample
        s, a, r, ns, d = generate_dummy_experience(
            state_space_defs, discrete_action_space_defs
        )
        # Make dones alternate to test n_step flush properly if n_step > 1 (though here it's 1)
        # For n_step=1, done status doesn't affect much beyond its own value.
        actual_dones = [True if i % 2 == 0 else False for _ in d]
        rb.add(s, a, r, ns, actual_dones)
        # Store the exact values that were passed to add, for later comparison with buffer contents
        added_experiences.append({'s': s[0], 'a': a[0], 'r': r[0], 'ns': ns[0], 'd': actual_dones[0]})

    # Check a few stored experiences directly (since n_step=1)
    for i in range(min(len(added_experiences), DEFAULT_BUFFER_SIZE)):
        original_exp = added_experiences[i]
        idx_in_buffer = i % DEFAULT_BUFFER_SIZE # Handles wrap-around if we added > capacity

        assert np.array_equal(rb.states_buffer[0][idx_in_buffer], original_exp['s']), f"State mismatch at index {i}"
        assert rb.actions_buffer[0][idx_in_buffer] == original_exp['a'], f"Action mismatch at index {i}"
        assert rb.rewards_buffer[0][idx_in_buffer] == pytest.approx(original_exp['r']), f"Reward mismatch at index {i}"
        assert np.array_equal(rb.next_states_buffer[0][idx_in_buffer], original_exp['ns']), f"Next state mismatch at index {i}"
        assert rb.dones_buffer[0][idx_in_buffer] == original_exp['d'], f"Done mismatch at index {i}"


def test_n_step_add_and_data_storage(state_space_defs, discrete_action_space_defs):
    n_step_val = 3
    buffer_cap = 10
    batch_s = 1 # Sample one by one to check easily
    rb_nstep = ReplayBuffer(
        buffer_size=buffer_cap, batch_size=batch_s, state_spaces=state_space_defs,
        action_spaces=discrete_action_space_defs, device=DEFAULT_DEVICE,
        n_step=n_step_val, gamma=0.9
    )

    experiences_raw = [] # To store s,a,r,ns,d as added
    # Add enough experiences to form several n-step transitions
    for i in range(n_step_val + 2):
        s, a, r, ns, d_original = generate_dummy_experience(state_space_defs, discrete_action_space_defs)
        # Ensure dones are False for the first n_step_val-1 items to allow full n-step accumulation
        # The last item's done status can be True or False.
        current_dones = [False] if i < n_step_val -1 else d_original
        rb_nstep.add(s, a, r, ns, current_dones)
        experiences_raw.append({'s':s[0], 'a':a[0], 'r':r[0], 'ns':ns[0], 'd':current_dones[0]})

    # After n_step_val + 2 additions, (n_step_val + 2) - n_step_val + 1 = 3 transitions should be stored if no early 'done'
    # If last 'done' was True, then (n_step_val + 2 - (n_step_val-1)) + (n_step_val-1 - (n_step_val-2)) ...
    # it will flush all, so potentially (n_step_val + 2) transitions if all were short.
    # Let's check the first stored transition (pos=0 in buffer)
    # It should be (s_0, a_0, R_0, ns_{n-1}, D_0)
    # R_0 = r_0 + gamma*r_1 + gamma^2*r_2 (for n_step=3)
    # D_0 = d_0 or d_1 or d_2

    assert rb_nstep.size > 0, "Buffer should have stored some n-step transitions"

    stored_s0 = rb_nstep.states_buffer[0][0]
    stored_a0 = rb_nstep.actions_buffer[0][0]
    stored_r0_n_step = rb_nstep.rewards_buffer[0][0]
    stored_ns0_n_step = rb_nstep.next_states_buffer[0][0]
    stored_d0_n_step = rb_nstep.dones_buffer[0][0]

    expected_s0 = experiences_raw[0]['s']
    expected_a0 = experiences_raw[0]['a']

    expected_R0_n_step = 0.0
    expected_D0_n_step = False
    for k in range(n_step_val):
        expected_R0_n_step += (rb_nstep.gamma ** k) * experiences_raw[k]['r']
        expected_D0_n_step = expected_D0_n_step or experiences_raw[k]['d']
        if expected_D0_n_step and k < n_step_val -1 : # if episode ended early within the n-step window
             # The actual next_state and done for this n-step transition would be from this point
             expected_ns0_n_step = experiences_raw[k]['ns']
             break # Reward accumulation stops here for this n-step transition
    else: # If no early done in the first n-steps
        expected_ns0_n_step = experiences_raw[n_step_val - 1]['ns']


    assert np.array_equal(stored_s0, expected_s0)
    assert stored_a0 == expected_a0
    assert stored_r0_n_step == pytest.approx(expected_R0_n_step)
    assert np.array_equal(stored_ns0_n_step, expected_ns0_n_step)
    assert stored_d0_n_step == expected_D0_n_step

def test_add_with_episode_done_flushes_cache(state_space_defs, discrete_action_space_defs):
    n_step_val = 3
    rb_nstep = ReplayBuffer(
        buffer_size=10, batch_size=1, state_spaces=state_space_defs,
        action_spaces=discrete_action_space_defs, n_step=n_step_val, gamma=0.9
    )

    # Add one experience, not done
    s1, a1, r1, ns1, _ = generate_dummy_experience(state_space_defs, discrete_action_space_defs)
    d1_is_false = [False]
    rb_nstep.add(s1, a1, r1, ns1, d1_is_false)
    assert rb_nstep.size == 0 # Not enough for a full n-step transition yet

    # Add second experience, which IS done. This should trigger a flush.
    s2, a2, r2, ns2, _ = generate_dummy_experience(state_space_defs, discrete_action_space_defs)
    d2_is_true = [True]
    rb_nstep.add(s2, a2, r2, ns2, d2_is_true)

    # Cache flushes:
    # 1. A 2-step transition (s1, a1, r1 + gamma*r2, ns2, d1_false or d2_is_true)
    # 2. A 1-step transition (s2, a2, r2, ns2, d2_is_true)
    assert rb_nstep.size == 2

    # Check 1st flushed transition (original s1, a1)
    expected_r1_2step = r1[0] + rb_nstep.gamma * r2[0]
    expected_d1_2step = d1_is_false[0] or d2_is_true[0] # True
    assert np.array_equal(rb_nstep.states_buffer[0][0], s1[0])
    assert rb_nstep.actions_buffer[0][0] == a1[0]
    assert rb_nstep.rewards_buffer[0][0] == pytest.approx(expected_r1_2step)
    assert np.array_equal(rb_nstep.next_states_buffer[0][0], ns2[0]) # Next state is from end of window
    assert rb_nstep.dones_buffer[0][0] == expected_d1_2step

    # Check 2nd flushed transition (original s2, a2)
    expected_r2_1step = r2[0]
    expected_d2_1step = d2_is_true[0] # True
    assert np.array_equal(rb_nstep.states_buffer[0][1], s2[0])
    assert rb_nstep.actions_buffer[0][1] == a2[0]
    assert rb_nstep.rewards_buffer[0][1] == pytest.approx(expected_r2_1step)
    assert np.array_equal(rb_nstep.next_states_buffer[0][1], ns2[0])
    assert rb_nstep.dones_buffer[0][1] == expected_d2_1step
