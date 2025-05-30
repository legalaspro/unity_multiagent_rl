import numpy as np
import torch
from gymnasium.spaces import Box, Discrete # For type hinting and construction

class MockMAEnv:
    """
    A simple mock multi-agent environment, enhanced for evaluation scenarios.
    Designed to be compatible with what UnityEnvWrapper might provide to evaluators.
    """
    def __init__(self, num_agents=2, obs_shapes=None, action_spaces=None, episode_length=10, fixed_rewards=None):
        self.num_agents = num_agents
        self.episode_length = episode_length
        self.current_step = 0
        self._fixed_rewards_pattern = fixed_rewards

        if obs_shapes is None:
            obs_shapes = [(4,) for _ in range(num_agents)]
        if action_spaces is None:
            action_spaces = [Box(low=-1, high=1, shape=(2,), dtype=np.float32) for _ in range(num_agents)]

        self.observation_space = [ # List of spaces, one per agent
            Box(low=-np.inf, high=np.inf, shape=obs_shapes[i], dtype=np.float32) for i in range(num_agents)
        ]
        self.action_space = action_spaces # List of gym.spaces instances for each agent position
        self.action_spaces = action_spaces # For compatibility if something looks for self.action_spaces

    def reset(self, seed=None, options=None, train_mode=True): # Added train_mode for compatibility
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        # Returns list of observations, one per agent (as numpy arrays)
        observations = [obs_space.sample() for obs_space in self.observation_space]
        # UnityEnvWrapper's reset often just returns obs. Info is not always passed or used by evaluator's reset call.
        return observations 

    def step(self, actions):
        # actions is expected to be a list or array of actions, one per agent
        if len(actions) != self.num_agents:
            raise ValueError(f"Expected {self.num_agents} actions, got {len(actions)}")
        
        self.current_step += 1
        
        next_observations = [obs_space.sample() for obs_space in self.observation_space] # list of np arrays
        
        if self._fixed_rewards_pattern:
            rewards_list = self._fixed_rewards_pattern[(self.current_step - 1) % len(self._fixed_rewards_pattern)]
            rewards_array = np.array(rewards_list, dtype=np.float32)
        else:
            if self.num_agents == 2: 
                rand_val = np.random.rand()
                if rand_val < 0.4: rewards_list = [1.0, -1.0] 
                elif rand_val < 0.8: rewards_list = [-1.0, 1.0]
                else: rewards_list = [0.0, 0.0]
                rewards_array = np.array(rewards_list, dtype=np.float32)
            else:
                rewards_array = np.array([np.random.rand() for _ in range(self.num_agents)], dtype=np.float32)
        
        global_terminated = self.current_step >= self.episode_length
        
        # UnityEnvWrapper step returns: next_obs (list), rewards (np.array), dones (np.array), info (dict with 'all_done')
        dones_array = np.array([global_terminated for _ in range(self.num_agents)], dtype=bool)
        info_dict = {"all_done": global_terminated}
        # Optionally add per-agent info if needed, but evaluator mainly uses 'all_done'
        # for i in range(self.num_agents):
        #     info_dict[f'agent_{i}_final_reward'] = rewards_array[i] if global_terminated else 0.0

        return next_observations, rewards_array, dones_array, info_dict

    def close(self):
        # print("MockMAEnv closed.")
        pass # In a real env, this would release resources

    @property
    def n_agents(self): 
        return self.num_agents


class MockReplayBuffer:
    def __init__(self, buffer_size, batch_size, state_spaces, action_spaces, device, n_step=1, gamma=0.99, num_agents=None):
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.state_spaces = state_spaces
        self.action_spaces = action_spaces
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        self.num_agents = num_agents if num_agents is not None else len(state_spaces)
        self.add_call_count = 0
        self.last_added = None

    def add(self, states, actions, rewards, next_states, dones, *args, **kwargs):
        self.add_call_count += 1
        self.last_added = {"states": states, "actions": actions, "rewards": rewards, "next_states": next_states, "dones": dones}

    def sample(self):
        if self.add_call_count == 0: return None
        states_li, actions_li, rewards_li, next_states_li, dones_li = [], [], [], [], []
        for i in range(self.num_agents):
            obs_shape = self.state_spaces[i].shape
            act_space_i = self.action_spaces[i]
            states_li.append(torch.randn(self.batch_size, *obs_shape, device=self.device))
            if isinstance(act_space_i, Box):
                act_shape = act_space_i.shape
                actions_li.append(torch.randn(self.batch_size, *act_shape, device=self.device))
            elif isinstance(act_space_i, Discrete):
                actions_li.append(torch.randint(0, act_space_i.n, (self.batch_size, 1), device=self.device).float())
            else: actions_li.append(torch.zeros((self.batch_size, 0), device=self.device))
            rewards_li.append(torch.rand(self.batch_size, 1, device=self.device))
            next_states_li.append(torch.randn(self.batch_size, *obs_shape, device=self.device))
            dones_li.append(torch.randint(0, 2, (self.batch_size, 1), device=self.device).float())
        
        try: 
            full_states = torch.cat([s.view(self.batch_size, -1) for s in states_li], dim=1)
            full_next_states = torch.cat([ns.view(self.batch_size, -1) for ns in next_states_li], dim=1)
            full_actions = torch.cat([a.view(self.batch_size, -1) for a in actions_li], dim=1)
        except: 
            full_states, full_next_states, full_actions = torch.tensor([],device=self.device), torch.tensor([],device=self.device), torch.tensor([],device=self.device)
            
        return states_li, actions_li, rewards_li, next_states_li, dones_li, full_states, full_next_states, full_actions

    def __len__(self):
        return self.add_call_count


class MockEvalAgent:
    """ 
    Mock agent/policy for evaluation.
    Its `act` method is designed to be compatible with how `UnityEvaluator` and `CompetitiveEvaluator`
    call `multi_agent.act(obs_tensor_all_agents, deterministic=True)`.
    """
    def __init__(self, agent_id, name, env_action_spaces_list, device='cpu'):
        self.id = agent_id 
        self.name = name
        # env_action_spaces_list: list of gym.space for ALL agents in the environment
        self.env_action_spaces = env_action_spaces_list
        self._device = device # Store device, make it a private attribute
        # num_agents attribute expected by UnityEvaluator's multi_agent
        self.num_agents = len(env_action_spaces_list)


    def act(self, obs_tensor_all_env_agents, deterministic=True):
        """
        obs_tensor_all_env_agents: Tensor of observations for ALL N agents in the environment.
                                   Shape: (N_total_env_agents, obs_dim_individual_agent)
                                   (Assumes homogeneous obs_dim for stacking by evaluator)
        Returns a list of action Tensors, one for each agent in the environment.
        """
        if not isinstance(obs_tensor_all_env_agents, torch.Tensor):
            # Evaluator calls torch.as_tensor(np.stack(obs), ...), so input should be tensor.
            # However, if a user calls this mock directly with list of numpy, handle it.
            try:
                obs_tensor_all_env_agents = torch.tensor(np.array(obs_tensor_all_env_agents), device=self.device, dtype=torch.float32)
            except Exception as e:
                raise ValueError(f"MockEvalAgent.act: obs_tensor_all_env_agents could not be converted to tensor. Input type: {type(obs_tensor_all_env_agents)}. Error: {e}")

        
        num_total_env_agents = obs_tensor_all_env_agents.shape[0]
        if num_total_env_agents != len(self.env_action_spaces):
            raise ValueError(f"Mismatch: obs tensor has {num_total_env_agents} agents, but agent configured for {len(self.env_action_spaces)} spaces.")

        output_actions_list_of_tensors = []
        for i in range(num_total_env_agents):
            action_sample = self.env_action_spaces[i].sample() # numpy array
            action_tensor = torch.tensor(action_sample, device=self.device, dtype=torch.float32)
            if isinstance(self.env_action_spaces[i], Discrete) and action_tensor.ndim == 0: # Ensure discrete actions are at least 1D for consistency
                action_tensor = action_tensor.unsqueeze(0)
            output_actions_list_of_tensors.append(action_tensor)
            
        return output_actions_list_of_tensors

    def get_id(self): 
        return self.id

    def get_name(self): 
        return self.name

    @property 
    def device(self): # device property expected by evaluator
        return self._device
    
    @device.setter
    def device(self, value):
        self._device = value
        # In a real agent with networks, move them to device: self.policy.to(self._device)
        pass
