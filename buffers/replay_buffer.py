import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from gymnasium import spaces

from utils.env_tools import get_shape_from_obs_space, get_shape_from_act_space

class ReplayBuffer:
    """Replay Buffer for multi agent environments with per-agent storage in separate arrays."""

    def __init__(
            self, 
            buffer_size, 
            batch_size, 
            state_spaces, 
            action_spaces, 
            *,
            device=torch.device("cpu"),
            n_step: int = 1,
            gamma: float = 0.99):
        """
        Initialize the ReplayBuffer with per-agent storage.
        
        Args:
            buffer_size (int): Maximum size of the buffer
            batch_size (int): Size of each training batch
            state_spaces (list): List of state spaces per agent (e.g., [14, 10, 10])
            action_spaces (list): List of action spaces per agent (e.g., [7, 4, 5])
            device (torch.device): Device to use for training
            n_step (int): Number of steps to look ahead for bootstrapping
            gamma (float): Discount factor for n-step returns
        """
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.device = device
        self.n_step = n_step
        self.gamma = gamma
        
        self.num_agents = len(state_spaces)
        self.state_shapes = [get_shape_from_obs_space(state_space) for state_space in state_spaces]
        self.action_shapes = [get_shape_from_act_space(action_space) for action_space in action_spaces]
        self.action_spaces = action_spaces

        # Initialize per-agent buffers as separate arrays
        self.states_buffer = []
        self.actions_buffer = []
        self.rewards_buffer = []
        self.next_states_buffer = []
        self.dones_buffer = []

        # Create separate buffers for each agent
        for i in range(self.num_agents):
            self.states_buffer.append(np.zeros((self.buffer_size, *self.state_shapes[i]), dtype=np.float32))
            self.rewards_buffer.append(np.zeros((self.buffer_size, 1), dtype=np.float32))
            self.next_states_buffer.append(np.zeros((self.buffer_size, *self.state_shapes[i]), dtype=np.float32))
            self.dones_buffer.append(np.zeros((self.buffer_size, 1), dtype=np.uint8))
            if isinstance(self.action_spaces[i], spaces.Box):
                action_dtype = np.float32
            elif isinstance(self.action_spaces[i], (spaces.Discrete, spaces.MultiDiscrete)):
                action_dtype = np.int64 # Or np.int32 if appropriate
            else:
                raise ValueError(f"Unsupported action space type for agent {i}")
            self.actions_buffer.append(np.zeros((self.buffer_size, self.action_shapes[i]), dtype=action_dtype))

        self.pos = 0
        self.size = 0

        # ---- short per‑agent n‑step caches -------------------------------------------
        self._cache_states = [deque(maxlen=n_step) for _ in range(self.num_agents)]
        self._cache_actions = [deque(maxlen=n_step) for _ in range(self.num_agents)]
        self._cache_rewards = [deque(maxlen=n_step) for _ in range(self.num_agents)]
        self._cache_dones = [deque(maxlen=n_step) for _ in range(self.num_agents)]
        self._cache_next_states = [deque(maxlen=n_step) for _ in range(self.num_agents)]

    def add(self, states, actions, rewards, next_states, dones):
        """
        Add a new experience to the buffer.
        
        Args:
            states (list): List of states per agent (variable sizes)
            actions (list): List of actions per agent (variable sizes)
            rewards (list or np.ndarray): Rewards per agent [num_agents]
            next_states (list): List of next states per agent (variable sizes)
            dones (list or np.ndarray): Done flags per agent [num_agents]
        """
        for a in range(self.num_agents):
            self._cache_states[a].append(states[a])
            self._cache_actions[a].append(actions[a])
            self._cache_rewards[a].append(float(rewards[a]))
            self._cache_dones[a].append(bool(dones[a]))
            self._cache_next_states[a].append(next_states[a])

        # write as many finished n‑step transitions as possible
        self._maybe_write(cache_finished=dones)
        # p = self.pos
        # # Store experience for each agent
        # for i in range(self.num_agents):
        #     self.states_buffer[i][p] = states[i] 
        #     self.rewards_buffer[i][p] = rewards[i]
        #     self.next_states_buffer[i][p] = next_states[i]
        #     self.dones_buffer[i][p] = dones[i]
        #     self.actions_buffer[i][p] = actions[i] #  can apply (2,) and (1, 2), numpy squeezes it
        
        # Update pos and size
        # self.pos = (p + 1) % self.buffer_size
        # self.size = min(self.size + 1, self.buffer_size)

    def sample(self):
        """
        Sample a batch of experiences from the buffer.
        
        Returns:
            tuple: (states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch, 
                   states_full, next_states_full, actions_full)
                  Each is a tensor with appropriate shape for MADDPG training
        """
        if self.size < self.batch_size:
            print(f"Warning: Not enough samples ({self.size}) in buffer to form a batch of size {self.batch_size}. Skipping sample.")
            return None
        
        # Sample indices
        indices = np.random.choice(self.size, self.batch_size, replace=False)
        
        # Initialize tensors for each agent
        states_batch_list = []
        actions_batch_list = []
        rewards_batch_list = []
        next_states_batch_list = []
        dones_batch_list = []
        
        # Collect experiences for each agent
        for i in range(self.num_agents):
            # Get data for this agent
            agent_states = torch.from_numpy(
                self.states_buffer[i][indices]
            ).float().to(self.device) # [batch_size, state_size_i]  
            agent_actions = torch.from_numpy(
                self.actions_buffer[i][indices]
            ).to(self.device)  # [batch_size, action_size_i]
            agent_rewards =  torch.from_numpy(
                self.rewards_buffer[i][indices]
            ).float().to(self.device)  # [batch_size, 1]
            agent_next_states = torch.from_numpy(
                self.next_states_buffer[i][indices]
            ).float().to(self.device)  # [batch_size, state_size_i]
            agent_dones = torch.from_numpy(
                self.dones_buffer[i][indices]
            ).float().to(self.device)  # [batch_size, 1]
            
            # Convert to tensors
            states_batch_list.append(agent_states)  # [agent_idx, batch_size, state_size_i]
            actions_batch_list.append(
                self._preprocess_actions_for_critic(i, agent_actions)
            ) # [agent_idx, batch_size, action_size_i] or [agent_idx, batch_size, sum(action_sizes)]
            rewards_batch_list.append(agent_rewards)  # [agent_idx, batch_size]
            next_states_batch_list.append(agent_next_states)  # [agent_idx, batch_size, state_size_i]
            dones_batch_list.append(agent_dones)  # [agent_idx, batch_size]

        # Stack rewards and dones directly without squeeze
        # rewards_batch = torch.stack(rewards_batch_list)  # [num_agents, batch_size, 1]
        # dones_batch = torch.stack(dones_batch_list)  # [num_agents, batch_size, 1]
        
        # Create full state and action tensors for centralized critic
        states_full = torch.cat([s.view(self.batch_size, -1) for s in states_batch_list], dim=1)  # [batch_size, sum(state_sizes)]
        next_states_full = torch.cat([ns.view(self.batch_size, -1) for ns in next_states_batch_list], dim=1)
        actions_full = torch.cat([a_proc.view(self.batch_size, -1) for a_proc in actions_batch_list], dim=1)
        
        return (states_batch_list, 
                actions_batch_list, 
                rewards_batch_list, 
                next_states_batch_list, 
                dones_batch_list, 
                states_full, 
                next_states_full, 
                actions_full)

    def _preprocess_actions_for_critic(
            self, 
            agent_id:int, 
            raw_actions:torch.Tensor):
        """
        Converts raw stored actions (indices for discrete) to critic-ready format (one-hot).
        
        Args:
            agent_idx: Index of the agent.
            raw_actions_tensor: Tensor of actions as stored in the buffer (e.g., indices).
                                Shape (batch_size, *action_shape_np[agent_idx]).
                                For Discrete, this is (batch_size, 1) and Long.
                                For MultiDiscrete, (batch_size, num_components) and Long.
                                For Box, (batch_size, *dim) and Float.
        
        Returns:
            actions_full (torch.Tensor): Concatenated actions for all agents [batch_size, sum(action_sizes)]
        """
        action_space = self.action_spaces[agent_id]

        if action_space.__class__.__name__ == "Box":
            return raw_actions.float()
        elif action_space.__class__.__name__ == "Discrete":
            num_classes = action_space.n
            # Ensure indices are long and remove extra dim
            return F.one_hot(raw_actions.squeeze(-1).long(), num_classes=num_classes).float()
        elif action_space.__class__.__name__ == "MultiDiscrete":
            one_hots_list = []
            num_classes_agent_i = action_space.nvec
            for i, num_classes_i in enumerate(num_classes_agent_i):
                indices = raw_actions[:, i].long() # Ensure indices are long
                one_hots_list.append(F.one_hot(indices, num_classes=num_classes_i).float())
            one_hot_actions = torch.cat(one_hots_list, dim=-1)
            return one_hot_actions
        else:
            raise NotImplementedError(f"Preprocessing not implemented for action space type: \
                                       {type(action_space)} for agent {agent_id}")
    
    def __len__(self):
        """Return the current size of the buffer."""
        return self.size
    

    def _maybe_write(self, cache_finished):
        """Write ready n‑step transitions, and flush the rest at episode end."""
        # first handle the common sliding‑window case – exactly one write per call
        if all(len(c) == self.n_step for c in self._cache_rewards):
            self._write_and_pop(n=self.n_step)

        # if the episode terminated this timestep, flush what is left (shorter windows)
        if any(cache_finished):
            while len(self._cache_rewards[0]):
                self._write_and_pop(n=len(self._cache_rewards[0]))

    def _write_and_pop(self, n):
        """Pop the oldest element from every agent cache and store it with an **n‑step** target."""
        p = self.pos
        Rn, done_n = [0.0]*self.num_agents, [False]*self.num_agents
        for k in range(n):
            g = self.gamma ** k
            for a in range(self.num_agents):
                Rn[a] += g * self._cache_rewards[a][k]
                done_n[a] = done_n[a] or self._cache_dones[a][k]

        for a in range(self.num_agents):
            self.states_buffer[a][p] = self._cache_states[a][0]
            self.actions_buffer[a][p] = self._cache_actions[a][0]
            self.rewards_buffer[a][p] = Rn[a]
            self.next_states_buffer[a][p] = self._cache_next_states[a][n - 1]
            self.dones_buffer[a][p] = done_n[a]

            # pop *oldest* element so the window slides by exactly 1 step
            self._cache_states[a].popleft()
            self._cache_actions[a].popleft()
            self._cache_rewards[a].popleft()
            self._cache_dones[a].popleft()
            self._cache_next_states[a].popleft()

        self.pos = (p + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)