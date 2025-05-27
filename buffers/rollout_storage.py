import numpy as np
import torch

from gymnasium import spaces

from utils.env_tools import get_shape_from_obs_space, get_shape_from_act_space

class RolloutStorage:
    """
    Rollout storage for collecting multi-agent experiences during training.
    Designed for MAPPO with n-step returns and MLP-based policies.
    """
    def __init__(self, args, n_steps, state_spaces, action_spaces, device=torch.device("cpu")):
        """
        Initialize Rollout Storage for collecting experiences per agent.

        Args:
            args (argparse.Namespace): Hyperparameters
                gamma (float): Discount factor
                gae_lambda (float): GAE lambda
            n_steps (int): Number of steps to collect before update (can be different from episode length)
            state_spaces (list): List of state spaces per agent (e.g., [14, 10, 10])
            action_spaces (list): List of action spaces per agent (e.g., [7, 4, 5])
            add_role_id (bool): Whether to add role IDs to observations (for centralized critic)
            device (torch.device): Device to use for training
        """
        self.n_steps = n_steps
        self.device = device
        self.args = args

        self.n_agents = len(state_spaces)
        self.state_shapes = [get_shape_from_obs_space(state_space) for state_space in state_spaces]
        self.action_shapes = [get_shape_from_act_space(action_space) for action_space in action_spaces]
        self.action_spaces = action_spaces

        # NOTE: In our specific case, we assume all agents have the same observation and action space
        # Available actions can be added later, if we want reuse same policy
        if len(set(self.state_shapes)) != 1:
            raise ValueError("All state spaces must have the same shape")
        if len(set(self.action_shapes)) != 1:
            raise ValueError("All action spaces must have the same shape")

        self.obs_dim = self.state_shapes[0]
        self.action_dim = self.action_shapes[0]

        # Current position in the buffer
        self.step = 0
        # (obs_0, state_0) → action_0 / action_log_prob_0 → (reward_0, obs_1, state_1, mask_1, trunc_1)
        # delta = reward_0 + gamma * value_1 * mask_1 - value_0

        # Core storage buffers - using numpy arrays for efficiency
        self.obs = np.zeros((self.n_steps + 1,  self.n_agents, *self.obs_dim), dtype=np.float32)
        self.rewards = np.zeros((self.n_steps, self.n_agents, 1), dtype=np.float32)
        if isinstance(self.action_spaces[0], spaces.Box):
            self.actions = np.zeros((self.n_steps, self.n_agents, self.action_dim), dtype=np.float32)
        else:
            self.actions = np.zeros((self.n_steps, self.n_agents, self.action_dim), dtype=np.int64)
        self.action_log_probs = np.zeros((self.n_steps, self.n_agents, 1), dtype=np.float32)
        self.values = np.zeros((self.n_steps + 1, self.n_agents, 1), dtype=np.float32)
        self.masks = np.ones((self.n_steps + 1, self.n_agents, 1), dtype=np.float32) # 0 if episode done, 1 otherwise
        self.truncated = np.zeros((self.n_steps + 1, self.n_agents, 1), dtype=np.bool_) # 1 if episode truncated, 0 otherwise

        # Extra buffers for the algorithm
        self.returns = np.zeros((self.n_steps + 1, self.n_agents, 1), dtype=np.float32)
        self.advantages = np.zeros((self.n_steps, self.n_agents, 1), dtype=np.float32)

    def insert(
            self,
            obs,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            truncates):
        """
        Insert a new transition into the buffer.

        Args:
            obs: Agent observations [n_agents, obs_shape]
            actions: Actions taken by agents [n_agents, action_shape]
            action_log_probs: Log probs of actions [n_agents, 1]
            values: Value predictions [n_agents, 1]
            rewards: Rewards received [n_agents, 1]
            masks: Episode termination masks [n_agents, 1], 0 if episode done, 1 otherwise
            truncates: Boolean array indicating if episode was truncated (e.g., due to time limit)
                      rather than terminated [n_agents, 1]
        """
        self.obs[self.step + 1] = obs
        self.actions[self.step] = actions
        self.action_log_probs[self.step] = action_log_probs
        self.values[self.step] = values
        self.rewards[self.step] = rewards
        self.masks[self.step + 1] = masks
        self.truncated[self.step + 1] = truncates

        self.step += 1


    def compute_returns_and_advantages(self,
                                       next_values,
                                       gamma=0.99,
                                       lambda_=0.95,
                                       use_gae=True,
                                       normalize_per_agent=False):
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).
        Properly handles truncated episodes by incorporating next state values.

        Args:
            next_values: Value estimates for the next observations [n_agents, 1]
            gamma: Discount factor
            lambda_: GAE lambda parameter for advantage weighting
            use_gae: Whether to use GAE or just n-step returns
        """
        # Set the value of the next observation
        self.values[-1] = next_values

        # Create arrays for storing returns and advantages
        advantages = np.zeros_like(self.rewards)
        returns = np.zeros_like(self.returns)

        if use_gae:
            # GAE advantage computation with vectorized operations for better performance
            gae = np.zeros((self.n_agents, 1))  # Initialize as vector for multi-agent

            for step in reversed(range(self.n_steps)):
                # For truncated episodes, we adjust rewards directly
                adjusted_rewards = self.rewards[step].copy() # [n_agents, 1]

                # Identify truncated episodes (done but not terminated)
                truncated_mask = (self.masks[step + 1] == 0) & (self.truncated[step + 1] == 1) # [n_agents, 1]
                if np.any(truncated_mask):
                    # Add bootstrapped value only for truncated episodes
                    adjusted_rewards[truncated_mask] += gamma * self.values[step + 1][truncated_mask]

                # Calculate delta (TD error) with adjusted rewards
                delta = (
                    adjusted_rewards +
                    gamma * self.values[step + 1] * self.masks[step + 1] -
                    self.values[step]
                ) # [n_agents, 1]

                # Standard GAE calculation
                gae = delta + gamma * lambda_ * self.masks[step + 1] * gae # [n_agents, 1]
                advantages[step] = gae # [n_agents, 1]

            # Compute returns as advantages + self.values
            returns[:-1] = advantages + self.values[:-1] # [n_agents]
            returns[-1] = self.values[-1] # [n_agents]

        else:
            # N-step returns without GAE (more efficient calculation)
            returns[-1] = self.values[-1]
            for step in reversed(range(self.n_steps)):
                # Adjust rewards for truncated episodes
                adjusted_rewards = self.rewards[step].copy()

                # Identify truncated episodes
                truncated_mask = (self.masks[step + 1] == 0) & (self.truncated[step + 1] == 1)

                # For truncated episodes, add discounted bootstrapped value directly to rewards
                if np.any(truncated_mask):
                    adjusted_rewards[truncated_mask] += gamma * returns[step + 1][truncated_mask]

                # Calculate returns with proper masking
                returns[step] = adjusted_rewards + gamma * returns[step + 1] * self.masks[step + 1]

            # Calculate advantages
            advantages = returns[:-1] - self.values[:-1]

        # Store results
        self.returns = returns

        # Normalize advantages (helps with training stability)
        # Use stable normalization with small epsilon
        if normalize_per_agent:
            # First assign the raw advantages, then normalize per agent
            self.advantages = advantages.copy()
            for i in range(self.n_agents):
                ai = advantages[:, i, 0]                  # shape (T ,)
                self.advantages[:, i, 0] = (ai - ai.mean()) / (ai.std() + 1e-8)
        else:
            adv_mean = advantages.mean()
            adv_std = advantages.std()
            self.advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        return self.advantages, self.returns

    def after_update(self):
        """Copy the last observation and masks to the beginning for the next update."""
        self.obs[0] = self.obs[-1]
        self.masks[0] = self.masks[-1]
        self.truncated[0] = self.truncated[-1]

        # Reset step counter
        self.step = 0

    def get_minibatches_shared(self, num_mini_batch, add_role_id=False, mini_batch_size=None):
        """
        Create minibatches for training with a shared policy and critic.

        Args:
            num_mini_batch (int): Number of minibatches to create
            add_role_id (bool): Whether to add role IDs to observations (for centralized critic)
            mini_batch_size (int, optional): Size of each minibatch, if None will be calculated
                                            based on num_mini_batch

        Returns:
            Generator yielding minibatches for training
        """
        # Calculate total steps and minibatch size
        T, N= self.n_steps, self.n_agents
        batch_size = T*N

        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch

        # Ensure we have valid batch sizes
        if mini_batch_size <= 0:
            raise ValueError(f"Invalid mini_batch_size: {mini_batch_size}. Check num_mini_batch: {num_mini_batch}")

        # Create random indices for minibatches
        batch_inds = np.random.permutation(batch_size)

        # Preshape data to improve performance (only do this once)
        # Batch size is [T, N, feat_dim] -> [T*N, feat_dim]
        # if same policy, we don't do this now
        data = {
            'obs': self.obs[:-1].reshape(-1, *self.obs.shape[2:]),
            'actions': self.actions.reshape(-1, *self.actions.shape[2:]),
            'values': self.values[:-1].reshape(-1, 1),
            'returns': self.returns[:-1].reshape(-1, 1),
            'masks': self.masks[:-1].reshape(-1, 1),
            'old_action_log_probs': self.action_log_probs.reshape(-1, 1),
            'advantages': self.advantages.reshape(-1, 1),
        }

        # Build global observations
        obs_flat = self.obs[:-1].reshape(T, N, -1)  # (T, N, obs_dim)
        global_obs = self._build_global_obs(obs_flat, add_role_id=add_role_id)  # (T, N, N*obs_dim)

        # Stack and reshape
        data['global_obs'] = global_obs.reshape(T*N, -1)  # (T*N, N*obs_dim + N)

        # Yield minibatches
        start_ind = 0
        for _ in range(num_mini_batch):
            end_ind = min(start_ind + mini_batch_size, batch_size)
            if end_ind - start_ind < 1:  # Skip empty batches
                continue

            batch_inds_subset = batch_inds[start_ind:end_ind]

            batch = {
                key: torch.from_numpy(data[key][batch_inds_subset]).float().to(self.device)
                for key in data.keys()
            }

            # Yield the minibatch as a tuple
            yield (
                batch['obs'], # [B, obs_dim]
                batch['global_obs'], # [B, N*obs_dim+N]
                batch['actions'], # [B, act_dim]
                batch['values'], # [B, 1]
                batch['returns'], # [B, 1]
                batch['masks'], # [B, 1]
                batch['old_action_log_probs'], # [B, 1]
                batch['advantages'] # [B, 1]
            )

            start_ind = end_ind


    def get_minibatches_per_agent(self, num_mini_batch, add_role_id=False, mini_batch_size=None):
        """
        Create minibatches for training per agent.

        Args:
            num_mini_batch (int): Number of minibatches to create
            add_role_id (bool): Whether to add role IDs to observations (for centralized critic)
            mini_batch_size (int, optional): Size of each minibatch, if None will be calculated
                                            based on num_mini_batch

        Returns:
            Generator yielding minibatches for training
        """
        # Calculate total steps and minibatch size
        T, N= self.n_steps, self.n_agents
        batch_size = T

        if mini_batch_size is None:
            mini_batch_size = batch_size // num_mini_batch

        # Ensure we have valid batch sizes
        if mini_batch_size <= 0:
            raise ValueError(f"Invalid mini_batch_size: {mini_batch_size}. Check num_mini_batch: {num_mini_batch}")

        # Create random indices for minibatches
        batch_inds = np.random.permutation(batch_size)

        # Preshape data to improve performance (only do this once)
        # Batch size is [T, N, feat_dim]
        data = {
            'obs': self.obs[:-1],
            'actions': self.actions,
            'values': self.values[:-1],
            'returns': self.returns[:-1],
            'masks': self.masks[:-1],
            'old_action_log_probs': self.action_log_probs,
            'advantages': self.advantages,
        }

        # Build global observations
        obs_flat = self.obs[:-1].reshape(T, N, -1)  # (T, N, obs_dim)
        global_obs = self._build_global_obs(obs_flat, add_role_id=add_role_id)  # (T, N, N*obs_dim)

        # Stack and reshape
        data['global_obs'] = global_obs  # (T,N, N*obs_dim + N)

        # Yield minibatches
        start_ind = 0
        for _ in range(num_mini_batch):
            end_ind = min(start_ind + mini_batch_size, batch_size)
            if end_ind - start_ind < 1:  # Skip empty batches
                continue

            batch_inds_subset = batch_inds[start_ind:end_ind]

            batch = {
                key: torch.from_numpy(data[key][batch_inds_subset]).to(self.device)
                for key in data.keys()
            }

            # Yield the minibatch as a tuple
            yield (
                batch['obs'].transpose(1,0), # [N, T, obs_dim]
                batch['global_obs'].transpose(1,0), # [N, T, N*obs_dim + N]
                batch['actions'].transpose(1,0), # [N, T, act_dim]
                batch['values'].transpose(1,0), # [N, T, 1]
                batch['returns'].transpose(1,0), # [N, T, 1]
                batch['masks'].transpose(1,0), # [N, T, 1]
                batch['old_action_log_probs'].transpose(1,0), # [N, T, 1]
                batch['advantages'].transpose(1,0) # [N, T, 1]
            )

            start_ind = end_ind
    
    def _build_global_obs(self, obs_flat: np.ndarray, add_role_id: bool = False) -> np.ndarray:
        """
        Parameters
        ----------
        obs_flat : (T, N, obs_dim)

        Returns
        -------
        global_obs : (T, N, N*obs_dim) or (T, N, N*obs_dim + N) if add_role_id
            Row order: agent-0-t0, agent-1-t0, …, agent-(N-1)-t0,
                   agent-0-t1, …
        """
        T, N, D = obs_flat.shape

        # Create team-based ordering
        if (not hasattr(self, "_perm_idx")) or self._perm_idx.shape[0] != N:
            if getattr(self.args, "teams", None):
                # Create team-based ordering
                teams = self.args.teams
                team_of_agent = {a: t for t, members in enumerate(teams) for a in members}
                
                permutations = []
                for agent_idx in range(N):
                    # Find which team this agent belongs to
                    own_team = teams[team_of_agent[agent_idx]]

                    ordering = [agent_idx] #yourself
                    ordering += [m for m in own_team if m != agent_idx] #teammates
                    ordering += [op
                                for t_idx, team in enumerate(teams)
                                if t_idx != team_of_agent[agent_idx]
                                for op in team]   # opponents

                    permutations.append(ordering)

                
                # Create permutation indices
                self._perm_idx = np.array(permutations, dtype=np.int64) # (N, N)
            else:
                # cache the permutation matrix
                self._perm_idx = (np.arange(N)[:, None] + np.arange(N)[None, :]) % N  # (N,N)

        # obs_flat[:, perm, :] → (T, N, N, D)
        # use broadcast-friendly indexing: add axis for T so shapes line up
        #   obs_flat[ t , i , :]   where
        #     t = 0..T-1  (slice)
        #     i = perm matrix    (advanced index)
        obs_reo = obs_flat[:,  self._perm_idx, :]  # (T,N,N,D)

        global_obs = obs_reo.reshape(T, N, N * D)  # (T, N, N*D)

        if add_role_id:
            role_eye = np.eye(N, dtype=global_obs.dtype)  # (N, N)
            # role_ids = np.repeat(role_eye[np.newaxis, :, :], T, axis=0)  # (T, N, N)
            role_ids = np.broadcast_to(role_eye, (T, N, N)) # (T, N, N)
            global_obs = np.concatenate([global_obs, role_ids], axis=2)  # (T, N, N*D+N)

        return global_obs