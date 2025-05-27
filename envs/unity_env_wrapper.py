import numpy as np

import gymnasium
from unityagents import UnityEnvironment


class UnityEnvWrapper:
    def __init__(self, env_id, worker_id=0, seed=0, no_graphics=True):

        self.env_id = env_id
        self.seed = seed
        self._n_agents = 0

        print(f"Attempting to load Unity environment: {env_id}")
        self.env = UnityEnvironment(file_name=f"app/{env_id}.app",
                                    worker_id=worker_id,
                                    seed=self.seed,
                                    no_graphics=no_graphics)
        self._brain_names = self.env.brain_names
        self._brain_agents_num = {} # {brain_name: num_agents}
        self._observation_space_map = {} # {brain_name: observation_space}
        self._action_space_map = {} # {brain_name: action_space}

        if not self._brain_names:
            raise Exception("No brain_names found in the environment.")

        print(f"Environment loaded. Brain names: {self.brain_names}")
        env_info = self.env.reset(train_mode=True)

        for brain_name in self.brain_names:
            num_agents = len(env_info[brain_name].agents)
            print(f"Number of agents for {brain_name}: {num_agents}")
            self._brain_agents_num[brain_name] = num_agents
            self._n_agents += num_agents

            # Get observation and action space info
            brain = self.env.brains[brain_name]
            action_size = brain.vector_action_space_size
            action_type = brain.vector_action_space_type
            num_stacked_obs = brain.num_stacked_vector_observations
            obs_size = brain.vector_observation_space_size

            # We assume observations are always Box spaces
            self._observation_space_map[brain_name] = gymnasium.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_size * num_stacked_obs,),
                dtype=np.float32
            )

            if action_type == "continuous":
                # Assuming actions are in [-1, 1] range by default
                action_space = gymnasium.spaces.Box(
                    low=-1,
                    high=1,
                    shape=(action_size,),
                    dtype=np.float32
                )
            else: # Discrete
                action_space = gymnasium.spaces.Discrete(action_size)

            self._action_space_map[brain_name] = action_space

        print(f"Total agents: {self._n_agents}")
        print("Observation Spaces:", self._observation_space_map)
        print("Action Spaces:", self._action_space_map)

    @property
    def n_agents(self):
        return self._n_agents

    @property
    def brain_names(self):
        return self._brain_names

    @property
    def observation_spaces(self):
        """Return a flat list of observation spaces for all agents across all brains."""
        spaces = []
        for brain_name in self.brain_names:
            # Add the same observation space multiple times (once for each agent)
            spaces.extend([self._observation_space_map[brain_name]] * self._brain_agents_num[brain_name])
        return spaces
    
    @property
    def action_spaces(self):
        """Return a flat list of action spaces for all agents across all brains."""
        spaces = []
        for brain_name in self.brain_names:
            # Add the same action space multiple times (once for each agent)
            spaces.extend([self._action_space_map[brain_name]] * self._brain_agents_num[brain_name])
        return spaces

    def brain_agents(self, brain_name):
        """Return the number of agents for a specific brain."""
        return self._brain_agents_num[brain_name]

    def observation_space(self, brain_name):
        """Return the observation space for a specific brain."""
        return self._observation_space_map[brain_name]

    def action_space(self, brain_name):
        """Return the action space for a specific brain."""
        return self._action_space_map[brain_name]

    def reset(self, train_mode=True):
        """
        Reset the environment.

        Args:
            train_mode (bool, optional): Whether to reset the environment in training mode.
              Defaults to True.

        Returns:
            obs: List of observations for all agents.
        """
        env_info = self.env.reset(train_mode=train_mode)

        obs = []
        for brain_name in self.brain_names:
            brain_info = env_info[brain_name]
            num_agents = len(brain_info.agents)
            for i in range(num_agents):
                obs.append(brain_info.vector_observations[i])

        return obs

    def step(self, actions):
        """
        Step the environment.

        Args:
            actions (list): List of actions for all agents.

        Returns:
            next_obs: List of next observations for all agents.
            rewards: List of rewards for all agents.
            dones: List of done flags for all agents.
            truncs: List of truncation flags for all agents.
            info (dict): Additional info.
                - all_done (bool): Whether the episode is done.
                - all_trunc (bool): Whether the episode is truncated.
        """
        actions_dict = dict()
        start_idx = 0
        for brain_name in self.brain_names:
            agents_num = self._brain_agents_num[brain_name]
            actions_dict[brain_name] = actions[start_idx:start_idx + agents_num]
            start_idx += agents_num

        env_info = self.env.step(actions_dict)

        next_obs = []
        rewards = []
        dones = []
        truncs = []
        for brain_name in self.brain_names:
            brain_info = env_info[brain_name]
            num_agents = len(brain_info.agents)
            for i in range(num_agents):
                next_obs.append(brain_info.vector_observations[i])
                rewards.append(brain_info.rewards[i])
                dones.append(brain_info.local_done[i])
                truncs.append(brain_info.max_reached[i])

        all_done = any(dones)
        all_trunc = any(truncs)
        return next_obs, rewards, dones, truncs, {"all_done": all_done, 
                                                  "all_trunc": all_trunc}

    def close(self):
        self.env.close()