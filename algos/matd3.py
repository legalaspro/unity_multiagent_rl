"""
Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) Implementation
"""
import os
import torch

from algos.marl_base import MultiAgentModule
from utils.env_tools import get_action_dim_for_critic_input
from .agent._matd3_agent import _MATD3Agent

class MATD3(MultiAgentModule):
    """
    Multi-Agent Twin Delayed Deep Deterministic Policy Gradient (MATD3) with centralized critics.
    """

    def __init__(self, args, obs_spaces, action_spaces, device=torch.device("cpu")):
        """
        Initialize a MATD3 agent.

        Args:
            args (argparse.Namespace): Hyperparameters
                gamma (float): Discount factor
                tau (float): Soft update parameter
                actor_lr (float): Learning rate for the actor
                critic_lr (float): Learning rate for the critic
                hidden_sizes (tuple): Sizes of hidden layers for networks
            obs_spaces (list): List of observation spaces for each agent
            action_spaces (list): List of action spaces for each agent
            device (torch.device): Device to use for training
        """
        self.device = device
        self.args = args
        self.num_agents = len(obs_spaces)
        self.obs_spaces = obs_spaces
        self.action_spaces = action_spaces
        self.gamma = args.gamma
        self.tau = args.tau
        self.exploration_noise = args.exploration_noise

        # Get observation and action sizes
        self.obs_sizes = [obs_space.shape[0] for obs_space in obs_spaces]
        self.action_sizes = [get_action_dim_for_critic_input(action_space) for action_space in action_spaces]

        # Calculate total observation and action sizes for centralized critic
        self.total_obs_size = sum(self.obs_sizes)
        self.total_action_size = sum(self.action_sizes)

        # Total training iterations
        self.total_iterations = 1

        # Create agents
        self.agents = []
        for i in range(self.num_agents):
            agent = _MATD3Agent(
                args,
                self.obs_sizes[i],
                self.action_spaces[i],
                idx=i,
                total_state_size=self.total_obs_size,
                total_action_size=self.total_action_size,
                device=self.device
            )
            self.agents.append(agent)

    def act(self, obs, deterministic=False):
        """
        Get actions from all agents based on current policy.

        Args:
            obs (list): List of observations for each agent
            deterministic (bool): Whether to add noise for exploration
        """
        add_noise = not deterministic
        # actions = [agent.act(observation, add_noise, self.exploration_noise)
        #           for agent, observation in zip(self.agents, obs)]
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(obs[i], add_noise, self.exploration_noise))
        return actions

    @torch.no_grad()
    def act_target(self, obs):
        """
        Get actions from all agents based on target policies.

        Args:
            obs: Observations for all agents [num_agents, batch_size, obs_size]

        Returns:
            actions: List of actions for each agent
        """
        actions = [agent.act_target(observation)
                   for agent, observation in zip(self.agents, obs)]

        return actions

    def train(self, buffer):
        """
        Update policy and value parameters for a specific agent using given batch of experience tuples.

        Args:
            buffer (Buffer): Replay buffer

        Returns:
            critic_loss (float): Loss of the critic network
            actor_loss (float): Loss of the actor network
        """
        obs, actions, rewards, next_obs, dones,\
            obs_full, next_obs_full, actions_full = buffer.sample()

        # Get predicted next actions for all agents using target networks
        next_actions = self.act_target(next_obs)

        # Define agent train infos
        agent_train_infos = {}

        for agent_idx, agent in enumerate(self.agents):
            # Extract the agent's specific rewards and dones
            agent_rewards = rewards[agent_idx]
            agent_dones = dones[agent_idx]

            agent_train_infos[agent_idx] = agent.train(
                (obs, actions, agent_rewards,
                next_obs, next_actions, agent_dones),
                total_iterations=self.total_iterations)

        self.total_iterations += 1

        # Update target networks for all agents
        self.update_targets()

        return agent_train_infos

    def update_targets(self):
        """
        Soft update target networks for all agents.
        This should be called after all agents have been updated.
        """
        # print("\nUpdating target networks:")
        for i, agent in enumerate(self.agents):
            # Perform soft update
            self.soft_update(agent.actor_target, agent.actor)
            self.soft_update(agent.critic_target, agent.critic)

    def soft_update(self, target, source):
        """
        Soft update model parameters.
        θ_target = τ*θ_source + (1 - τ)*θ_target

        Args:
            target: Model with weights to update
            source: Model with weights to copy from
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)


    def save(self, path, save_args=False):
        """
        Save all agent models to a single file.

        Args:
            path (str): Path to save the models
        """
        # Create a dictionary to store all models
        models_dict = {}

        for i, agent in enumerate(self.agents):
            # Save actor and critic models
            models_dict[f'actor_{i}_state_dict'] = agent.actor.state_dict()
            models_dict[f'actor_{i}_optimizer'] = agent.actor_optimizer.state_dict()
            models_dict[f'critic_{i}__state_dict'] = agent.critic.state_dict()
            models_dict[f'critic_{i}_optimizer'] = agent.critic_optimizer.state_dict()

        # Save all models to a single file
        torch.save(models_dict, path)

        # Save args separately
        if save_args:
            args_path = path + '.args'
            torch.save({'args': self.args}, args_path)

    def load(self, path):
        """
        Load all agent models from a single file.

        Args:
            path (str): Path to load the models from
        """
        # Load the dictionary containing all models
        models_dict = torch.load(path, map_location=self.device, weights_only=True)

        # Load agent models and optimizers
        for i, agent in enumerate(self.agents):
            # Load actor model
            actor_key = f'actor_{i}_state_dict'
            agent.actor.load_state_dict(models_dict[actor_key])
            agent.actor_target.load_state_dict(models_dict[actor_key])
            agent.actor_optimizer.load_state_dict(models_dict[f'actor_{i}_optimizer'])
            # Load critic model (handle both old and new key formats)
            critic_key = f'critic_{i}_state_dict' if f'critic_{i}_state_dict' in models_dict \
                else f'critic_{i}__state_dict'
            agent.critic.load_state_dict(models_dict[critic_key])
            agent.critic_target.load_state_dict(models_dict[critic_key])
            agent.critic_optimizer.load_state_dict(models_dict[f'critic_{i}_optimizer'])

        # Load args separately if they exist
        args_path = path + '.args'
        if os.path.exists(args_path):
            args_dict = torch.load(args_path, weights_only=False)
            self.args = args_dict['args']
    
    @property
    def _modules(self):
        """Yield all modules that should be moved / toggled."""
        for agent in self.agents:
            yield agent.actor
            yield agent.actor_target
            yield agent.critic
            yield agent.critic_target
    
    @property
    def _actor_modules(self):
        """Yield only the *actors* (modules required for acting in the env)."""
        for agent in self.agents:
            yield agent.actor
