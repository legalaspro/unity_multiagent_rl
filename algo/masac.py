"""
Multi-Agent Soft Actor-Critic (MASAC) Implementation
"""
import os
from typing import Optional

import torch

from algo.marl_base import MultiAgentModule
from utils.env_tools import get_action_dim_for_critic_input
from .agent._masac_agent import _MASACAgent

class MASAC(MultiAgentModule):
    """
    Multi-Agent Soft Actor-Critic (MASAC) with centralized critics.
    """

    def __init__(self, args, obs_spaces, action_spaces, device=torch.device("cpu")):
        """
        Initialize a MASAC agent.

        Args:
            args (argparse.Namespace): Hyperparameters
                gamma (float): Discount factor
                tau (float): Soft update parameter
                actor_lr (float): Learning rate for the actor
                critic_lr (float): Learning rate for the critic
                autotune_alpha (bool): Whether to autotune alpha
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
        self.autotune_alpha = args.autotune_alpha
        self.gumbel_tau = args.gumbel_tau

        # Get observation and action sizes
        self.obs_sizes = [obs_space.shape[0] for obs_space in obs_spaces]
        self.action_sizes = [get_action_dim_for_critic_input(action_space) for action_space in action_spaces]

        # Calculate total observation and action sizes for centralized critic
        self.total_obs_size = sum(self.obs_sizes)
        self.total_action_size = sum(self.action_sizes)

        # Create agents
        self.agents = []
        for i in range(self.num_agents):
            agent = _MASACAgent(
                args,
                self.obs_sizes[i],
                self.action_spaces[i],
                idx=i,
                total_state_size=self.total_obs_size,
                total_action_size=self.total_action_size,
                device=self.device
            )
            self.agents.append(agent)
    
    @property
    def gumbel_tau(self):
        """Get the current Gumbel-Softmax temperature."""
        return self._gumbel_tau

    @gumbel_tau.setter
    def gumbel_tau(self, value):
        """Set the Gumbel-Softmax temperature."""
        self._gumbel_tau = value
        # Update all agents
        if hasattr(self, 'agents'):
            for agent in self.agents:
                agent.gumbel_tau = value

    def act(self, obs:torch.Tensor,
            *,
            deterministic=False):
        """
        Get actions from all agents based on current policy.

        Args:
            obs (list): List of observations for each agent
            deterministic (bool): Whether to use deterministic sampling
        Returns:
            actions (list): List of actions for each agent
        """
        # actions = [agent.act(observation, deterministic=deterministic, gumbel_tau=self.gumbel_tau)
        #           for agent, observation in zip(self.agents, obs)]
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(obs[i], deterministic=deterministic, gumbel_tau=self.gumbel_tau))
        return actions

    def _sample(self, obs:torch.Tensor,
                *,
                compute_log_prob=True,
                deterministic=False,
                gumbel_tau:Optional[float]=None):
        """
        Sample actions and log probabilities for all agents.
        Args:
            obs (list): List of observations for each agent
            compute_log_prob (bool): Compute and return log probabilities of the action. Defaults to True.
            deterministic (bool): If True, return the mean action. Defaults to False.
            gumbel_tau (float): Temperature for Gumbel-Softmax (discrete only)
        Returns:
            actions (list): List of actions for each agent
            log_probs (list): List of log probabilities for each agent (if compute_log_prob is True)
            dist_infos (list): List of distribution-specific information for each agent
        """
        samples = [agent.sample(observation, compute_log_prob=compute_log_prob,
                                deterministic=deterministic, gumbel_tau=gumbel_tau)
                  for agent, observation in zip(self.agents, obs)]
        actions, log_probs, dist_infos = zip(*samples)
        return list(actions), list(log_probs), list(dist_infos)

    @torch.no_grad()
    def _sample_next_actions_for_q_target(self, next_obs:torch.Tensor):
        """
        Samples next actions (critic representation) and their log_probs from each agent's
        target actor (or current actor if no separate target actor).
        This is for calculating the Q-target.
        """
        next_actions, next_log_probs, next_dist_infos = self._sample(next_obs, gumbel_tau=self.gumbel_tau)

        # We assume all agents have the same action space type
        if self.action_spaces[0].__class__.__name__ == "Box":
            return next_actions, next_log_probs
        else: # Discrete/MultiDiscrete
            return next_dist_infos, next_log_probs

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
        next_actions, next_log_probs = self._sample_next_actions_for_q_target(next_obs)

        # Define agent train infos
        agent_train_infos = {}

        for agent_idx, agent in enumerate(self.agents):
            # Extract the agent's specific rewards and dones
            agent_rewards = rewards[agent_idx]
            agent_dones = dones[agent_idx]

            agent_train_infos[agent_idx] = agent.train(
                (obs, actions, agent_rewards,
                next_obs, next_actions, next_log_probs, agent_dones))

        # Update target networks for all agents

        self.update_targets()

        return agent_train_infos

    def update_targets(self):
        """
        Soft update target networks for all agents.
        This should be called after all agents have been updated.
        """
        # print("\nUpdating target networks:")
        for agent in self.agents:
            # Perform soft update
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
            if self.autotune_alpha:
                models_dict[f'log_alpha_{i}'] = agent.log_alpha
                models_dict[f'log_alpha_{i}_optimizer'] = agent.alpha_optimizer.state_dict()

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
            agent.actor_optimizer.load_state_dict(models_dict[f'actor_{i}_optimizer'])
            # Load critic model
            critic_key = f'critic_{i}_state_dict'
            agent.critic.load_state_dict(models_dict[critic_key])
            agent.critic_target.load_state_dict(models_dict[critic_key])
            agent.critic_optimizer.load_state_dict(models_dict[f'critic_{i}_optimizer'])
            if self.autotune_alpha:
                agent.log_alpha = models_dict[f'log_alpha_{i}']
                agent.alpha_optimizer.load_state_dict(models_dict[f'log_alpha_{i}_optimizer'])

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
            yield agent.critic
            yield agent.critic_target
    
    @property
    def _actor_modules(self):
        """Yield only the *actors* (modules required for acting in the env)."""
        for agent in self.agents:
            yield agent.actor
