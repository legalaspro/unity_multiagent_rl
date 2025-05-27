"""
Multi-Agent Proximal Policy Optimization (MAPPO) Implementation
"""
from collections import defaultdict
import os
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from algo.marl_base import MultiAgentModule
from networks.actors.stochastic_policy import StochasticPolicy
from utils.env_tools import get_action_dim_for_critic_input
from networks.critics.v_net import VNet
from .agent._mappo_agent import _MAPPOAgent

class MAPPO(MultiAgentModule):
    """
    Multi-Agent Proximal Policy Optimization (MAPPO) with centralized critics.
    """

    def __init__(
            self,
            args,
            obs_spaces,
            action_spaces,
            device=torch.device("cpu")):
        """
        Initialize a MAPPO agent.

        Args:
            args (argparse.Namespace): Hyperparameters
                gamma (float): Discount factor
                gae_lambda (float): GAE lambda
                actor_lr (float): Learning rate for the actor
                critic_lr (float): Learning rate for the critic
            obs_spaces (list): List of observation spaces for each agent
            action_spaces (list): List of action spaces for each agent
            device (torch.device): Device to use for training
        """
        self.device = device
        self.args = args
        self.num_agents = len(obs_spaces)
        self.obs_spaces = obs_spaces
        self.action_spaces = action_spaces

        self.ppo_epoch = args.ppo_epoch
        self.use_clipped_value_loss = args.use_clipped_value_loss
        self.clip_param = args.clip_param
        self.num_mini_batch = args.num_mini_batch
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.entropy_coef = args.entropy_coef

        # Get observation and action sizes
        self.obs_sizes = [obs_space.shape[0] for obs_space in obs_spaces]
        self.action_sizes = [get_action_dim_for_critic_input(action_space) for action_space in action_spaces]

        # Calculate total observation size for centralized critic
        self.total_obs_size = sum(self.obs_sizes) #+ sum(self.action_sizes)
        if args.use_role_id:
            # Add role id to the observation
            self.total_obs_size += self.num_agents

        # Create agents
        if args.shared_policy:
            print("Using shared policy")
            self.shared_actor = StochasticPolicy(self.obs_sizes[0], self.action_spaces[0], 
                                      state_dependent_std=args.state_dependent_std,
                                      hidden_sizes=args.hidden_sizes,
                                      device=device)
            self.shared_actor_optimizer = optim.Adam(self.shared_actor.parameters(), lr=args.actor_lr)
        else:
            print("Using independent policies")
            self.agents = []
            for i in range(self.num_agents):
                agent = _MAPPOAgent(
                    args,
                    self.obs_sizes[i],
                    self.action_spaces[i],
                    idx=i,
                    total_state_size=self.total_obs_size,
                    device=self.device
                )
                self.agents.append(agent)

        
        if args.shared_critic:
            print("Using shared critic")
            # Centralized Critic - for multiagent PPO usually we have a shared critic
            # because Value functions are observation-dependent only
            self.shared_critic = VNet(self.total_obs_size, args.hidden_sizes, device=device)
            self.shared_critic_optimizer = optim.Adam(self.shared_critic.parameters(), lr=args.critic_lr)

    @torch.no_grad()
    def act(self, obs:torch.Tensor, deterministic=False):
        """
        Get actions from all agents based on current policy.

        Args:
            obs (list): List of observations for each agent
            deterministic (bool): Whether to use deterministic sampling
        Returns:
            actions (list): List of actions for each agent
        """
        actions = []
        log_probs = []
        if self.args.shared_policy:
            self.shared_actor.eval()
            # Get action from network (already scaled to [action_low, action_high])
            actions, log_probs = self.shared_actor(obs, deterministic=deterministic)
            self.shared_actor.train()
        else:
            for i, agent in enumerate(self.agents):
                action, log_prob = agent.act(obs[i], deterministic=deterministic)
                actions.append(action)
                log_probs.append(log_prob)

        # Keep API similar to off-policy  multi-agents
        if deterministic:
            return actions

        return actions, log_probs
    
    @torch.no_grad()
    def get_values(self, obs:torch.Tensor):
        """
        Get values from all agents based on current critic.

        Args:
            obs :  (N, obs_dim)          # single env step 
        """
        N, obs_dim = obs.shape
        device = obs.device

        if (not hasattr(self, "_perm_idx")) or self._perm_idx.shape[0] != N:
            # Create team-based ordering
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
                self._perm_idx = torch.tensor(permutations, dtype=torch.long,
                                          device=device) # (N, N)
            else:
                # cache the permutation matrix
                idx = torch.arange(N, device=device)
                self._perm_idx = (idx.unsqueeze(1) + idx) % N  # (N,N)

        # obs[perm] → (N, N, D)   (advanced indexing does the broadcast)
        obs_reordered = obs[self._perm_idx]  # (N,N,D)
        critic_input = obs_reordered.reshape(N, -1)  # (N, N*D)
        if self.args.use_role_id:
            role_eye   = torch.eye(N, device=device)   # (N, N)
            critic_input  = torch.cat(
                [critic_input, role_eye],
                dim=-1
            )    # (N, N*D + N)

        if self.args.shared_critic:
            v = self.shared_critic(critic_input)  # Shape: (N, 1)
        else:
            v = [agent.get_value(critic_input[i]) for i, agent in enumerate(self.agents)]
            v = torch.cat(v, dim=0)  # Shape: (N, 1)

        return v

    def update_critic(self, global_obs_batch, value_preds_batch, returns_batch):
        """
        Update the critic network for all agents.

        Args:
            global_obs_batch: Observations for all agents [batch_size, global_obs_size]
            value_preds_batch: Values for all agents [batch_size, 1]
            returns_batch: Returns for all agents [batch_size, 1]
        """
        train_info = {
            'critic_loss': 0,
            'critic_grad_norm': 0,
        }

        values = self.shared_critic(global_obs_batch)

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + torch.clamp(
                values - value_preds_batch,
                -self.clip_param,
                self.clip_param
            )
            value_losses = (values - returns_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - returns_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (values - returns_batch).pow(2).mean()


        # Update the critic for the current agent
        self.shared_critic_optimizer.zero_grad()
        value_loss.backward()
        critic_grad_norm = self._clip_gradients(self.shared_critic)
        self.shared_critic_optimizer.step()

        train_info['critic_loss'] = value_loss.item()
        if self.use_max_grad_norm:
            train_info['critic_grad_norm'] = critic_grad_norm

        return train_info
    
    def update_policy(self, obs_batch, actions_batch, old_action_log_probs_batch, advantages_batch):
        """
        Update the shared policy network for all agents.

        Args:
            obs_batch (): Observations for all agents [batch_size, obs_dim]
            actions_batch (): Actions for all agents [batch_size, action_dim]
            old_action_log_probs_batch (): Old action log probabilities for all agents [batch_size, 1]
            advantages_batch (): Advantages for all agents [batch_size, 1]
        """
        train_info = {}

         # Evaluate actions
        action_log_probs, dist_entropy = self.shared_actor.evaluate(
            obs_batch, actions_batch)

        # Calculate PPO ratio and KL divergence
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        clip_ratio = (torch.abs(ratio - 1) > self.clip_param).float().mean().item()

        # Actor Loss
        surr1 = ratio * advantages_batch
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages_batch
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coef * torch.mean(dist_entropy)
        actor_loss = policy_loss + entropy_loss

        # Update actor
        self.shared_actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.shared_actor)
        self.shared_actor_optimizer.step()

        train_info['actor_loss'] = actor_loss.item()
        train_info['entropy_loss'] = entropy_loss.item()
        train_info['approx_kl'] = approx_kl
        train_info['clip_ratio'] = clip_ratio
        if self.use_max_grad_norm:
            train_info['actor_grad_norm'] = actor_grad_norm

        return train_info

    def update(self, mini_batch):
        """
        Update policy and value parameters using a mini-batch of experiences.

        Args:
            mini_batch (dict): Dictionary containing mini-batch data

        Returns:
            dict: Dictionary containing training information
        """
        train_info = defaultdict(dict)

        # Extract data from mini-batch
        (obs_batch, global_obs_batch, actions_batch, values_batch, returns_batch, masks_batch,
            old_action_log_probs_batch, advantages_batch) = mini_batch
        
        if self.args.shared_critic:
            critic_train_info = self.update_critic(global_obs_batch, values_batch, returns_batch)
            train_info[0].update(critic_train_info)

        if self.args.shared_policy:
            # Update the policy for the current agent
            policy_train_info = self.update_policy(obs_batch, actions_batch, old_action_log_probs_batch,
                                            advantages_batch)
            train_info[0].update(policy_train_info)
        else:
            
            # Update the policy (and critic) for each agent
            for i, agent in enumerate(self.agents):
                agent_batch = (obs_batch[i], actions_batch[i],
                           old_action_log_probs_batch[i], advantages_batch[i],
                           values_batch[i], returns_batch[i], global_obs_batch)

                agent_train_infos = agent.train(agent_batch)
                train_info[i].update(agent_train_infos)

        return train_info

    
    def train(self, rollout_storage):
        """
        Update policy and value parameters for a specific agent using given batch of experience tuples.

        Args:
            rollout_storage (RolloutStorage): Rollout Storage

        Returns:
            critic_loss (float): Loss of the critic network
            actor_loss (float): Loss of the actor network
        """
        train_info = defaultdict(lambda: defaultdict(float))

        # Train for ppo_epoch iterations
        for _ in range(self.ppo_epoch):

            # Generate mini-batches
            if self.args.shared_policy:
                mini_batches = rollout_storage.get_minibatches_shared(self.num_mini_batch)
            else:
                mini_batches = rollout_storage.get_minibatches_per_agent(self.num_mini_batch)

            # Update for each mini-batch
            for mini_batch in mini_batches:
                
                # Update the policy and critic
                udpate_info = self.update(mini_batch)

                # Update training info
                for i, agent_info in udpate_info.items():
                    for k, v in agent_info.items():
                        train_info[i][k] += v

        # Calculate means
        num_updates = self.ppo_epoch * self.num_mini_batch
        for i, agent_info in train_info.items():
            for k in agent_info.keys():
                train_info[i][k] /= num_updates

        return train_info

    def save(self, path, save_args=False):
        """
        Save all agent models to a single file.

        Args:
            path (str): Path to save the models
        """
        # Create a dictionary to store all models
        models_dict = {}

        if self.args.shared_critic:
            models_dict['critic_state_dict'] = self.shared_critic.state_dict()
            models_dict['critic_optimizer'] = self.shared_critic_optimizer.state_dict()

        if self.args.shared_policy:
            models_dict['actor_state_dict'] = self.shared_actor.state_dict()
            models_dict['actor_optimizer'] = self.shared_actor_optimizer.state_dict()
        else:
            for i, agent in enumerate(self.agents):
                models_dict[f'actor_{i}_state_dict'] = agent.actor.state_dict()
                models_dict[f'actor_{i}_optimizer'] = agent.actor_optimizer.state_dict()
                if not self.args.shared_critic:
                    models_dict[f'critic_{i}_state_dict'] = agent.critic.state_dict()
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

        if self.args.shared_critic:
            self.shared_critic.load_state_dict(models_dict['critic_state_dict'])
            self.shared_critic_optimizer.load_state_dict(models_dict['critic_optimizer'])

        if self.args.shared_policy:
            self.shared_actor.load_state_dict(models_dict['actor_state_dict'])
            self.shared_actor_optimizer.load_state_dict(models_dict['actor_optimizer'])
        else:
            for i, agent in enumerate(self.agents):
                agent.actor.load_state_dict(models_dict[f'actor_{i}_state_dict'])
                agent.actor_optimizer.load_state_dict(models_dict[f'actor_{i}_optimizer'])
                if not self.args.shared_critic:
                    agent.critic.load_state_dict(models_dict[f'critic_{i}_state_dict'])
                    agent.critic_optimizer.load_state_dict(models_dict[f'critic_{i}_optimizer'])

        # Load args separately if they exist
        args_path = path + '.args'
        if os.path.exists(args_path):
            args_dict = torch.load(args_path, weights_only=False)
            self.args = args_dict['args']

    def _clip_gradients(self, network: nn.Module) -> Optional[float]:
        """Clip gradients and return gradient norm."""
        if self.use_max_grad_norm:
            return nn.utils.clip_grad_norm_(
                network.parameters(),
                self.max_grad_norm
            )
        return None
    
    @property
    def _modules(self):
        """Yield all modules that should be moved / toggled."""
        if self.args.shared_critic:
            yield self.shared_critic
        if self.args.shared_policy:
            yield self.shared_actor
        else:
            for agent in self.agents:
                yield agent.actor
                if not self.args.shared_critic:
                    yield agent.critic

    @property
    def _actor_modules(self):
        """Yield only the actor modules needed for inference."""
        if self.args.shared_policy:
            yield self.shared_actor
        else:
            for agent in self.agents:
                yield agent.actor
