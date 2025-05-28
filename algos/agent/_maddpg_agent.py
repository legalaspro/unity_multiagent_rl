from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks.actors.deterministic_policy import DeterministicPolicy
from networks.critics.single_q_net import SingleQNet

class _MADDPGAgent:
    """
    DDPG Agent with Actor and Single Critic network
    """
    def __init__(self, args,
                 state_size,
                 action_space,
                 idx=0,
                 total_state_size=None,
                 total_action_size=None, device=torch.device("cpu")):
        """
        Initialize a DDPG agent.

        Args:
            args (argparse.Namespace): Hyperparameters
                gamma (float): Discount factor
                tau (float): Soft update parameter
                actor_lr (float): Learning rate for the actor
                critic_lr (float): Learning rate for the critic
                hidden_sizes (tuple): Sizes of hidden layers for networks
                use_max_grad_norm (bool): Whether to clip gradients
                max_grad_norm (float): Maximum gradient norm for clipping
            state_size (int): Dimension of the state space
            action_space (gymnasium.spaces.Space): Action space
            idx (int): Index of the agent (for logging)
            total_state_size (int): Total dimension of all agents' states (for centralized critic)
            total_action_size (int): Total dimension of all agents' actions (for centralized critic)
            device (torch.device): Device to use for training
        """
        self.state_size = state_size
        self.action_space = action_space
        self.idx = idx

        self.tau = args.tau
        self.gamma = args.gamma
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.device = device

        # Get action bounds from action space
        self.action_low = torch.tensor(action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_space.high, dtype=torch.float32, device=device)

        # Actor Networks (Local and Target)
        self.actor = DeterministicPolicy(state_size, action_space, args.hidden_sizes, device=device)
        self.actor_target = DeterministicPolicy(state_size, action_space, args.hidden_sizes, device=device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        # Critic Networks (Local and Target) if using centralized critic
        self.critic = SingleQNet(total_state_size, total_action_size, args.hidden_sizes, device=device)
        self.critic_target = SingleQNet(total_state_size, total_action_size, args.hidden_sizes, device=device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        # Initialize target networks with local network weights
        self.hard_update(self.critic_target, self.critic)
        self.hard_update(self.actor_target, self.actor)

    def act(self, state:torch.Tensor, add_noise=True, exploration_noise=0.0):
        """
        Returns actions for given state as per current policy.

        Args:
            state: Current state
            add_noise (bool): Whether to add noise for exploration
            exploration_noise (float): Exploration noise scale
        """
        self.actor.eval()
        with torch.no_grad():
            # Get action from network (already scaled to [action_low, action_high])
            action = self.actor(state)
        self.actor.train()

        if add_noise:
            # Scale noise by action range and noise_scale
            noise = torch.normal(0, exploration_noise, size=action.shape).to(self.device)
            action += noise

        # Clip to [action_low, action_high] range
        return torch.clamp(action, self.action_low, self.action_high)

    @torch.no_grad()
    def act_target(self, state:torch.Tensor):
        """
        Returns actions for given state as per current target policy.
        Keeps gradients for learning.

        Args:
            state: Current state (tensor)

        Returns:
            action: Action from target policy (tensor)
        """
        return self.actor_target(state)

    def update_critic(self, states, actions, rewards, next_states, next_actions, dones):
        """
        Update the critic network for a specific agent.

        Args:
            states: States for all agents [num_agents, batch_size, state_size]
            actions: Actions for all agents [num_agents, batch_size, action_size]
            rewards: Rewards for current agent [batch_size, 1]
            next_states: Next states for all agents [num_agents, batch_size, state_size]
            next_actions: Next  predicted actions for all agents [num_agents, batch_size, action_size]
            dones: Done flags for current agent [batch_size, 1]
        """
        with torch.no_grad():
             # Concatenate next actions for all agents (they're already tensors)
            next_actions_full = torch.cat(next_actions, dim=-1)
            next_states_full = torch.cat(next_states, dim=-1)

            # Compute target Q-value
            Q_targets_next = self.critic_target(next_states_full, next_actions_full)

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        states_full = torch.cat(states, dim=-1)
        actions_full = torch.cat(actions, dim=-1)

        # Compute critic loss
        Q_expected = self.critic(states_full, actions_full)
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Update the critic for the current agent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._clip_gradients(self.critic)
        self.critic_optimizer.step()

        return critic_loss.item(), critic_grad_norm

    def train(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        Implements the standard DDPG algorithm without centralized critics.

        Args:
            experiences (tuple): (states, actions, rewards, next_states, next_actions, dones)

        Returns:
            critic_loss (float): Loss of the critic network
            actor_loss (float): Loss of the actor network
        """
        train_info = {}

        states, actions, rewards, next_states, next_actions,\
            dones = experiences

        # Update Critic
        critic_loss, critic_grad_norm = self.update_critic(states, actions, rewards, next_states, next_actions, dones)

        # Update Actor
        # Compute actor loss
        actions_pred = []
        for i, actions_i in enumerate(actions):
            if i == self.idx:
                actions_pred.append(self.actor(states[i]))
            else: # Detach actions from other agents to prevent gradient flow
                actions_pred.append(actions_i.detach())

        actions_full_pred = torch.cat(actions_pred, dim=-1)
        states_full = torch.cat(states, dim=-1)
        # Compute actor loss using the agent's critic
        actor_loss = -self.critic(states_full, actions_full_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor)
        self.actor_optimizer.step()

        train_info['critic_loss'] = critic_loss
        train_info['actor_loss'] = actor_loss.item()
        if self.use_max_grad_norm:
            train_info['actor_grad_norm'] = actor_grad_norm
            train_info['critic_grad_norm'] = critic_grad_norm
       

        return train_info

    def hard_update(self, target_model, source_model):
        """
        Hard update model parameters.
        θ_target = θ_source

        Args:
            target_model: Model with weights to copy to
            source_model: Model with weights to copy from
        """
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(source_param.data)

    def _clip_gradients(self, network: nn.Module) -> Optional[float]:
        """Clip gradients and return gradient norm."""
        if self.use_max_grad_norm:
            return nn.utils.clip_grad_norm_(
                network.parameters(),
                self.max_grad_norm
            )
        return None