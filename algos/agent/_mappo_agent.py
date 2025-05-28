
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from networks.actors.stochastic_policy import StochasticPolicy
from networks.critics.v_net import VNet

class _MAPPOAgent:
    """
    PPO Agent with Stochastic Actor
    """
    def __init__(
            self,
            args,
            state_size,
            action_space,
            *,
            idx=0,
            total_state_size=None,
            device=torch.device("cpu")):
        """
        Initialize a PPO agent.

        Args:
            args (argparse.Namespace): Hyperparameters
                gamma (float): Discount factor
                actor_lr (float): Learning rate for the actor
                hidden_sizes (array): Sizes of hidden layers for networks
                use_max_grad_norm (bool): Whether to clip gradients
                max_grad_norm (float): Maximum gradient norm for clipping
            state_size (int): Dimension of the state space
            action_space (gymnasium.spaces.Space): Action space
            idx (int): Index of the agent (for logging)
            total_state_size (int): Total dimension of all agents' states (for centralized critic)
            device (torch.device): Device to use for training
        """
        self.args = args
        self.state_size = state_size
        self.action_space = action_space
        self.idx = idx
        self.device = device

        self.gamma = args.gamma
        self.clip_param = args.clip_param
        self.entropy_coef = args.entropy_coef
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.use_clipped_value_loss = args.use_clipped_value_loss


        # Actor Network
        self.actor = StochasticPolicy(state_size, action_space, 
                                      state_dependent_std=args.state_dependent_std,
                                      hidden_sizes=args.hidden_sizes,
                                      device=device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        if not args.shared_critic:
            # Critic Network
            self.critic = VNet(total_state_size, args.hidden_sizes, device=device)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

    @torch.no_grad()
    def act(self, state:torch.Tensor, deterministic=False):
        """
        Returns actions for given state as per current policy.

        Args:
            state: Current state
            deterministic (bool): Whether to use deterministic sampling

        Returns:
            action: Action from policy (tensor)
            log_prob: Log probability of the action (tensor)
        """
        self.actor.eval()
        # Get action from network (already scaled to [action_low, action_high])
        action, log_prob = self.actor(state, deterministic=deterministic)
        self.actor.train()

        return action, log_prob
    
    @torch.no_grad()
    def get_value(self, state:torch.Tensor):
        """
        Get value from critic for a given state.

        Args:
            state: Current state
        Returns:
            value: Value of the state (tensor)
        """
        
        value = self.critic(state)

        # Ensure value has shape (1,1)
        if value.dim() == 1:
            value = value.unsqueeze(1)

        return value

    def evaluate(self,
                state:torch.Tensor,
                action:torch.Tensor):
        """
        Evaluate the log probability and entropy of a given action.

        Args:
            state: Current state
            action: Action to evaluate
        Returns:
            log_prob: Log probability of the action (tensor)
            entropy: Entropy of the action distribution (tensor)

        """
        return self.actor.evaluate(state, action)

    def update_critic(self, state_batch:torch.Tensor, value_preds_batch, returns_batch):
        """
        Update the critic network for a specific agent.

        Args:
            obs_batch (torch.Tensor): States
            value_preds_batch (torch.Tensor): Values
            returns_batch (torch.Tensor): Returns

        Returns:
            critic_loss (float): Loss of the critic network
            critic_grad_norm (float): Gradient norm of the critic network
        """
        train_info = {
            'critic_loss': 0,
            'critic_grad_norm': 0,
        }

        values = self.critic(state_batch) #(B, 1)
        value_preds_batch = value_preds_batch
        returns_batch = returns_batch

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
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        critic_grad_norm = self._clip_gradients(self.critic)
        self.critic_optimizer.step()

        train_info['critic_loss'] = value_loss.item()
        if self.use_max_grad_norm:
            train_info['critic_grad_norm'] = critic_grad_norm

        return train_info

    def train(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        Implements the standard PPO algorithm update for policy.

        Args:
            experiences (tuple): (obs_batch, actions_batch, old_action_log_probs, 
                        advantages, values_batch, returns_batch, global_obs_batch) #(B, x)

        Returns:
            train_info (dict): Dictionary containing training information
        """
        train_info = {}

        obs_batch, actions_batch, old_action_log_probs, \
            advantages, values_batch, returns_batch, global_obs_batch = experiences
        
        if not self.args.shared_critic:
            critic_train_info = self.update_critic(global_obs_batch, values_batch, returns_batch)
            train_info.update(critic_train_info)

        # Evaluate actions
        action_log_probs, dist_entropy = self.evaluate(
            obs_batch, actions_batch)

        # Calculate PPO ratio and KL divergence
        ratio = torch.exp(action_log_probs - old_action_log_probs)
        approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
        clip_ratio = (torch.abs(ratio - 1) > self.clip_param).float().mean().item()

        # Actor Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coef * torch.mean(dist_entropy)
        actor_loss = policy_loss + entropy_loss

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor)
        self.actor_optimizer.step()

        train_info['actor_loss'] = actor_loss.item()
        train_info['entropy_loss'] = entropy_loss.item()
        train_info['approx_kl'] = approx_kl
        train_info['clip_ratio'] = clip_ratio
        if self.use_max_grad_norm:
            train_info['actor_grad_norm'] = actor_grad_norm

        return train_info

    def _clip_gradients(self, network: nn.Module) -> Optional[float]:
        """Clip gradients and return gradient norm."""
        if self.use_max_grad_norm:
            return nn.utils.clip_grad_norm_(
                network.parameters(),
                self.max_grad_norm
            )
        return None