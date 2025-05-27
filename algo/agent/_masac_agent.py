import math
from typing import Optional
from gymnasium import spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from networks.actors.reparam_stochastic_policy import ReparamStochasticPolicy
from networks.critics.twin_q_net import TwinQNet

class _MASACAgent:
    """
    SAC Agent with Reparameterizable Actor and Twin Critic network
    """
    def __init__(self, args, state_size, action_space, idx=0, total_state_size=None, total_action_size=None,
                  device=torch.device("cpu")):
        """
        Initialize a SAC agent.

        Args:
            args (argparse.Namespace): Hyperparameters
                gamma (float): Discount factor
                tau (float): Soft update parameter
                actor_lr (float): Learning rate for the actor
                critic_lr (float): Learning rate for the critic
                autotune_alpha (bool): Whether to autotune alpha
                alpha_init (float): Entropy regularization coefficient
                gumbel_tau (float): Temperature for Gumbel-Softmax
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
        
        self.args = args
        
        self.tau = args.tau
        self.gamma = args.gamma
        self.autotune_alpha = args.autotune_alpha
        self.alpha_init = args.alpha_init
        self.use_max_grad_norm = args.use_max_grad_norm
        self.max_grad_norm = args.max_grad_norm
        self.device = device

        self.gumbel_tau = args.gumbel_tau

        # Actor Network
        self.actor = ReparamStochasticPolicy(
            state_size,
            action_space,
            args.hidden_sizes,
            gumbel_tau=args.gumbel_tau,
            device=device
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)

        # Critic Networks (Local and Target) if using centralized critic
        self.critic = TwinQNet(total_state_size, total_action_size, args.hidden_sizes, device=device)
        self.critic_target = TwinQNet(total_state_size, total_action_size, args.hidden_sizes, device=device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        if self.autotune_alpha:
            # Target entropy is -|A|, where |A| is the action space dimensionality (SAC paper) (e.g. -float(action_size))
            self.target_entropy = self._default_target_entropy(action_space)
            # Initialize log_alpha as a learnable parameter, ensuring alpha = exp(log_alpha) >= 0
            self.log_alpha = torch.log(torch.tensor(self.alpha_init, device=device)).requires_grad_(True)
            # Optimize log_alpha with Adam
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=args.actor_lr)
        else:
            # Fixed alpha if autotune is disabled
            self.log_alpha = torch.log(torch.tensor(self.alpha_init, device=device)).requires_grad_(False)

        # Initialize target networks with local network weights
        self.hard_update(self.critic_target, self.critic)

    @property
    def alpha(self):
        """Alpha is computed as exponential of log_alpha."""
        return self.log_alpha.exp()
    
    @property
    def gumbel_tau(self):
        """Get the current Gumbel-Softmax temperature."""
        return self._gumbel_tau

    @gumbel_tau.setter
    def gumbel_tau(self, value):
        """Set the Gumbel-Softmax temperature."""
        self._gumbel_tau = value

    def act(self,
            state:torch.Tensor,
            *,
            deterministic=False,
            gumbel_tau:Optional[float]=None):
        """
        Returns actions for given state as per current policy.

        Args:
            state: Current state
            deterministic (bool): Whether to use deterministic sampling
            gumbel_tau (float): Temperature for Gumbel-Softmax (discrete only)

        Returns:
            action: Action from policy (tensor)
        """
        gumbel_tau = gumbel_tau if gumbel_tau is not None else self.gumbel_tau
        self.actor.eval()
        with torch.no_grad():
            action, _, _ = self.sample(state,
                                       compute_log_prob=False,
                                       deterministic=deterministic,
                                       gumbel_tau=gumbel_tau)
        self.actor.train()

        return action

    def sample(
            self,
            state:torch.Tensor,
            *,
            compute_log_prob:bool=True,
            deterministic:bool=False,
            gumbel_tau:Optional[float]=None):
        """
        Sample an action from the policy.

        Args:
            state: Current state
            compute_log_prob (bool): Whether to compute log probabilities
            deterministic (bool): Whether to use deterministic sampling
            gumbel_tau (float): Temperature for Gumbel-Softmax (discrete only)

        Returns:
            action: Action from policy (tensor)
            log_prob: Log probability of the action (tensor)
            dist_info: Additional distribution-specific information
                    (Discrete: one-hot encoding, MultiDiscrete: list of one-hot encodings) (tensor)
        """
        gumbel_tau = gumbel_tau if gumbel_tau is not None else self.gumbel_tau
        return self.actor.sample(
            state,
            compute_log_prob=compute_log_prob,
            deterministic=deterministic,
            gumbel_tau=gumbel_tau)

    def update_critic(
            self,
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            next_log_probs,
            dones):
        """
        Update the critic network for a specific agent.

        Args:
            states: States for all agents [num_agents, batch_size, state_size]
            actions: Actions for all agents [num_agents, batch_size, action_size]
            rewards: Rewards for current agent [batch_size, 1]
            next_states: Next states for all agents [num_agents, batch_size, state_size]
            next_actions: Next predicted actions for all agents [num_agents, batch_size, action_size]
            next_log_probs: Log probabilities of next actions for all agents [num_agents, batch_size, 1]
            dones: Done flags for current agent [batch_size, 1]
        """
        with torch.no_grad():
            # Concatenate next actions and next states for all agents
            next_actions_full = torch.cat(next_actions, dim=-1)
            next_states_full = torch.cat(next_states, dim=-1)

            # Compute target Q-value
            Q1_next_targets, Q2_next_targets = self.critic_target(next_states_full, next_actions_full)
            Q_next_targets = torch.min(Q1_next_targets, Q2_next_targets)
            Q_next_targets -= self.alpha * next_log_probs[self.idx]

            # Compute Q targets for current states (y_i)
            Q_targets = rewards + (self.gamma * Q_next_targets * (1 - dones))

        # concatenate all states and actions
        states_full = torch.cat(states, dim=-1)
        actions_full = torch.cat(actions, dim=-1)

        # Compute critic loss
        Q1_expected, Q2_expected = self.critic(states_full, actions_full)
        Q1_loss = 0.5 * F.mse_loss(Q1_expected, Q_targets)
        Q2_loss = 0.5 * F.mse_loss(Q2_expected, Q_targets)
        critic_loss = Q1_loss + Q2_loss

        # Update the critic for the current agent
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = self._clip_gradients(self.critic)
        self.critic_optimizer.step()

        return critic_loss.item(), critic_grad_norm

    def train(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        Implements the standard SAC algorithm.

        Args:
            experiences (tuple): (states, actions, rewards, next_states, next_actions, next_log_probs, dones)

        Returns:
            train_info (dict): Dictionary containing training information
        """
        train_info = {}

        states, actions, rewards, next_states, next_actions, next_log_probs,\
            dones = experiences

        # Update Critic
        critic_loss, critic_grad_norm = self.update_critic(
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            next_log_probs,
            dones)

        # Update Actor
        actions_pred = []
        sampled_log_probs = None
        for i, actions_i in enumerate(actions):
            if i == self.idx:
                sampled_actions, sampled_log_probs, sampled_one_hot_actions = self.sample(states[i],
                                                                 compute_log_prob=True,
                                                                 deterministic=False,
                                                                 gumbel_tau=self.gumbel_tau)
                if self.action_space.__class__.__name__ == "Box":
                    actions_pred.append(sampled_actions)
                else: # Use one-hot encoding for discrete actions
                    actions_pred.append(sampled_one_hot_actions)
                sampled_log_probs = sampled_log_probs # [batch_size, 1]
            else: # Detach actions from other agents to prevent gradient flow
                actions_pred.append(actions_i.detach())

        actions_full_pred = torch.cat(actions_pred, dim=-1)
        states_full = torch.cat(states, dim=-1)
        # Compute actor loss using the agent's critic
        q1, q2 = self.critic(states_full, actions_full_pred)
        q_value = torch.min(q1, q2)
        actor_loss = torch.mean(self.alpha*sampled_log_probs - q_value)

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.actor)
        self.actor_optimizer.step()

        # Update alpha if autotuning is enabled
        if self.autotune_alpha:
            # NOTE: This is redundant, as we already computed log_probs in the actor loss
            # with torch.no_grad():
            #     _, log_probs, _ = self.sample(states[i], compute_log_prob=True, deterministic=False)
            #     log_probs = log_probs.mean()

            alpha_loss = self.alpha * (-sampled_log_probs.mean().detach() - self.target_entropy)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self.log_alpha], self.max_grad_norm)
            self.alpha_optimizer.step()
        else:
            alpha_loss = torch.tensor(0.0)


        train_info['critic_loss'] = critic_loss
        train_info['actor_loss'] = actor_loss.item()
        train_info['alpha_loss'] = alpha_loss.item()
        train_info['alpha'] = self.alpha.item()
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

    def _default_target_entropy(self, action_space: spaces.Space):
        """
        Default target entropy is -|A|, where |A| is the action space dimensionality (SAC paper) (e.g. -float(action_size))

            For discrete action spaces, it is the negative logarithm of the number of actions (e.g. -math.log(action_space.n))
            For multi-discrete action spaces, it is the sum of the negative logarithm of the number of actions for each dimension
            For continuous action spaces, it is the negative of the product of the action space shape (e.g. -float(math.prod(action_space.shape)))

        Args:
            action_space: Action space
        """
         # Get scale factor from args if available, otherwise default to 1.0
        scale = getattr(self.args, 'target_entropy_scale', 1.0)
        # The 0.98 factor slightly reduces the target entropy below this theoretical maximum.

        if isinstance(action_space, spaces.Discrete):
            return -scale * math.log(action_space.n) #  
        elif isinstance(action_space, spaces.MultiDiscrete):
            return -scale * sum(math.log(n) for n in action_space.nvec)
        elif isinstance(action_space, spaces.Box):
            return -float(math.prod(action_space.shape))
        else:
            raise NotImplementedError(f"Target entropy for action space {action_space} not implemented")