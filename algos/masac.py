"""
Multi-Agent Soft Actor-Critic (MASAC) Implementation
"""
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from algos.marl_base import MultiAgentModule
from utils.env_tools import get_action_dim_for_critic_input
from .agent._masac_agent import _MASACAgent
from networks.critics.twin_q_net import TwinQNet

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
                device=self.device,
                parent=self
            )
            self.agents.append(agent)

        if args.shared_critic:
            print("Using shared critic")
            # Centralized Critic - for multiagent SAC usually we have a shared critic
            global_obs_size = sum(self.obs_sizes)
            global_action_size = max(self.action_sizes) * 4 # Use max action size for all agents
            self.max_action_size = max(self.action_sizes)
            print(f"global_obs_size: {global_obs_size}, global_action_size: {global_action_size}")
            self.critic = TwinQNet(global_obs_size, global_action_size, args.hidden_sizes, device=device)
            self.critic_target = TwinQNet(global_obs_size, global_action_size, args.hidden_sizes, device=device)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

            self.critic_alpha_mode = getattr(args, "critic_alpha_mode", "per_agent")
            if self.critic_alpha_mode == "shared":
                # single learnable log α_C for the critic
                self.log_alpha_C = torch.tensor(
                    float(args.alpha_init), device=device).log().requires_grad_(True)
                self.alpha_C_optimizer = optim.Adam([self.log_alpha_C], lr=args.alpha_lr)

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

    @property
    def alpha_C(self):
        if self.critic_alpha_mode == "shared":
            return self.log_alpha_C.exp().item()
        return None

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

    def get_action_values(self, obs:torch.Tensor, actions:torch.Tensor, agent_idx:int):
        """
        Get action values from the critic for a given observations and actions.

        Args:
            obs (torch.Tensor): Observations for all agents [num_agents, batch_size, obs_size]
            actions (torch.Tensor): Actions for all agents [num_agents, batch_size, action_size]
            agent_idx (int): Index of the agent for which to get the action values
        Returns:
            action_values (torch.Tensor): Action values for the given agent [batch_size, 1]
        """
        dev = obs[0].device
        perm_idx = self._get_permutation_indices(dev)          # (N,N)

        perm = perm_idx[agent_idx]                             # List[int]
        obs_full     = self._concat_joint(obs,     perm)       # (B, ΣS)
        actions_full = self._concat_joint(actions, perm, max_A=self.max_action_size)       # (B, ΣA)

        return self.critic(obs_full, actions_full)             # tuple (q1,q2)

    def update_critic(self,
            states,
            actions,
            rewards,
            next_states,
            next_actions,
            next_log_probs,
            dones):
        """
        Update the shared critic network.

        Args:
            states: States for all agents [num_agents, batch_size, state_size]
            actions: Actions for all agents [num_agents, batch_size, action_size]
            rewards: Rewards for current agent [num_agents, batch_size, 1]
            next_states: Next states for all agents [num_agents, batch_size, state_size]
            next_actions: Next predicted actions for all agents [num_agents, batch_size, action_size]
            next_log_probs: Log probabilities of next actions for all agents [num_agents, batch_size, 1]
            dones: Done flags for current agent [num_agents, batch_size, 1]
        """
        B   = states[0].shape[0]
        N   = self.num_agents
        dev = states[0].device

        # ---------- permutation matrix (build once per device / agent-count)
        self._perm_idx = self._get_permutation_indices(dev)      # (N,N)

        # ---------- build joint (B , ΣD) for every agent  ------------------
        states_full, next_states_full = [], []
        actions_full, next_actions_full = [], []
        logp_full, r_full, d_full = [], [], []

        for i in range(N):
            perm = self._perm_idx[i]

            states_full.append(self._concat_joint(states, perm))
            next_states_full.append(self._concat_joint(next_states, perm))

            actions_full.append(self._concat_joint(actions, perm, max_A=self.max_action_size))
            next_actions_full.append(self._concat_joint(next_actions, perm, max_A=self.max_action_size))

            logp_full.append(next_log_probs[i])      # (B,1)
            r_full.append(rewards[i])                # (B,1)
            d_full.append(dones[i].float())          # (B,1)

        # ---------- flatten lists to (N*B , ·)
        states_full = torch.cat(states_full, dim=0)
        next_states_full = torch.cat(next_states_full, dim=0)
        actions_full = torch.cat(actions_full, dim=0)
        next_actions_full = torch.cat(next_actions_full, dim=0)

        next_logp_flat = torch.cat(logp_full, dim=0)           # (N*B,1)
        r_flat    = torch.cat(r_full, dim=0)           # (N*B,1)
        d_flat    = torch.cat(d_full, dim=0)

        # ---------- per-agent α replicated row-wise
        # alpha_vec = torch.tensor(                                            # (N,1)→(N*B,1)
        #     [ag.alpha.item() for ag in self.agents], dtype=next_logp_flat.dtype, # α₀…αₙ₋₁
        #     device=next_logp_flat.device
        # ).view(N,1).repeat_interleave(B, 0)         # (N*B,1)

        with torch.no_grad():
            # Compute target Q-value
            Q1_next_targets, Q2_next_targets = self.critic_target(next_states_full, next_actions_full)
            Q_next_targets = torch.min(Q1_next_targets, Q2_next_targets)

            if self.critic_alpha_mode == "per_agent":
                # ------- per-agent α replicated row-wise
                alpha_vec = torch.tensor(
                    [ag.alpha.item() for ag in self.agents],
                    dtype=next_logp_flat.dtype, device=self.device
                ).view(N,1).repeat_interleave(B, 0)           # (N*B,1)
                Q_next_targets = Q_next_targets - alpha_vec * next_logp_flat

            else:  # "shared"
                # ------- HASAC style (single α_C) --------------------------
                joint_logp = torch.stack(next_log_probs, 0).sum(0)   # (B,1)
                joint_logp = joint_logp.repeat(N, 1)                 # (N*B,1)
                alpha_C    = self.log_alpha_C.exp()
                Q_next_targets  = Q_next_targets - alpha_C * joint_logp

            # Compute Q targets for current states (y_i)
            Q_targets = r_flat + (self.gamma * Q_next_targets * (1 - d_flat))

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

        # ------------ update shared α_C (if enabled) ------------------------
        if self.critic_alpha_mode == "shared" and self.autotune_alpha:
            # joint_logp for the *current* minibatch (size B) — we cached it above
            joint_logp_batch = torch.stack(next_log_probs, 0).sum(0)  # (B,1)
            # sum of target entropies for all agents (a constant)
            H_target_sum = sum(agent.target_entropy for agent in self.agents)
            alpha_loss_C = self.log_alpha_C.exp() * (
                -joint_logp_batch.mean().detach() - H_target_sum)

            self.alpha_C_optimizer.zero_grad()
            alpha_loss_C.backward()
            nn.utils.clip_grad_norm_([self.log_alpha_C], self.args.max_grad_norm)
            self.alpha_C_optimizer.step()

        return critic_loss.item(), critic_grad_norm

    def train(self, buffer):
        """
        Update policy and value parameters for a specific agent using given batch of experience tuples.

        Args:
            buffer (Buffer): Replay buffer

        Returns:
            critic_loss (float): Loss of the critic network
            actor_loss (float): Loss of the actor network
        """
        #  Define agent train infos
        agent_train_infos = {}

        obs, actions, rewards, next_obs, dones,\
            obs_full, next_obs_full, actions_full = buffer.sample()

        # Get predicted next actions for all agents using target networks
        next_actions, next_log_probs = self._sample_next_actions_for_q_target(next_obs)

        if self.args.shared_critic:
            critic_loss, critic_grad_norm = self.update_critic(
                obs, actions, rewards, next_obs, next_actions, next_log_probs, dones)

        for agent_idx, agent in enumerate(self.agents):
            # Extract the agent's specific rewards and dones
            agent_rewards = rewards[agent_idx]
            agent_dones = dones[agent_idx]

            agent_train_infos[agent_idx] = agent.train(
                (obs, actions, agent_rewards,
                next_obs, next_actions, next_log_probs, agent_dones))

        # Add critic loss to the first agent's train info
        if self.args.shared_critic:
            agent_train_infos[0]['critic_loss'] = critic_loss
            if self.args.use_max_grad_norm:
                agent_train_infos[0]['critic_grad_norm'] = critic_grad_norm

        # Update target networks for all agents
        self.update_targets()

        return agent_train_infos

    def update_targets(self):
        """
        Soft update target networks for all agents.
        This should be called after all agents have been updated.
        """
        if self.args.shared_critic:
            self.soft_update(self.critic_target, self.critic)
        else:
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

        if self.args.shared_critic:
            models_dict['critic_state_dict'] = self.critic.state_dict()
            models_dict['critic_optimizer'] = self.critic_optimizer.state_dict()

        for i, agent in enumerate(self.agents):
            # Save actor and critic models
            models_dict[f'actor_{i}_state_dict'] = agent.actor.state_dict()
            models_dict[f'actor_{i}_optimizer'] = agent.actor_optimizer.state_dict()
            if not self.args.shared_critic:
                models_dict[f'critic_{i}_state_dict'] = agent.critic.state_dict()
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

        if self.args.shared_critic:
            self.critic.load_state_dict(models_dict['critic_state_dict'])
            self.critic_target.load_state_dict(models_dict['critic_state_dict'])
            self.critic_optimizer.load_state_dict(models_dict['critic_optimizer'])

        # Load agent models and optimizers
        for i, agent in enumerate(self.agents):
            # Load actor model
            actor_key = f'actor_{i}_state_dict'
            agent.actor.load_state_dict(models_dict[actor_key])
            agent.actor_optimizer.load_state_dict(models_dict[f'actor_{i}_optimizer'])

            if not self.args.shared_critic:
                # Load critic model (handle both old and new key formats)
                critic_key = f'critic_{i}_state_dict' if f'critic_{i}_state_dict' in models_dict \
                    else f'critic_{i}__state_dict'

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
        if self.args.shared_critic:
            yield self.critic
            yield self.critic_target
        for agent in self.agents:
            yield agent.actor
            if not self.args.shared_critic:
                yield agent.critic
                yield agent.critic_target

    @property
    def _actor_modules(self):
        """Yield only the *actors* (modules required for acting in the env)."""
        for agent in self.agents:
            yield agent.actor

    def _clip_gradients(self, network: nn.Module) -> Optional[float]:
        """Clip gradients and return gradient norm."""
        if self.args.use_max_grad_norm:
            return nn.utils.clip_grad_norm_(
                network.parameters(),
                self.args.max_grad_norm
            )
        return None

    def _get_permutation_indices(self, device=None):
        """
        Create permutation indices for team-based or shift-based ordering.

        Args:
            device (torch.device): Device to place the tensor on

        Returns:
            torch.Tensor: Permutation indices of shape [num_agents, num_agents]
        """
        if hasattr(self, "_perm_idx") and self._perm_idx.shape[0] == self.num_agents:
            return self._perm_idx

        N = self.num_agents
        if device is None:
            device = self.device

        # Create team-based ordering
        if hasattr(self.args, "teams") and self.args.teams:
            teams = self.args.teams
            team_of_agent = {a: t for t, members in enumerate(teams) for a in members}

            permutations = []
            for agent_idx in range(N):
                # Find which team this agent belongs to
                own_team = teams[team_of_agent[agent_idx]]

                ordering = [agent_idx]  # yourself first
                ordering += [m for m in own_team if m != agent_idx]  # teammates
                ordering += [op for t_idx, team in enumerate(teams)
                        if t_idx != team_of_agent[agent_idx]
                        for op in team]  # opponents

                permutations.append(ordering)
        else:
            # Simple shift-based ordering if no teams
            permutations = []
            for i in range(N):
                # Start with self, then others in order
                perm = [(i + j) % N for j in range(N)]
                permutations.append(perm)

        # Create permutation indices
        self._perm_idx = torch.tensor(permutations, dtype=torch.long,
                                          device=device) # (N, N)

        return  self._perm_idx

    def _concat_joint(self, tensor_list, perm, max_A=None):
        """
        Concatenate tensors from a list according to a permutation.

        Args:
            tensor_list (list): List of tensors to concatenate
            perm (list): Permutation of indices
            max_A (int): Maximum action size (for padding)

        Returns:
            torch.Tensor: Concatenated tensor
        """
        if max_A is None:
            return torch.cat([tensor_list[j] for j in perm], dim=-1)

        tensors_to_concat = []
        for j in perm:
            t = tensor_list[j]
            if t.shape[-1] < max_A:
                t = F.pad(t, (0, max_A - t.shape[-1]))  # right-pad zeros
            tensors_to_concat.append(t)

        return torch.cat(tensors_to_concat, dim=-1)



# ---- 5'. build joint log-π once per env transition (B,1) ----------
# joint_logp = torch.stack(next_log_probs, 0).sum(0)   # (B,1)

# # repeat for N copies in flattened tensor
# joint_logp = joint_logp.repeat(N, 1)                 # (N*B,1)

# alpha_C   = self.alpha_C         # scalar tensor
# with torch.no_grad():
#     q1_next, q2_next = self.critic_target(next_states_full,
#                                           next_actions_full)
#     q_next  = torch.min(q1_next, q2_next)
#     q_next -= alpha_C * joint_logp
#     y       = r_flat + self.gamma * q_next * (1 - d_flat)

# alpha_loss = alpha_C * (-joint_logp.detach()[:B].mean() - H_star_sum)
# where H_star_sum = sum(H_star_i) is the sum of target entropies for all
# agents (a constant).
# joint_logp = torch.stack(next_log_probs, 0).sum(0)     # (B,1)
# joint_logp = joint_logp.repeat(N, 1)                   # (N*B,1)

# Q_next = torch.min(Q1_next, Q2_next) - self.alpha_C * joint_logp

# joint_logp = torch.stack(next_log_probs, 0).sum(0)    # (B,1)
# joint_logp = joint_logp.repeat(N, 1)                  # (N*B,1)

# Q_next_targets = torch.min(Q1_next_targets, Q2_next_targets) \
#                  - self.alpha_C * joint_logp
# alpha_loss = self.alpha_C * (-joint_logp[:B].detach().mean() - H_star_sum)
# Variant	TD target term you subtract	What it means	When people use it
# A. Per-agent temperatures	(\sum_{i=1}^{N}\alpha_i \log\pi_i(a_i|s_i))  (one α per agent)	Critic approximates N independent soft-Q functions that share weights but treat each agent's policy entropy separately.
# B. One shared temperature	(\alpha_C \sum_{i=1}^{N}\log\pi_i(a_i|s_i))  (single α)	Critic approximates a joint soft-Q of the whole team; the exploration temperature is coupled across agents.

# Both subtract a penalty that is exactly the entropy term used in the corresponding policy-gradient derivation, so they are self-consistent:

# Q_target = r + γ(min_{k=1,2} Q_tgt,k(s',a') - penalty term)