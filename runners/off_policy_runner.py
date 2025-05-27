from collections import defaultdict, deque
import numpy as np
import os
import torch
from buffers.replay_buffer import ReplayBuffer
from algo import ALGO_REGISTRY
import gymnasium

from evals.unity_evaluator import UnityEvaluator
from evals.competitive_evaluator import CompetitiveEvaluator


class OffPolicyRunner:
    """
    Runner class for off-policy algorithms.
    """
    def __init__(self, args, env, logger, device):
        """
        Initialize the runner.

        Args:
            args: Arguments containing training parameters
            env: Environment to train on
            logger: Logger to use for logging
            device: Torch device to use for computations
        """
        self.args = args
        self.device = device
        self.logger = logger

        self.env = env

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            args.buffer_size,
            args.batch_size,
            env.observation_spaces,
            env.action_spaces,
            device=self.device,
            n_step=args.n_step if args.n_step else 1,
            gamma=args.gamma
        )

        self.is_discrete = [isinstance(sp, gymnasium.spaces.Discrete) for sp in env.action_spaces]

        # Create agent
        self.multi_agent = ALGO_REGISTRY[args.algo](args,
                            self.env.observation_spaces,
                            self.env.action_spaces,
                            self.device)

        print(f'Number of agents: {self.multi_agent.num_agents}')
        print(f'Action spaces: {self.multi_agent.action_spaces}')
        print(f'Observation spaces: {self.multi_agent.obs_spaces}')

        # Create evaluator
        if self.args.zerosum:
            print("Zero-sum game detected. Enabling competitive evaluation.")
            self.evaluator = CompetitiveEvaluator(logger, args,
                                                    make_agent_snapshot=lambda: self.multi_agent.policy_snapshot("cpu"))
            self.is_zerosum = True
        else:
            print("Non-zero-sum game detected. Normal evaluation.")
            self.evaluator = UnityEvaluator(logger, args,
                                            make_agent_snapshot=lambda: self.multi_agent.policy_snapshot("cpu"))
            self.is_zerosum = False

    def warmup(self):
        """
        Warmup the replay buffer with random actions.

        Returns:
            obs (list): List of last observations for all agents.
        """
        obs = self.env.reset()

        for _ in range(self.args.learning_starts):
            #simulated actions
            actions = self._random_actions()
            next_obs, rewards, dones, truncs, info = self.env.step(actions)
            if info["all_done"] or info["all_trunc"]:
                next_obs = self.env.reset()
            self.replay_buffer.add(obs, actions, rewards, next_obs, dones)
            obs = next_obs

        return obs

    def _random_actions(self):
        """
        Simulate random actions based on the action spaces.

        Returns:
            actions (list): List of simulated actions for all agents.
        """
        return [sp.sample() for sp in self.env.action_spaces]

    def train(self):
        """
        Train the agent.
        """

        print("Warming up replay buffer...")
        obs = self.warmup()
        print("Replay buffer warmup complete. Let's start training.")

        scores_history = []
        episode_length_history = []
        steps_history = []
        if self.is_zerosum:
            self._cum_wins = np.zeros(3, dtype=int)  # [t0_wins, t1_wins, draws]
            self._recent_outcomes = deque(maxlen=100)   # 0=t0 win, 1=t1 win, 2=draw
            game_outcomes_history = []

        episode_rewards = np.zeros(self.multi_agent.num_agents)
        episode_length = 0
        episode_num = 0

        best_mean_score = -np.inf

        for step in range(1, self.args.max_steps + 1):

            # Evaluate
            if step % self.args.eval_interval == 0 or step == 1:
                current_best = self.evaluator.run(step)
                if current_best > best_mean_score:
                    best_mean_score = current_best
                    save_path = os.path.join(self.logger.dir_name, f"best-torch.model")
                    self.multi_agent.save(save_path)
                    self.logger.log_model(
                        file_path=save_path,
                        name="best-model",
                        artifact_type="model",
                        metadata={"max_score": best_mean_score,
                                "step": step},
                        alias="latest"
                    )
                    print(f"Saved best model with score {best_mean_score:.4f}")

            # interact
            with torch.no_grad():
                obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=self.device)
                # [torch.from_numpy(o).float().to(self.device) for o in obs]
                actions_t = self.multi_agent.act(obs_t, deterministic=False)
                actions = [a.cpu().numpy() for a in actions_t]
            next_obs, rewards, dones, truncs, info = self.env.step(actions)
            episode_rewards += rewards
            episode_length += 1

            # Handle episode termination
            if info["all_done"]:
                next_obs = self.env.reset()
                for i in range(self.multi_agent.num_agents):
                    self.logger.add_scalar(f'train/agent{i}_rewards', episode_rewards[i], step)
                self.logger.add_scalar(f'train/episode_length', episode_length, step)
                if self.is_zerosum:
                    # Assume teams are defined
                    t0, t1 = map(np.asarray, self.args.teams)        # [[0,2],[1,3]]
                    ret0   = float(episode_rewards[t0].sum())
                    ret1   = float(episode_rewards[t1].sum())

                    # outcome flag 0/1/2
                    if   ret0 > ret1: flag = 0          # team-0 win
                    elif ret1 > ret0: flag = 1          # team-1 win
                    else:             flag = 2          # draw

                    # -------- cumulative counters -------- #
                    self._cum_wins[flag] += 1
                    self.logger.add_scalar("train/zero_sum/cum_win_team0", self._cum_wins[0], step)
                    self.logger.add_scalar("train/zero_sum/cum_win_team1", self._cum_wins[1], step)
                    self.logger.add_scalar("train/zero_sum/cum_draw",      self._cum_wins[2], step)

                    # -------- moving-window win-rate (last 100 eps) -------- #
                    self._recent_outcomes.append(flag)
                    wins_arr = np.asarray(self._recent_outcomes)
                    w0 = (wins_arr == 0).mean()
                    w1 = (wins_arr == 1).mean()
                    d  = (wins_arr == 2).mean()

                    self.logger.add_scalar("train/zero_sum/winrate100_team0", w0, step)
                    self.logger.add_scalar("train/zero_sum/winrate100_team1", w1, step)
                    self.logger.add_scalar("train/zero_sum/drawrate100", d,  step)

                    game_outcomes_history.append(flag)

                # NOTE: Just for the debugging
                # if info["all_trunc"]:
                #     print(f"Episode truncated, {episode_length}")
                scores_history.append(episode_rewards)
                episode_length_history.append(episode_length)
                steps_history.append(step)
                episode_num += 1
                episode_length = 0
                episode_rewards = np.zeros(self.multi_agent.num_agents)

            self.insert(obs, actions, rewards, next_obs, dones)

            obs = next_obs

            # Learn
            if step % self.args.update_every == 0:
                train_info = defaultdict(lambda: defaultdict(list))
                for _ in range(self.args.gradient_steps):
                    training_info = self.multi_agent.train(self.replay_buffer)

                    # Collect metrics per agent
                    for agent_id, agent_info in training_info.items():
                        for k, v in agent_info.items():
                            train_info[agent_id][k].append(v)

                # Log average metrics per agent
                for agent_id, agent_info in train_info.items():
                    for k, v in agent_info.items():
                        self.logger.add_scalar(f'train/agent{agent_id}/{k}', np.mean(v), step)

            # Log
            if step % self.args.log_interval == 0:
                self.log_results(scores_history, episode_length_history, step, episode_num)
            
            self._update_gumbel_temperature(step)

        # Save final model
        save_path = os.path.join(self.logger.dir_name, f"final-torch.model")
        self.multi_agent.save(save_path)
        self.logger.log_model(
                file_path=save_path,
                name="final-model",
                artifact_type="model",
                metadata={"max_score": best_mean_score},
                alias="latest"
        )
        # Save score history and episode length history
        np.save(os.path.join(self.logger.dir_name, "episode_length_history.npy"), episode_length_history)
        np.save(os.path.join(self.logger.dir_name, "scores_history.npy"), scores_history)
        np.save(os.path.join(self.logger.dir_name, "steps_history.npy"), steps_history)
        if self.is_zerosum:
            np.save(os.path.join(self.logger.dir_name, "game_outcomes_history.npy"), game_outcomes_history)
        print(f"Saved final model to {save_path}")
        self.evaluator.close()

    def insert(
            self,
            raw_obs,
            raw_actions,
            raw_rewards,
            raw_next_obs,
            raw_dones):

        obs = [np.asarray(s, dtype=np.float32) for s in raw_obs]
        next_obs = [np.asarray(s, dtype=np.float32) for s in raw_next_obs]
        actions = [np.asarray(a, dtype=np.int32 if disc else np.float32)
           for a, disc in zip(raw_actions, self.is_discrete)]
        rewards = np.asarray(raw_rewards, dtype=np.float32)
        dones   = np.asarray(raw_dones,   dtype=np.uint8)

        self.replay_buffer.add(obs, actions, rewards, next_obs, dones)

    def log_results(self, scores_history, episode_length_history, step, episode_num):
        """
        Log results and return the current best mean score.

        Args:
            scores_history (list): List of scores for each episode.
            episode_length_history (list): List of episode lengths for each episode.
            step (int): Current step in the training.
            episode_num (int): Current episode number.

        Returns:
            float: Current best mean score.
        """
        window_size = min(100, len(scores_history))
        if window_size == 0:
            return -np.inf
        scores = np.asarray(scores_history[-window_size:])   # (W, N_agents)
        mean_episode_length = np.mean(episode_length_history[-window_size:])

        if self.is_zerosum:
            # Log zero-sum metrics
            team_returns = np.stack(
                [scores[:, idxs].sum(1) for idxs in self.args.teams], axis=1
            ) # (W, T)
            mean_team   = team_returns.mean(0)      # (T,)
            max_team    = team_returns.max(1).mean()
            win_rate_t0 = (team_returns[:,0] > team_returns[:,1]).mean()
          
            print(f"{step}/{self.args.max_steps}. Ep{episode_num}  "
              f"L={mean_episode_length:.1f}  maxTeam={max_team:.3f}  "
              f"teamMean={np.round(mean_team,3)}  win%={win_rate_t0:.2f}")
        else:
            mean_agent_scores = scores.mean(axis=0)                     # (N_agents,)
            mean_max_agent = scores.max(axis=1).mean()           # scalar

            print(f"{step}/{self.args.max_steps}. Ep{episode_num}"
                f"\n\tMean Ep. Length: {mean_episode_length:.2f}"
                f". Max Mean Agent-Return: {mean_max_agent:.3f}"
                f"\n\tMean Agent Returns {mean_agent_scores}")
    
    def _update_gumbel_temperature(self, step):
        """Update Gumbel-Softmax temperature with linear annealing"""
        if hasattr(self.args, 'gumbel_tau') and hasattr(self.args, 'gumbel_tau_min'):
            # Linear annealing from initial to minimum temperature
            init_temp = self.args.gumbel_tau
            min_temp = self.args.gumbel_tau_min
            anneal_steps = self.args.gumbel_anneal_steps
            
            # Calculate current temperature
            progress = min(1.0, step / anneal_steps)
            current_temp = init_temp - progress * (init_temp - min_temp)
            
            # Update temperature in the agent
            self.multi_agent.gumbel_tau = current_temp
            
            # Log the current temperature
            if step % self.args.log_interval == 0:
                self.logger.add_scalar("train/gumbel_temperature", current_temp, step)
