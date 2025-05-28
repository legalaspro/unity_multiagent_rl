
from collections import deque
import gymnasium
import numpy as np
import torch
import os

from buffers.rollout_storage import RolloutStorage
from algos import ALGO_REGISTRY
from utils.reward_normalizer import StandardNormalizer

from evals.unity_evaluator import UnityEvaluator
from evals.competitive_evaluator import CompetitiveEvaluator

class OnPolicyRunner:
    """
    Runner class for on-policy algorithms.
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

        # Initialize rollout storage
        self.rollout_storage = RolloutStorage(
            args,
            args.n_steps,
            env.observation_spaces,
            env.action_spaces,
            device=self.device
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
        
        self.reward_norm = StandardNormalizer(clip=None)

    def warmup(self):
        """
        Warmup the agent.
        """
        # Reset environment
        obs = self.env.reset(train_mode=True)
        self.rollout_storage.obs[0] = np.array(obs, dtype=np.float32)

    def train(self):
        """
        Train the agent.
        """

        self.warmup()

        self.history_data = {
            'episode_lengths': [],
            'episode_rewards': [],
            'steps': [],
            'game_outcomes_history': []
        }
        if self.is_zerosum:
            self._cum_wins = np.zeros(3, dtype=int)  # [t0_wins, t1_wins, draws]
            self._recent_outcomes = deque(maxlen=100)   # 0=t0 win, 1=t1 win, 2=draw
        self.episode_num = 0
        self.episode_length = 0
        self.episode_rewards = np.zeros(self.multi_agent.num_agents)
        self.total_steps = 0

        self.log_num = 0
        self.eval_num = 0

        best_mean_score = -np.inf

        while self.total_steps < self.args.max_steps:

            # Evaluate
            if self.total_steps // self.args.eval_interval > self.eval_num or self.total_steps == 0:
                current_best = self.evaluator.run(self.total_steps)
                if current_best > best_mean_score:
                    best_mean_score = current_best
                    save_path = os.path.join(self.logger.dir_name, f"best-torch.model")
                    self.multi_agent.save(save_path)
                    self.logger.log_model(
                        file_path=save_path,
                        name="best-model",
                        artifact_type="model",
                        metadata={"max_score": best_mean_score,
                                "step": self.total_steps},
                        alias="latest"
                    )
                    print(f"Saved best model with score {best_mean_score:.4f}")
                self.eval_num += 1

            # Collect trajectories
            self.collect_rollouts()
            self.total_steps += self.args.n_steps

            # Compute returns and advantages
            self.compute_returns()

            # Train agent
            train_info = self.multi_agent.train(self.rollout_storage)

            # # Log training information
            self.log_training(train_info, self.total_steps)
            
            # Log
            if self.total_steps // self.args.log_interval > self.log_num:
                self.log_results(self.history_data['episode_rewards'], 
                                 self.history_data['episode_lengths'], 
                                 self.total_steps, 
                                 self.episode_num)
                self.log_num += 1

            # Reset rollout storage for next rollout
            self.rollout_storage.after_update()

        # Save final model
        save_path = os.path.join(self.logger.dir_name, f"final-torch.model")
        self.multi_agent.save(save_path)
        self.logger.log_model(
                file_path=save_path,
                name="final-model",
                artifact_type="model",
                metadata={"max_mean_reward": best_mean_score},
                alias="latest"
        )
        # Save score history and episode length history
        np.save(os.path.join(self.logger.dir_name, "episode_length_history.npy"), self.history_data['episode_lengths'])
        np.save(os.path.join(self.logger.dir_name, "scores_history.npy"), self.history_data['episode_rewards'])
        np.save(os.path.join(self.logger.dir_name, "steps_history.npy"), self.history_data['steps'])
        if self.is_zerosum:
            np.save(os.path.join(self.logger.dir_name, "game_outcomes_history.npy"), self.history_data['game_outcomes_history'])
        print(f"Saved final model to {save_path}")
        self.evaluator.close()

    def collect_rollouts(self):
        """
        Collect trajectories by interacting with the environment.
        """

        for step in range(self.args.n_steps):

            with torch.no_grad():
                obs_t = torch.as_tensor(self.rollout_storage.obs[step], dtype=torch.float32, device=self.device)
                actions_t, action_log_probs_t = self.multi_agent.act(obs_t, deterministic=False)
                actions = [a.cpu().numpy() for a in actions_t]
                action_log_probs = [lp.cpu().numpy() for lp in action_log_probs_t]
                values_t = self.multi_agent.get_values(obs_t)
                values = values_t.cpu().numpy()

            next_obs, rewards, dones, truncs, info = self.env.step(actions)
            self.episode_rewards += rewards
            self.episode_length += 1

            # Handle episode termination
            if info["all_done"]:
                next_obs = self.env.reset()
                current_step = self.total_steps + step
                for i in range(self.multi_agent.num_agents):
                    self.logger.add_scalar(f'train/agent{i}_rewards', self.episode_rewards[i], current_step)
                self.logger.add_scalar(f'train/episode_length', self.episode_length, current_step)
                if self.is_zerosum:
                    # Assume teams are defined
                    t0, t1 = map(np.asarray, self.args.teams)        # [[0,2],[1,3]]
                    ret0   = float(self.episode_rewards[t0].sum())
                    ret1   = float(self.episode_rewards[t1].sum())

                    # outcome flag 0/1/2
                    if   ret0 > ret1: flag = 0          # team-0 win
                    elif ret1 > ret0: flag = 1          # team-1 win
                    else:             flag = 2          # draw

                    # -------- cumulative counters -------- #
                    self._cum_wins[flag] += 1
                    self.logger.add_scalar("train/zero_sum/cum_win_team0", self._cum_wins[0], current_step)
                    self.logger.add_scalar("train/zero_sum/cum_win_team1", self._cum_wins[1], current_step)
                    self.logger.add_scalar("train/zero_sum/cum_draw",      self._cum_wins[2], current_step)

                    # -------- moving-window win-rate (last 100 eps) -------- #
                    self._recent_outcomes.append(flag)
                    wins_arr = np.asarray(self._recent_outcomes)
                    w0 = (wins_arr == 0).mean()
                    w1 = (wins_arr == 1).mean()
                    d  = (wins_arr == 2).mean()

                    self.logger.add_scalar("train/zero_sum/winrate100_team0", w0, current_step)
                    self.logger.add_scalar("train/zero_sum/winrate100_team1", w1, current_step)
                    self.logger.add_scalar("train/zero_sum/drawrate100", d,  current_step)

                    self.history_data['game_outcomes_history'].append(flag)
                self.history_data['episode_lengths'].append(self.episode_length)
                self.history_data['episode_rewards'].append(self.episode_rewards)
                self.history_data['steps'].append(current_step)
                self.episode_num += 1
                self.episode_length = 0
                self.episode_rewards = np.zeros(self.multi_agent.num_agents)

            # Insert collected data
            data = (next_obs, actions, action_log_probs, values, rewards, dones, truncs)
            self.insert(data)

    def insert(self, data):
        """
        Insert a new transition into the buffer.

        Args:
            data (tuple): Tuple containing the data to insert
        """
        raw_obs, raw_actions, raw_action_log_probs, \
        raw_values, raw_rewards, raw_dones, raw_truncs = data

        obs = np.asarray(raw_obs, dtype=np.float32)
        actions = np.asarray(raw_actions, dtype=np.int64 if self.is_discrete[0] else np.float32)
        action_log_probs = np.asarray(raw_action_log_probs, dtype=np.float32)
        values = np.asarray(raw_values, dtype=np.float32)
        if self.args.use_reward_norm:
            rewards = np.asarray(self.reward_norm([raw_rewards]), dtype=np.float32).T 
        else:
            rewards = np.asarray([raw_rewards], dtype=np.float32).T
        masks = 1 - np.asarray([raw_dones], dtype=np.float32).T
        truncs = np.asarray([raw_truncs], dtype=np.uint8).T

        self.rollout_storage.insert(obs, actions, action_log_probs, values, rewards, masks, truncs)

    def compute_returns(self):
        """
        Compute returns and advantages for the collected trajectories.
        """
        with torch.no_grad():
            next_value = self.multi_agent.get_values(
                torch.as_tensor(self.rollout_storage.obs[-1], dtype=torch.float32, device=self.device)
            )
            next_value = next_value.cpu().numpy()
            self.rollout_storage.compute_returns_and_advantages(next_value, self.args.gamma, self.args.gae_lambda)
    
    def log_training(self, train_info, current_step):
        """
        Log training information.

        Args:
            train_info (dict): Dictionary containing training information
            current_step (int): Current step in the rollout
        """
        for agent_id, agent_info in train_info.items():
            for k, v in agent_info.items():
                self.logger.add_scalar(f'train/agent{agent_id}/{k}', v, self.total_steps)

        # Log additional metrics for debugging
        if hasattr(self.rollout_storage, 'advantages'):
            adv_mean = self.rollout_storage.advantages.mean()
            adv_std = self.rollout_storage.advantages.std()
            self.logger.add_scalar('train/advantages_mean', adv_mean, self.total_steps)
            self.logger.add_scalar('train/advantages_std', adv_std, self.total_steps)
    
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