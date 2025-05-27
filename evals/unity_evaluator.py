import torch, numpy as np
from envs.unity_env_wrapper import UnityEnvWrapper
# from utils.profiler import Profiler

class UnityEvaluator:
    """
    Evaluator for Unity environments.
    Runs deterministic roll-outs in a separate Unity environment.
    Never feeds data back to training.
    """
    def __init__(self, logger, cfg, make_agent_snapshot):
        """
        logger      :  same logger as training
        cfg         :  args Namespace (needs eval_interval, n_eval_rollout_threads, render)
        make_agent_snapshot:  callable returning a *frozen* MultiAgent
                              (no gradients, on CPU or spare GPU)
        """
        self.logger   = logger
        self.cfg      = cfg
        self.snapshot_fn = make_agent_snapshot

        self.env = UnityEnvWrapper(cfg.env_id,
                                   worker_id=cfg.worker_id + 10,
                                   seed=cfg.seed + 1000)

        self.team_indices  = [[idx] if isinstance(idx, int) else list(idx)
                              for idx in getattr(cfg, "teams", [])]

        # NOTE: Analysis shown that env_step is the bottleneck when train_mode=False,
        # (train_mode=True)
        # [eval@10000] snapshot:   9.7 ms  env_reset:  26.7 ms
        # obs→tensor:   3.4 ms  actor_forward:  38.8 ms  env_step: 665.2 ms  total_eval: 745.5 ms
        # VS (train_mode=False)
        # eval@10000] snapshot:   9.6 ms  env_reset: 180.0 ms
        # obs→tensor:  30.0 ms  actor_forward: 176.4 ms  env_step:27017.1 ms  total_eval:27423.7 ms
        # profiling code
        # with self.prof.track("total_eval"):
        #   with self.prof.track("env_reset"):
        #     obs = self.env.reset(train_mode=False)
        #   with self.prof.track("env_step"):
        #     obs, rewards, dones, truncs, info = self.env.step(actions)
        # self.prof.report()

    @torch.no_grad()
    def run(self, global_step: int):
        multi_agent = self.snapshot_fn()

        n_teams   = len(self.team_indices)

        all_ep_agent_returns = []
        ep_lengths = []
        eval_episodes = self.cfg.eval_episodes
        for ep in range(eval_episodes):
            obs = self.env.reset(train_mode=True)
            done  = False
            ep_rewards = np.zeros(multi_agent.num_agents)
            steps = 0

            while not done:
                # deterministic actions – no noise, no sampling stochasticity
                obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32,
                                device=multi_agent.device)
                actions_t = multi_agent.act(obs_t, deterministic=True)
                actions = [a.cpu().numpy() for a in actions_t]
                obs, rewards, dones, truncs, info = self.env.step(actions)
                ep_rewards += rewards
                steps += 1
                done = info["all_done"]

            all_ep_agent_returns.append(ep_rewards)
            ep_lengths.append(steps)

        all_ep_agent_returns = np.asarray(all_ep_agent_returns)   # (E, N)
        mean_ep_length = float(np.mean(ep_lengths))

        # Metrics
        mean_agent_return    = all_ep_agent_returns.mean(axis=0)  # (N,)
        max_agent_per_ep     = all_ep_agent_returns.max(axis=1)   # (E,)
        mean_max_agent_ret   = float(max_agent_per_ep.mean())

        if n_teams:
            # shape (E, T)
            team_returns = np.stack([
                all_ep_agent_returns[:, idxs].sum(axis=1)
                for idxs in self.team_indices        # preserve order
            ], axis=1)

            mean_team_return  = team_returns.mean(axis=0)         # (T,)
            max_team_per_ep   = team_returns.max(axis=1)
            mean_max_team_ret = float(max_team_per_ep.mean())

        # Logging
        for i, r in enumerate(mean_agent_return):
            self.logger.add_scalar(f"eval/agent{i}_mean_return", r, global_step)
        self.logger.add_scalar("eval/agent_mean_max_return",
                               mean_max_agent_ret, global_step)

        if n_teams:
            for t, r in enumerate(mean_team_return):
                self.logger.add_scalar(f"eval/team{t}_mean_return", r, global_step)
            self.logger.add_scalar("eval/team_mean_max_return",
                                   mean_max_team_ret, global_step)
        self.logger.add_scalar("eval/mean_episode_length",
                               mean_ep_length, global_step)

        # Console output
        if n_teams:
            print(f"[eval] step={global_step:,}  len={mean_ep_length:.1f}  "
                  f"agent_max={mean_max_agent_ret:.3f}  "
                  f"team_max={mean_max_team_ret:.3f}\n"
                  f"        agent_mean={np.round(mean_agent_return,4)}\n"
                  f"        team_mean ={np.round(mean_team_return,4)}")
        else:
            print(f"[eval] step={global_step:,}  len={mean_ep_length:.1f}  "
                  f"mean_max_return={mean_max_agent_ret:.3f}  "
                  f"mean_return={np.round(mean_agent_return,4)}")
        
        return mean_max_agent_ret
    
    def close(self):
        self.env.close()
        print("Closed evaluation environment.")