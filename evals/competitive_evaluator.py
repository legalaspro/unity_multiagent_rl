import torch
import numpy as np
import os
import pickle
from typing import List, Tuple, Any

from envs.unity_env_wrapper import UnityEnvWrapper
from evals.elo_rating import EloRatingSystem
from algo.random import RandomPolicy

class CompetitiveEvaluator:
    """
    Competitive evaluator for zero-sum multi-agent environments.

    RandomPolicy is used as the baseline in the beginning.
    
    Implements Elo rating system and maintains a pool of model snapshots
    for competitive evaluation and skill progression tracking.
    """
    
    def __init__(self, logger, cfg, make_agent_snapshot):
        """
        Initialize competitive evaluator.
        
        Args:
            logger: Training logger
            cfg: Configuration object
            make_agent_snapshot: Function to create agent snapshots
        """
        self.logger = logger
        self.cfg = cfg
        self.snapshot_fn = make_agent_snapshot
        self.teams = cfg.teams
        
        # Create separate evaluation environment
        self.env = UnityEnvWrapper(
            cfg.env_id,
            worker_id=cfg.worker_id + 10,
            seed=cfg.seed + 1000
        )
        
        # Elo rating system
        self.elo_system = EloRatingSystem(k_factor=32, initial_rating=1200)
        
        # Model snapshot management
        self.model_snapshots = {}  # {step: model_snapshot}
        self.snapshot_ratings = {}  # {step: elo_rating}
        self.max_snapshots = getattr(cfg, 'max_model_snapshots', 10)
        
        # Random baseline policy for initial evaluation
        self.random_episodes = 16
        self.random_policy = RandomPolicy(self.env.action_spaces)
        
        # Competitive evaluation settings
        self.eval_episodes_per_matchup = getattr(cfg, 'competitive_eval_episodes', 5)
        
        print(f"Competitive evaluator initialized with teams: {self.teams}")
        print(f"Episodes per matchup: {self.eval_episodes_per_matchup}")
    
    @torch.no_grad()
    def run(self, global_step: int) -> float:
        """
        Run competitive evaluation at the given training step.
        
        Args:
            global_step: Current training step
        """
        current_agent = self.snapshot_fn()
        
        # Run vs random baseline
        self._evaluate_vs_random(current_agent, global_step)
        
        # Run competitive matches
        if len(self.model_snapshots) > 0:
            self._run_competitive_matches(current_agent, global_step)
        
        # Save snapshot
        self._save_model_snapshot(current_agent, global_step)

        return self.snapshot_ratings[global_step]

    @torch.no_grad()
    def play_one_episode(self, agentA, agentB, *, teams=None, env=None) -> int:
        """
        Run a single match between assigned agents.
        
        Args:
            agentA: First agent
            agentB: Second agent
            teams: List of team configurations [[team0_agents], [team1_agents]]
            env: Environment to use (if None, use internal env)
            
        Returns:
            Score of 1/½/0 from team-A perspective
        """
        env   = env   or self.env
        teams = teams or self.teams
        # A is always index 0 team; 
        team0, team1 = map(np.asarray, teams)  # [[0,2], [1,3]]
        n_agents = len(team0) + len(team1)

        obs = env.reset(train_mode=True)
        done = False
        ep_ret = np.zeros(n_agents)
        
        while not done:
            obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32,
                                    device=agentA.device)
            actions_A = agentA.act(obs_t, deterministic=True)
            actions_B = agentB.act(obs_t, deterministic=True)

            aA = np.vstack([a.numpy() for a in actions_A])   # (N, act_dim)
            aB = np.vstack([b.numpy() for b in actions_B])

            # interleave so that correct agents get correct policy
            joint_actions = np.empty_like(aA) # (N, act_dim)
            joint_actions[team0] = aA[team0]
            joint_actions[team1] = aB[team1]
            # actions = [None] * len(obs)
            # for i in team0: actions[i] = actions_A[i].cpu().numpy()
            # for i in team1: actions[i] = actions_B[i].cpu().numpy()

            obs, rew, _, _, info = env.step(joint_actions)
            ep_ret += rew
            done = info["all_done"]

        score_A = ep_ret[team0].sum()
        score_B = ep_ret[team1].sum()
        return 1.0 if score_A > score_B else 0.5 if score_A == score_B else 0.0
    
    def _evaluate_vs_random(self, agent, global_step: int) -> None:
        """Evaluate current agent against random baseline."""
        wins = losses = draws = 0
        
        for episode in range(self.random_episodes):
            # Randomly assign agent to team
            if np.random.randint(2) == 0:
                teams = [self.teams[0], self.teams[1]]
            else:
                teams = [self.teams[1], self.teams[0]]

            score = self.play_one_episode(agent, self.random_policy, teams=teams)
            
            if score == 1.0 :
                wins += 1
            elif score == 0.0:
                losses += 1
            else:
                draws += 1
        
        total_games = wins + losses + draws
        win_rate = wins / total_games if total_games > 0 else 0
        
        # Log results
        self.logger.add_scalar("competitive/win_rate_vs_random", win_rate, global_step)
        
        print(f"[competitive] step={global_step:,} vs random: "
              f"win_rate={win_rate:.3f} ({wins}W-{losses}L-{draws}D)")
        
    def _run_competitive_matches(self, current_agent, global_step: int) -> None:
        """Run competitive matches against previous model snapshots."""
        # Set current agent's rating
        current_rating = self._get_best_rating()
        
        # Select opponents 
        opponents = self._select_opponents()
        
        total_wins = 0
        total_draws = 0
        total_losses = 0
        
        for opponent_step, opponent_agent in opponents:
            opponent_rating = self.snapshot_ratings[opponent_step]
            wins = draws = losses = 0
        

            for episode in range(self.eval_episodes_per_matchup):
                # Randomly assign agent to team
                if np.random.randint(2) == 0:
                    teams = [self.teams[0], self.teams[1]]
                else:
                    teams = [self.teams[1], self.teams[0]]

                score = self.play_one_episode(current_agent, opponent_agent, teams=teams)
                
                if score == 1.0 :
                    wins += 1
                elif score == 0.0:
                    losses += 1
                else:
                    draws += 1
   
                #Update ELO Rating
                current_rating, opponent_rating = self.elo_system.update_ratings(
                    current_rating, opponent_rating, score
                )

            # Update stored ratings
            self.snapshot_ratings[opponent_step] = opponent_rating

            # Log match results
            self.logger.add_scalar(f"competitive/vs_step_{opponent_step}/wins", wins, global_step)
            self.logger.add_scalar(f"competitive/vs_step_{opponent_step}/win_rate", 
                                 wins / self.eval_episodes_per_matchup, 
                                 global_step)
            print(f"[competitive] step={global_step:,} vs step={opponent_step:,}: "
                  f"win_rate={wins / self.eval_episodes_per_matchup:.3f} ({wins}W-{losses}L-{draws}D)")
            
            total_wins += wins
            total_draws += draws
            total_losses += losses

        # Update current agent's rating
        self.snapshot_ratings[global_step] = current_rating
        win_rate = total_wins / (len(opponents) * self.eval_episodes_per_matchup)

        # Log match results
        self.logger.add_scalar("competitive/elo_rating", current_rating, global_step)
        self.logger.add_scalar("competitive/overall_win_rate", win_rate, global_step)
        self.logger.add_scalar("competitive/total_opponents", len(opponents), global_step)

        print(f"[competitive] step={global_step:,} vs {len(opponents)} opponents: "
              f"win_rate={win_rate:.3f} (rating: {current_rating:.0f}) ({total_wins}W-{total_losses}L-{total_draws}D)")

    def _save_model_snapshot(self, agent_snapshot, global_step: int,) -> None:
        """Save a model snapshot for future competitive evaluation."""
        # Remove oldest snapshot if we're at capacity
        if len(self.model_snapshots) >= self.max_snapshots:
            oldest_step = min(self.model_snapshots.keys())
            del self.model_snapshots[oldest_step]
            if oldest_step in self.snapshot_ratings:
                del self.snapshot_ratings[oldest_step]
        
        # Save new snapshot
        self.model_snapshots[global_step] = agent_snapshot
        
        # Initialize with current best rating or default
        if global_step not in self.snapshot_ratings:
            self.snapshot_ratings[global_step] = self.elo_system.initial_rating
        
        print(f"Stored model snapshot at step {global_step} (rating: {self.snapshot_ratings[global_step]:.0f})")
    
    def _get_best_rating(self) -> float:
        """Get the best rating from the snapshot ratings."""
        if not self.snapshot_ratings:
            return self.elo_system.initial_rating
        return max(self.snapshot_ratings.values())
    
    def _select_opponents(self) -> List[Tuple[int, Any]]:
        """Select opponent snapshots for competitive evaluation."""
        if len(self.model_snapshots) <= 4:
            # Use all available snapshots if we have few
            return list(self.model_snapshots.items())
        
        # Strategic selection: recent snapshots + best performing
        opponents = []
        
        # Add most recent snapshots
        recent_steps = sorted(self.model_snapshots.keys())[-3:]
        for step in recent_steps:
            opponents.append((step, self.model_snapshots[step]))
        
        # Add highest rated snapshot
        if self.snapshot_ratings:
            best_step = max(self.snapshot_ratings.keys(), 
                          key=lambda x: self.snapshot_ratings[x])
            if best_step not in recent_steps:
                opponents.append((best_step, self.model_snapshots[best_step]))
        
        return opponents
    
    def close(self) -> None:
        """Clean up resources."""
        self.env.close()
        print("Closed competitive evaluation environment.")
    
    def save_state(self, save_dir: str) -> None:
        """Save competitive evaluator state."""
        state = {
            'snapshot_ratings': self.snapshot_ratings,
            'elo_system_state': self.elo_system.get_state()
        }
        
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'competitive_state.pkl'), 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, save_dir: str) -> None:
        """Load competitive evaluator state."""
        state_file = os.path.join(save_dir, 'competitive_state.pkl')
        if os.path.exists(state_file):
            with open(state_file, 'rb') as f:
                state = pickle.load(f)
            
            self.snapshot_ratings = state.get('snapshot_ratings', {})
            if 'elo_system_state' in state:
                self.elo_system.load_state(state['elo_system_state'])
