import torch
import numpy as np
import os

from algos import ALGO_REGISTRY

class RenderRunner:
    """
    Runner class for rendering trained models.
    """
    
    def __init__(self, args, env, device):
        """
        Initialize the render runner.

        Args:
            args: Arguments containing rendering parameters
            env: Environment to render on
            device: Torch device to use for computations
        """
        self.args = args
        self.device = device
        self.env = env
        
        # Validate model path
        if not hasattr(args, 'model_path') or not args.model_path:
            raise ValueError("model_path must be specified for rendering")
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        # Initialize the multi-agent algorithm
        self.multi_agent = ALGO_REGISTRY[args.algo](
            args,
            env.observation_spaces,
            env.action_spaces,
            device
        )
        
        # Load the trained model
        print(f"Loading model from: {args.model_path}")
        self.multi_agent.load(args.model_path)
        self.multi_agent.set_eval_mode()  # Set to evaluation mode
    
    @torch.no_grad()
    def render(self):
        """
        Render episodes using the trained model.
        """
        # Get rendering parameters with defaults
        num_episodes = self.args.render_episodes
        
        episode_rewards = []
        episode_lengths = []
        
        print(f"\nRendering {num_episodes} episodes...")
        print(f"Environment: {self.args.env_id}")
        print(f"Algorithm: {self.args.algo}")
        print("-" * 50)
        
        for episode in range(num_episodes):
            obs = self.env.reset(train_mode=False)
            done  = False
            ep_rewards = np.zeros(self.multi_agent.num_agents)
            steps = 0
            
            print(f"Episode {episode + 1}/{num_episodes}")
            
            while not done:
                # Convert observations to tensors
                obs_t = torch.as_tensor(np.stack(obs), dtype=torch.float32,
                                device=self.multi_agent.device)
                actions_t = self.multi_agent.act(obs_t, deterministic=True)
                actions = [a.cpu().numpy() for a in actions_t]
                obs, rewards, dones, truncs, info = self.env.step(actions)
                ep_rewards += rewards
                steps += 1
                done = info["all_done"]
          
            
            episode_rewards.append(ep_rewards)
            episode_lengths.append(steps)
            
            # Print episode results
            mean_reward = np.mean(ep_rewards)
            max_reward = np.max(ep_rewards)
            print(f"  Length: {steps:4d} | Mean Reward: {mean_reward:8.3f} | Max Reward: {max_reward:8.3f}")
            print(f"  Agent Rewards: {ep_rewards}")
            print()
        
        # Print summary statistics
        self._print_summary_statistics(episode_rewards, episode_lengths, num_episodes)
    
    def _print_summary_statistics(self, episode_rewards, episode_lengths, num_episodes):
        """
        Print summary statistics for the rendered episodes.
        
        Args:
            episode_rewards: List of episode rewards
            episode_lengths: List of episode lengths
            num_episodes: Number of episodes rendered
        """
        print("=" * 50)
        print("SUMMARY STATISTICS")
        print("=" * 50)
        
        episode_rewards = np.array(episode_rewards)
        episode_lengths = np.array(episode_lengths)
        
        mean_episode_reward = np.mean(episode_rewards, axis=0)
        std_episode_reward = np.std(episode_rewards, axis=0)
        mean_episode_length = np.mean(episode_lengths)
        std_episode_length = np.std(episode_lengths)
        
        print(f"Episodes: {num_episodes}")
        print(f"Mean Episode Length: {mean_episode_length:.2f} ± {std_episode_length:.2f}")
        print(f"Mean Agent Rewards: {mean_episode_reward}")
        print(f"Std Agent Rewards:  {std_episode_reward}")
        print(f"Overall Mean Reward: {np.mean(mean_episode_reward):.3f}")
        print(f"Overall Max Reward:  {np.max(mean_episode_reward):.3f}")
