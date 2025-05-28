import argparse
import torch
from utils.arg_tools import load_config, merge_cli
from utils.seeding import set_global_seeds

from envs.unity_env_wrapper import UnityEnvWrapper
from runners import RenderRunner

def parse_args():
    """
    CLI for rendering trained models in Unity environments.
    """
    p = argparse.ArgumentParser("Render trained RL agents in Unity environments.")
    p.add_argument("--algo", choices=["mappo", "matd3", "masac", "maddpg"],
                   default="maddpg", help="RL algorithm to use")
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to the trained model file")
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML file with experiment overrides")

    # Environment settings
    p.add_argument("--env_id", choices=["Tennis", "Soccer"],
                   default="Tennis", help="Environment name")
    p.add_argument("--worker_id", type=int, default=0,
                   help="Worker ID for Unity environment")
    p.add_argument("--seed", type=int, default=1,
                   help="Master RNG seed (numpy / torch / env)")

    # Rendering settings
    p.add_argument("--render_episodes", type=int, default=3,
                   help="Number of episodes to render")

    # Device settings
    p.add_argument("--cuda", action='store_false', default=True,
                   help="Use GPU if available")
    p.add_argument("--cuda_deterministic", action="store_false", default=True,
                   help="Turn off CuDNN autotune for exact reproducibility")

    cli, unknown_cli = p.parse_known_args()
    cfg = load_config(cli.algo, cli.config)  # load from YAML
    args = merge_cli(cfg, cli, unknown_cli)  # override from CLI

    return args


def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    set_global_seeds(args.seed, args.cuda_deterministic)
    
    # Initialize environment with rendering enabled
    env = UnityEnvWrapper(args.env_id, worker_id=args.worker_id, seed=args.seed, no_graphics=False)
    
    print(f'Environment: {args.env_id}')
    print(f'Number of agents: {env.n_agents}')
    print(f'Brain names: {env.brain_names}')
    print(f'Action spaces: {env.action_spaces}')
    print(f'Observation spaces: {env.observation_spaces}')
    
    # Set device
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    try:
        # Initialize render runner
        render_runner = RenderRunner(args, env, device)
        
        # Start rendering
        render_runner.render()
        
    except Exception as e:
        print(f"Error during rendering: {e}")
        raise
    finally:
        # Close environment and logger
        env.close()


if __name__ == "__main__":
    main()
