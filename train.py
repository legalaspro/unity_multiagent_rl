import argparse
import time
import torch
from utils.arg_tools import load_config, merge_cli
from utils.seeding import set_global_seeds

from envs.unity_env_wrapper import UnityEnvWrapper
from utils.logger import Logger
from runners import RUNNER_REGISTRY


def parse_args():
    """
    Global CLI for every MARL algorithm.
    Additional flags are parsed from YAML and can be overridden via CLI.
    """
    p = argparse.ArgumentParser("Run RL algorithms on Unity environments.")
    p.add_argument("--algo", choices=["mappo", "matd3", "masac", "maddpg"],
                   default="maddpg", help="RL algorithm to use")
    p.add_argument("--run_name", type=str, default="test_run",
                   help="Optional run name for logging")
    p.add_argument("--config", type=str, default=None,
                   help="Optional YAML file with experiment overrides")

    # env
    p.add_argument("--env_id", choices=["Tennis", "Soccer"],
                   default="Tennis", help="Environment name")
    p.add_argument("--worker_id", type=int, default=0,
                   help="Worker ID for Unity environment")
    p.add_argument("--seed", type=int, default=1,
                   help="Master RNG seed (numpy / torch / env)")
    p.add_argument("--max_steps", type=int, default=200000,
                   help="Total environment steps")


    # device, perfomance
    p.add_argument("--cuda", action='store_false', default=True,
                   help="by default True, will use GPU to train; or else will use CPU;")
    p.add_argument("--torch_threads", type=int, default=1,
                   help="Limit Torch + MKL/OMP CPU threads")
    p.add_argument("--cuda_deterministic", action="store_false", default=True,
                   help="Turn off CuDNN autotune for exact reproducibility")

    # logging
    p.add_argument("--log_interval", type=int, default=1000,
                   help="Env steps between logging")
    p.add_argument("--use_wandb", action="store_true", default=False,
                   help="Use Weights & Biases for logging")
    p.add_argument("--eval_interval", type=int, default=10000,
                   help="Env steps between evaluation episodes")
    p.add_argument("--eval_episodes", type=int, default=10,
                   help="Number of evaluation episodes")
    p.add_argument("--render", action="store_true",
                   help="Render env in a window during evaluation")

    cli, unknown_cli =  p.parse_known_args()
    cfg = load_config(cli.algo, cli.config) # load from YAML
    args = merge_cli(cfg, cli, unknown_cli) # override from CLI
    print(args)

    return args

def main():
    args = parse_args()

     # Set thread configuration
    print(args)
    # cpu_threads = args.torch_threads or 1
    # torch.set_num_threads(cpu_threads)

    set_global_seeds(args.seed, args.cuda_deterministic)

    # Initialize environment
    env = UnityEnvWrapper(args.env_id, worker_id=args.worker_id, seed=args.seed)

    print(f'Number of agents: {env.n_agents}')
    print(f'Brain names: {env.brain_names}')
    print(f'Action spaces: {env.action_spaces}')
    print(f'Observation spaces: {env.observation_spaces}')

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(f'Number of CUDA devices: {torch.cuda.device_count()}')

    run_name = (
        f"{args.run_name}_"
        f"lr{args.actor_lr}_"
        f"gamma{args.gamma}_"
        f"{int(time.time())}"
    )
    hyperparams = vars(args)
    logger = Logger(
        run_name=run_name,
        env=args.env_id,
        algo=args.algo,
        use_wandb=args.use_wandb,
        config=hyperparams)

    # Initialize runner
    runner = RUNNER_REGISTRY[args.algo](args, env, logger, device)

    try:
        runner.train()
    finally:
        # Close environment
        env.close()
        logger.close()

if __name__ == "__main__":
    main()
