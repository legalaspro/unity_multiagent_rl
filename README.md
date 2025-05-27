# Multi-Agent Reinforcement Learning

A collection of multi-agent reinforcement learning algorithms implemented for Unity environments.

## Algorithms

- **MAPPO** (Multi-Agent Proximal Policy Optimization)
  - All shared variant
  - Critic shared variant
- **MATD3** (Multi-Agent Twin Delayed Deep Deterministic Policy Gradient)
- **MASAC** (Multi-Agent Soft Actor-Critic)
- **MADDPG** (Multi-Agent Deep Deterministic Policy Gradient)

## Environments

- **Tennis**: Two-agent collaborative environment
- **Soccer**: Multi-agent competitive environment

## Quick Start

### Training

```bash
python train.py --env_id Tennis --algo masac
```

### Visualization

```bash
# Algorithm comparison plots
python render_results.py --env_id Tennis

# Individual algorithm analysis
python render_results.py --env_id Tennis --single_algo masac

# All individual plots
python render_results.py --env_id Tennis --individual
```

## Project Structure

```
├── algo/                    # Algorithm implementations
│   ├── agent/              # Individual agent implementations
│   ├── maddpg.py          # Multi-Agent DDPG
│   ├── mappo.py           # Multi-Agent PPO
│   ├── masac.py           # Multi-Agent SAC
│   └── matd3.py           # Multi-Agent TD3
├── networks/               # Neural network architectures
│   ├── actors/            # Policy networks
│   ├── critics/           # Value networks
│   └── modules/           # Shared network components
├── envs/                   # Environment wrappers
│   └── unity_env_wrapper.py
├── buffers/                # Experience replay and storage
│   ├── replay_buffer.py   # Off-policy buffer
│   └── rollout_storage.py # On-policy buffer
├── runners/                # Training loop implementations
│   ├── on_policy_runner.py
│   └── off_policy_runner.py
├── evals/                  # Evaluation and metrics
│   ├── elo_rating.py      # Elo rating system
│   └── competitive_evaluator.py
├── configs/                # Configuration files
│   ├── algos/             # Algorithm-specific configs
│   └── env_tuned/         # Environment-tuned configs
├── app/                    # Unity environment executables
│   ├── Tennis.app
│   └── Soccer.app
├── results/                # Training results and logs
├── figures/                # Generated plots and visualizations
├── utils/                  # Utility functions
├── python/                 # Unity ML-Agents Python API
├── train.py               # Main training script
└── render_results.py      # Visualization and analysis script
```

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create environment from the provided environment.yaml
conda env create -f environment.yaml

# Activate the environment
conda activate control-reacher
```

### Option 2: Using Pip

```bash
pip install -r requirements.txt

# Install the local Unity environment package:
pip install -e ./python
```

## Requirements

- Python 3.11+
- PyTorch
- Unity ML-Agents
- NumPy
- Matplotlib
- Pandas

## Results

Training results are automatically saved to `./results/` and visualizations to `./figures/`.

---

_More detailed documentation coming soon._
