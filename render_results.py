#!/usr/bin/env python3
"""
Simplified script to render training results from different algorithms.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def load_algorithm_data(results_dir, env_id, algo_name):
    """Load training data for a single algorithm."""
    algo_dir = os.path.join(results_dir, env_id, algo_name)

    if not os.path.exists(algo_dir):
        print(f"Directory not found: {algo_dir}")
        return None

    # Load numpy files
    files = {
        "steps": os.path.join(algo_dir, "steps_history.npy"),
        "rewards": os.path.join(algo_dir, "scores_history.npy"),
        "episode_length": os.path.join(algo_dir, "episode_length_history.npy")
    }

    # Check if all files exist
    for file_path in files.values():
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            return None

    try:
        steps = np.load(files["steps"])
        rewards = np.load(files["rewards"])
        episode_lengths = np.load(files["episode_length"])

        # If rewards is 2D (multiple agents), average across agents
        if len(rewards.shape) > 1:
            rewards = np.mean(rewards, axis=1)

        return {
            "steps": steps,
            "rewards": rewards,
            "episode_length": episode_lengths
        }
    except Exception as e:
        print(f"Error loading data for {algo_name}: {e}")
        return None

def plot_metric(data, env_id, metric, algo_titles, save_dir="figures"):
    """Create a comparison plot for a specific metric."""
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, (algo_name, algo_data) in enumerate(data.items()):
        if algo_data is None or metric not in algo_data:
            continue

        steps = algo_data["steps"]
        values = algo_data[metric]

        # Calculate 100-episode rolling average
        window_size = min(100, len(values))
        rolling_values = pd.Series(values).rolling(window=window_size, min_periods=1).mean().values

        color = colors[i % len(colors)]

        # Get display name from algo_titles or use original name
        display_name = algo_titles.get(algo_name, algo_name)

        # Plot raw values as light dashed line
        plt.plot(steps, values, '--', alpha=0.3, linewidth=1.1, color=color)

        # Plot rolling average as thick line
        plt.plot(steps, rolling_values, linewidth=3, color=color, label=display_name)

    # Styling
    metric_title = "Rewards" if metric == "rewards" else "Episode Length"
    plt.title(f"{env_id} Algorithm Comparison - {metric_title}\n(100-episode moving average)",
             fontsize=14, fontweight='bold')
    plt.xlabel("Environment Steps", fontsize=12)
    plt.ylabel(metric_title, fontsize=12)

    # Position legend
    legend_loc = 'lower right' if metric == 'rewards' else 'upper left'
    plt.legend(loc=legend_loc, fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    save_path = os.path.join(save_dir, f"{env_id}_{metric}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {metric_title} figure to {save_path}")
    plt.close()


def plot_individual_algorithm(algo_name, algo_data, env_id, algo_titles, save_dir="figures"):
    """Create individual plots for a specific algorithm."""
    if algo_data is None:
        print(f"No data available for {algo_name}")
        return

    os.makedirs(save_dir, exist_ok=True)
    display_name = algo_titles.get(algo_name, algo_name)

    # Colors for individual plots
    raw_color = '#A8D0E6'  # Light blue
    rolling_color = '#2E86C1'  # Medium blue

    for metric in ["rewards", "episode_length"]:
        if metric not in algo_data:
            continue

        steps = algo_data["steps"]
        values = algo_data[metric]

        # Calculate 100-episode rolling average
        window_size = min(100, len(values))
        rolling_values = pd.Series(values).rolling(window=window_size, min_periods=1).mean().values

        plt.figure(figsize=(12, 8))

        # Plot raw values as thin dashed line
        plt.plot(steps, values, '--', alpha=0.7, linewidth=1.5, color=raw_color, label='Episode Values')

        # Plot rolling average as thick line
        plt.plot(steps, rolling_values, linewidth=3, color=rolling_color, label='100-Episode Moving Average')

        # Styling
        metric_title = "Rewards" if metric == "rewards" else "Episode Length"
        plt.title(f"{env_id} - {display_name} Training {metric_title}\n(100-episode moving average)",
                 fontsize=14, fontweight='bold')
        plt.xlabel("Environment Steps", fontsize=12)
        plt.ylabel(metric_title, fontsize=12)

        # Position legend
        legend_loc = 'lower right' if metric == 'rewards' else 'upper left'
        plt.legend(loc=legend_loc, fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        safe_algo_name = algo_name.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = os.path.join(save_dir, f"{env_id}_{safe_algo_name}_{metric}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {display_name} {metric_title} figure to {save_path}")
        plt.close()


def print_summary(data, env_id, algo_titles):
    """Print summary statistics."""
    print(f"\n=== {env_id} Training Results Summary ===")

    for metric in ["rewards", "episode_length"]:
        print(f"\n{metric.upper()}:")
        print("-" * 60)

        for algo_name, algo_data in data.items():
            if algo_data is None or metric not in algo_data:
                continue

            values = algo_data[metric]

            # Final values (last 10% of training)
            final_portion = max(1, len(values) // 10)
            final_mean = np.mean(values[-final_portion:])
            final_std = np.std(values[-final_portion:])

            # Overall statistics
            overall_mean = np.mean(values)
            overall_min = np.min(values)
            overall_max = np.max(values)

            # Get display name from algo_titles or use original name
            display_name = algo_titles.get(algo_name, algo_name)

            print(f"{display_name:25s} | "
                  f"Final: {final_mean:8.3f}±{final_std:6.3f} | "
                  f"Overall: {overall_mean:8.3f} | "
                  f"Range: [{overall_min:8.3f}, {overall_max:8.3f}]")


def main():
    # Configuration - edit these as needed
    env_id = "Tennis"  # or "Soccer"
    algorithms = ["mappo_all_shared", "mappo_critic_shared", "matd3", "masac", "maddpg"]

    # Render mode: "comparison", "individual", or specific algorithm name like "masac"
    render_mode = "comparison"  # Change to "individual" or specific algorithm name

    # Custom display names for algorithms (optional - edit these for nicer plot labels)
    algo_titles = {
        "mappo_all_shared": "MAPPO (all shared)",
        "mappo_critic_shared": "MAPPO (critic shared)",
        "matd3": "MATD3",
        "masac": "MASAC",
        "maddpg": "MADDPG"
    }

    results_dir = "results"
    save_dir = "figures"

    print(f"Loading data for {env_id} environment...")

    # Load data for all algorithms
    data = {}
    for algo in algorithms:
        print(f"Loading {algo}...")
        data[algo] = load_algorithm_data(results_dir, env_id, algo)

    # Filter out algorithms with no data
    data = {k: v for k, v in data.items() if v is not None}

    if not data:
        print("No data loaded. Check your paths and algorithm names.")
        return

    print(f"Successfully loaded data for: {list(data.keys())}")

    # Print summary statistics
    print_summary(data, env_id, algo_titles)

    # Create plots based on render mode
    print(f"\nCreating plots...")

    if render_mode == "comparison":
        # Create comparison plots
        plot_metric(data, env_id, "rewards", algo_titles, save_dir)
        plot_metric(data, env_id, "episode_length", algo_titles, save_dir)
        print(f"Created comparison plots in {save_dir}/")

    elif render_mode == "individual":
        # Create individual plots for all algorithms
        for algo_name, algo_data in data.items():
            plot_individual_algorithm(algo_name, algo_data, env_id, algo_titles, save_dir)
        print(f"Created individual plots for all algorithms in {save_dir}/")

    elif render_mode in data:
        # Create individual plots for specific algorithm
        plot_individual_algorithm(render_mode, data[render_mode], env_id, algo_titles, save_dir)
        display_name = algo_titles.get(render_mode, render_mode)
        print(f"Created individual plots for {display_name} in {save_dir}/")

    else:
        print(f"Invalid render_mode: '{render_mode}'")
        print(f"Valid options: 'comparison', 'individual', or one of: {list(data.keys())}")
        return

    print(f"\nDone!")


if __name__ == "__main__":
    main()