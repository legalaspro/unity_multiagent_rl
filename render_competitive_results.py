#!/usr/bin/env python3
"""
Simple script to load and visualize Soccer competitive evaluation metrics from TensorBoard event files.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Error: TensorBoard is required. Install with: pip install tensorboard")
    exit(1)


def load_soccer_competitive_data():
    """Load competitive data from Soccer algorithm tfevents files."""
    algorithms = {
        "MAPPO (Shared Critic)": "results/Soccer/mappo_shared_critic/events.out.tfevents.1748427104.DM23.local.1978.0",
        "MASAC (Shared Critic)": "results/Soccer/masac_shared_critic/events.out.tfevents.1748430831.DM23.local.79682.0"
    }

    metrics = ["competitive/win_rate_vs_random", "competitive/elo_rating", "competitive/overall_win_rate"]
    all_data = {}

    for algo_name, tfevents_file in algorithms.items():
        if not os.path.exists(tfevents_file):
            print(f"File not found: {tfevents_file}")
            continue

        print(f"Loading {algo_name} data from: {tfevents_file}")

        # Initialize EventAccumulator
        ea = EventAccumulator(tfevents_file)
        ea.Reload()

        # Load the three competitive metrics
        algo_data = {}
        for metric in metrics:
            scalar_events = ea.Scalars(metric)
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]

            algo_data[metric] = {
                "steps": np.array(steps),
                "values": np.array(values)
            }
            print(f"  Loaded {len(steps)} data points for {metric}")

        all_data[algo_name] = algo_data

    return all_data


def plot_competitive_metrics(data):
    """Create plots for the three competitive metrics."""
    os.makedirs("figures", exist_ok=True)

    metric_titles = {
        "competitive/win_rate_vs_random": "Win Rate vs Random",
        "competitive/elo_rating": "Elo Rating",
        "competitive/overall_win_rate": "Overall Win Rate"
    }

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Get all metrics from first algorithm
    first_algo = list(data.keys())[0]
    metrics = list(data[first_algo].keys())

    for metric in metrics:
        plt.figure(figsize=(12, 8))

        for i, (algo_name, algo_data) in enumerate(data.items()):
            if metric not in algo_data:
                continue

            steps = algo_data[metric]["steps"]
            values = algo_data[metric]["values"]

            # Calculate 100-step rolling average
            window_size = min(100, len(values))
            rolling_values = pd.Series(values).rolling(window=window_size, min_periods=1).mean().values

            color = colors[i % len(colors)]

            # Plot raw values as light dashed line
            plt.plot(steps, values, '--', alpha=0.3, linewidth=1.1, color=color)

            # Plot rolling average as thick line
            plt.plot(steps, rolling_values, linewidth=3, color=color, label=algo_name)

        # Styling
        metric_title = metric_titles[metric]
        plt.title(f"Soccer Competitive Evaluation - {metric_title}\n(100-step moving average)",
                 fontsize=14, fontweight='bold')
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel(metric_title, fontsize=12)

        # Set y-axis limits for win rates
        if "win_rate" in metric:
            plt.ylim(0, 1)

        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Save figure
        metric_name = metric.split("/")[-1]
        save_path = f"figures/Soccer_competitive_{metric_name}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric_title} figure to {save_path}")
        plt.close()


def print_summary(data):
    """Print summary statistics."""
    print("\n=== Soccer Competitive Evaluation Summary ===")

    # Get all metrics from first algorithm
    first_algo = list(data.keys())[0]
    metrics = list(data[first_algo].keys())

    for metric in metrics:
        metric_name = metric.split("/")[-1].replace("_", " ").title()
        print(f"\n{metric_name}:")
        print("-" * 60)

        for algo_name, algo_data in data.items():
            if metric not in algo_data:
                continue

            values = algo_data[metric]["values"]

            # Final values (last 10% of training)
            final_portion = max(1, len(values) // 10)
            final_mean = np.mean(values[-final_portion:])
            final_std = np.std(values[-final_portion:])

            # Overall statistics
            overall_min = np.min(values)
            overall_max = np.max(values)
            overall_mean = np.mean(values)

            print(f"{algo_name:25s} | "
                  f"Final: {final_mean:8.3f}±{final_std:6.3f} | "
                  f"Overall: {overall_mean:8.3f} | "
                  f"Range: [{overall_min:8.3f}, {overall_max:8.3f}]")


def main():
    # Load data
    data = load_soccer_competitive_data()
    if data is None:
        return

    # Print summary
    print_summary(data)

    # Create plots
    plot_competitive_metrics(data)


if __name__ == "__main__":
    main()
