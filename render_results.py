import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import pandas as pd

def parse_args():
    """Parse command line arguments for the render script."""
    p = argparse.ArgumentParser("Render training results from different algorithms.")
    p.add_argument("--env_id", choices=["Tennis", "Soccer"], default="Tennis",
                   help="Environment to visualize results for")
    p.add_argument("--algorithms", nargs='+',
                   default=["mappo_all_shared", "mappo_critic_shared", "matd3", "masac", "maddpg"],
                   help="Algorithms to compare")
    p.add_argument("--algo_titles", nargs='+',
                   default=["MAPPO (all shared)", "MAPPO (critic shared)", "MATD3", "MASAC", "MADDPG"],
                   help="Custom display names for algorithms (same order as --algorithms)")
    p.add_argument("--results_dir", type=str, default="./results",
                   help="Directory containing resuts files")
    p.add_argument("--save_dir", type=str, default="./figures",
                   help="Directory to save figures")
    p.add_argument("--metrics", nargs='+',
                   default=["rewards", "episode_length"],
                   help="Metrics to visualize")
    p.add_argument("--individual", action="store_true",
                   help="Create individual plots for each algorithm")
    p.add_argument("--single_algo", type=str,
                   help="Create plots for only this specific algorithm")
    return p.parse_args()

def load_data(log_dir, env_id, algorithms, metrics):
    """Load training data for each algorithm from numpy files."""
    data = {}

    # Base directory for results
    base_dir = os.path.join(log_dir, env_id)

    for algo in algorithms:
        # Find all runs for this algorithm
        algo_dirs = glob.glob(os.path.join(base_dir, f"{algo}"))

        if not algo_dirs:
            print(f"No directories found for {algo} in {base_dir}")
            continue

        # Initialize data structure for this algorithm
        algo_data = {metric: {"steps": [], "values": []} for metric in metrics}

        for run_dir in algo_dirs:
            try:
                # Load numpy files
                steps_file = os.path.join(run_dir, "steps_history.npy")
                scores_file = os.path.join(run_dir, "scores_history.npy")
                ep_length_file = os.path.join(run_dir, "episode_length_history.npy")

                if os.path.exists(steps_file) and os.path.exists(scores_file) and os.path.exists(ep_length_file):
                    steps = np.load(steps_file)
                    scores = np.load(scores_file)
                    ep_lengths = np.load(ep_length_file)

                    # If scores is 2D (multiple agents), average across agents
                    if len(scores.shape) > 1:
                        scores = np.mean(scores, axis=1)

                    # Add to data
                    if "rewards" in metrics:
                        algo_data["rewards"]["steps"].append(steps)
                        algo_data["rewards"]["values"].append(scores)

                    if "episode_length" in metrics:
                        algo_data["episode_length"]["steps"].append(steps)
                        algo_data["episode_length"]["values"].append(ep_lengths)
                else:
                    print(f"Missing required files in {run_dir}")

            except Exception as e:
                print(f"Error processing {run_dir}: {e}")

        # Calculate statistics across runs for each algorithm
        for metric in metrics:
            if algo_data[metric]["steps"]:
                # Interpolate to common x-axis
                all_steps = np.unique(np.concatenate(algo_data[metric]["steps"]))
                all_steps.sort()

                interp_values = []
                for steps, values in zip(algo_data[metric]["steps"], algo_data[metric]["values"]):
                    if len(steps) > 1:  # Need at least 2 points for interpolation
                        interp_values.append(np.interp(
                            all_steps,
                            steps,
                            values,
                            left=np.nan,
                            right=np.nan
                        ))

                if interp_values:
                    interp_array = np.array(interp_values)

                    # Calculate statistics
                    mean_values = np.nanmean(interp_array, axis=0)
                    min_values = np.nanmin(interp_array, axis=0)
                    max_values = np.nanmax(interp_array, axis=0)
                    std_values = np.nanstd(interp_array, axis=0)

                    # Remove NaN values from the beginning and end
                    valid_indices = ~np.isnan(mean_values)
                    if np.any(valid_indices):
                        valid_steps = all_steps[valid_indices]
                        valid_mean = mean_values[valid_indices]
                        valid_min = min_values[valid_indices]
                        valid_max = max_values[valid_indices]
                        valid_std = std_values[valid_indices]

                        # Calculate 100-step rolling average
                        window_size = min(100, len(valid_mean))
                        if window_size > 1:
                            rolling_mean = pd.Series(valid_mean).rolling(window=window_size, min_periods=1).mean().values
                        else:
                            rolling_mean = valid_mean

                        algo_data[metric] = {
                            "steps": valid_steps,
                            "mean": valid_mean,
                            "min": valid_min,
                            "max": valid_max,
                            "std": valid_std,
                            "rolling_mean": rolling_mean
                        }
                    else:
                        algo_data[metric] = {
                            "steps": [],
                            "mean": [],
                            "min": [],
                            "max": [],
                            "std": [],
                            "rolling_mean": []
                        }

        data[algo] = algo_data

    return data

def render_comparison(data, env_id, metrics, save_dir, algo_titles=None):
    """Render comparison plots for the specified metrics with statistics."""
    os.makedirs(save_dir, exist_ok=True)

    # Set up the plot style
    plt.style.use('default')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

    # Create mapping from algorithm names to display titles
    algo_names = list(data.keys())
    if algo_titles and len(algo_titles) == len(algo_names):
        title_map = dict(zip(algo_names, algo_titles))
    else:
        title_map = {algo: algo for algo in algo_names}  # Use original names if no titles provided

    # Create separate plots for each metric
    for metric in metrics:
        # Check if we have data for this metric
        has_data = any(metric in data[algo] and len(data[algo][metric]["steps"]) > 0 for algo in data.keys())
        if not has_data:
            continue

        # Create single plot
        _, ax = plt.subplots(1, 1, figsize=(12, 8))

        for i, (algo, algo_data) in enumerate(data.items()):
            if metric in algo_data and len(algo_data[metric]["steps"]) > 0:
                steps = algo_data[metric]["steps"]
                mean_vals = algo_data[metric]["mean"]
                rolling_vals = algo_data[metric]["rolling_mean"]

                color = colors[i % len(colors)]
                display_title = title_map[algo]

                # Plot raw mean as light dashed line
                ax.plot(steps, mean_vals, '--', alpha=0.3, linewidth=1.1, color=color)

                # Plot 100-episode rolling average as thick line
                ax.plot(steps, rolling_vals, linewidth=3, color=color, label=display_title)

        # Styling
        if metric == "rewards":
            metric_title = "Rewards"
            main_title = f"{env_id} Algorithm Comparison - Rewards"
            subtitle = "(100-episode moving average)"
        else:
            metric_title = "Episode Length"
            main_title = f"{env_id} Algorithm Comparison - Episode Length"
            subtitle = "(100-episode moving average)"

        ax.set_title(f"{main_title}\n{subtitle}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Environment Steps", fontsize=12)
        ax.set_ylabel(metric_title, fontsize=12)

        # Position legend appropriately
        legend_loc = 'lower right' if metric == 'rewards' else 'upper left'
        ax.legend(loc=legend_loc, fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save individual figure
        save_path = os.path.join(save_dir, f"{env_id}_{metric}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {metric} figure to {save_path}")
        plt.close()

def render_individual_algorithm(data, env_id, algo, metrics, save_dir, algo_title=None):
    """Render individual plots for a specific algorithm showing mean and moving average."""
    os.makedirs(save_dir, exist_ok=True)

    if algo not in data:
        print(f"No data found for algorithm: {algo}")
        return

    # Set up colors for mean and rolling average
    mean_color = '#A8D0E6'  # Light blue
    rolling_color = '#2E86C1'  # Medium blue

    display_title = algo_title if algo_title else algo

    # Create separate plots for each metric
    for metric in metrics:
        if metric not in data[algo] or len(data[algo][metric]["steps"]) == 0:
            continue

        # Create single plot
        _, ax = plt.subplots(1, 1, figsize=(12, 8))

        steps = data[algo][metric]["steps"]
        mean_vals = data[algo][metric]["mean"]
        rolling_vals = data[algo][metric]["rolling_mean"]

        # Plot raw mean as thin dashed line
        ax.plot(steps, mean_vals, '--', alpha=0.7, linewidth=1.5, color=mean_color,
                label='Episode Mean')

        # Plot 100-episode rolling average as thick line
        ax.plot(steps, rolling_vals, linewidth=3, color=rolling_color,
                label='100-Episode Moving Average')

        # Styling
        if metric == "rewards":
            metric_title = "Rewards"
            main_title = f"{env_id} - {display_title} Training Rewards"
        else:
            metric_title = "Episode Length"
            main_title = f"{env_id} - {display_title} Training Episode Length"

        subtitle = "(100-episode moving average)"

        ax.set_title(f"{main_title}\n{subtitle}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Environment Steps", fontsize=12)
        ax.set_ylabel(metric_title, fontsize=12)

        # Position legend appropriately
        legend_loc = 'lower right' if metric == 'rewards' else 'upper left'
        ax.legend(loc=legend_loc, fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save individual figure
        safe_algo_name = algo.replace(" ", "_").replace("(", "").replace(")", "")
        save_path = os.path.join(save_dir, f"{env_id}_{safe_algo_name}_{metric}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved {display_title} {metric} figure to {save_path}")
        plt.close()

def print_summary_statistics(data, env_id, metrics):
    """Print summary statistics for each algorithm and metric."""
    print(f"\n=== Summary Statistics for {env_id} ===")

    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * 50)

        for algo, algo_data in data.items():
            if metric in algo_data and len(algo_data[metric]["steps"]) > 0:
                mean_vals = algo_data[metric]["mean"]
                min_vals = algo_data[metric]["min"]
                max_vals = algo_data[metric]["max"]
                rolling_vals = algo_data[metric]["rolling_mean"]

                # Final values (last 10% of training)
                final_portion = max(1, len(mean_vals) // 10)
                final_mean = np.mean(mean_vals[-final_portion:])
                final_std = np.std(mean_vals[-final_portion:])
                final_rolling = np.mean(rolling_vals[-final_portion:])

                # Overall statistics
                overall_min = np.min(min_vals)
                overall_max = np.max(max_vals)
                overall_mean = np.mean(mean_vals)

                print(f"{algo:20s} | "
                      f"Final: {final_mean:8.3f}±{final_std:6.3f} | "
                      f"Overall: {overall_mean:8.3f} | "
                      f"Range: [{overall_min:8.3f}, {overall_max:8.3f}] | "
                      f"Final 100-avg: {final_rolling:8.3f}")

def main():
    args = parse_args()

    # Define metrics to extract
    metrics = ["rewards", "episode_length"]

    # Load data for each algorithm
    data = load_data(args.results_dir, args.env_id, args.algorithms, metrics)

    # Print summary statistics
    print_summary_statistics(data, args.env_id, metrics)

    # Create mapping from algorithm names to display titles
    algo_names = list(data.keys())
    if args.algo_titles and len(args.algo_titles) == len(algo_names):
        title_map = dict(zip(algo_names, args.algo_titles))
    else:
        title_map = {algo: algo for algo in algo_names}

    # Handle different rendering modes
    if args.single_algo:
        # Render plots for a single specific algorithm
        if args.single_algo in data:
            algo_title = title_map.get(args.single_algo, args.single_algo)
            render_individual_algorithm(data, args.env_id, args.single_algo, metrics,
                                      args.save_dir, algo_title)
        else:
            print(f"Algorithm '{args.single_algo}' not found in data. Available: {list(data.keys())}")
    elif args.individual:
        # Render individual plots for all algorithms
        for algo in data.keys():
            algo_title = title_map.get(algo, algo)
            render_individual_algorithm(data, args.env_id, algo, metrics,
                                      args.save_dir, algo_title)
    else:
        # Default: render comparison plots
        render_comparison(data, args.env_id, metrics, args.save_dir, args.algo_titles)

if __name__ == "__main__":
    main()