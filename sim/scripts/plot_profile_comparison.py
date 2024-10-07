import argparse

import matplotlib.pyplot as plt
import pandas as pd


def plot_trajectories(sim_log: str, real_log: str) -> None:
    # Load data from CSV files
    sim_df = pd.read_csv(sim_log, header=0, names=['Step', 'Desired Position', 'Actual Position'])
    real_df = pd.read_csv(real_log, header=0, names=['Step', 'Desired Position', 'Actual Position'])

    # Convert 'Step' column to integers
    sim_df['Step'] = sim_df['Step'].astype(int)
    real_df['Step'] = real_df['Step'].astype(int)

    # Clip data to the first 10,000 steps
    sim_df = sim_df[sim_df['Step'] < 10000]
    real_df = real_df[real_df['Step'] < 10000]

    # Get actual positions
    sim_actual_position = sim_df['Actual Position']
    real_actual_position = real_df['Actual Position']

    # Get desired positions
    sim_desired_position = sim_df['Desired Position']
    real_desired_position = real_df['Desired Position']

    # Plot actual and desired positions
    plt.figure(figsize=(10, 6))
    plt.plot(sim_df['Step'].to_numpy(), sim_actual_position.to_numpy(), label='Simulated Actual Position', color='blue')
    plt.plot(real_df['Step'].to_numpy(), real_actual_position.to_numpy(), label='Real Actual Position', color='red')
    # plt.plot(sim_df['Step'].to_numpy(), sim_desired_position.to_numpy(), label='Simulated Desired Position', color='cyan', linestyle='--')
    # plt.plot(real_df['Step'].to_numpy(), real_desired_position.to_numpy(), label='Real Desired Position', color='magenta', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Position')
    plt.title('Trajectory Comparison')
    plt.legend()
    plt.grid(True)

    # Save plot to file
    plt.savefig('profile_comparison.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot trajectories from CSV files.')
    parser.add_argument('--sim_log', default='sim_log.csv', type=str, help='Path to the sim log csv')
    parser.add_argument('--real_log', default='real_log.csv', type=str, help='Path to the real log csv')

    args = parser.parse_args()

    plot_trajectories(args.sim_log, args.real_log)
