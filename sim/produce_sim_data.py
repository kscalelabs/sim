import argparse
import multiprocessing as mp
import subprocess
import time

from tqdm import tqdm


def run_simulation(sim_idx: int, args: argparse.Namespace) -> None:
    """Run a single simulation instance with the given parameters."""
    cmd = [
        "python",
        "sim/sim2sim.py",
        "--embodiment",
        args.embodiment,
        "--load_model",
        args.load_model,
        "--log_h5",
        "--no_render",
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Simulation {sim_idx} failed with error: {e}")


def run_parallel_sims(num_threads: int, args: argparse.Namespace) -> None:
    """Run multiple simulation instances in parallel with a delay between each start."""
    processes = []
    delay_between_starts = 0.1  # Adjust the delay (in seconds) as needed

    for idx in range(num_threads):
        p = mp.Process(target=run_simulation, args=(idx, args))
        p.start()
        processes.append(p)
        time.sleep(delay_between_starts)  # Introduce a delay before starting the next process

    # Wait for all processes to finish with a progress bar
    for p in tqdm(processes, desc="Running parallel simulations"):
        p.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run parallel simulations.")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of parallel simulations to run")
    parser.add_argument("--embodiment", default="stompypro", type=str, help="Embodiment name")
    parser.add_argument("--load_model", default="examples/walking_pro.onnx", type=str, help="Path to model to load")

    args = parser.parse_args()

    # Run 100 examples total, in parallel batches
    num_examples = 2000
    num_batches = (num_examples + args.num_threads - 1) // args.num_threads

    for batch in range(num_batches):
        examples_remaining = num_examples - (batch * args.num_threads)
        threads_this_batch = min(args.num_threads, examples_remaining)
        print(f"\nRunning batch {batch+1}/{num_batches} ({threads_this_batch} simulations)")
        run_parallel_sims(threads_this_batch, args)
