import argparse
import sys
from typing import Any, Dict, List

from isaacgym import gymutil


def create_base_parser(add_help=False):
    """Create base parser with core arguments used across all scripts."""
    parser = argparse.ArgumentParser(add_help=add_help)

    # General
    parser.add_argument(
        "--task",
        type=str,
        default="stompymicro",
        help="Resume training or start testing from a checkpoint. Overrides config file if provided.",
    )
    parser.add_argument(
        "--rl_device",
        type=str,
        default="cuda:0",
        help="Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
    )
    parser.add_argument(
        "--num_envs", type=int, help="Number of environments to create. Overrides config file if provided."
    )
    parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1001,
        help="Maximum number of iterations. Refers to training iterations for `train.py` and playing steps for `play.py`.",
    )

    # Loading model
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from a checkpoint")
    parser.add_argument(
        "--experiment_name", type=str, help="Name of the experiment to run or load. Overrides config file if provided."
    )
    parser.add_argument("--run_name", type=str, help="Name of the run. Overrides config file if provided.")
    parser.add_argument(
        "--load_run", type=str, help="Name of the run to load when resume=True. If -1: will load the last run."
    )
    parser.add_argument(
        "--checkpoint", type=int, help="Saved model checkpoint number. If -1: will load the last checkpoint."
    )

    # Rendering
    parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times")

    return parser


def convert_to_gymutil_format(parser: argparse.ArgumentParser) -> List[Dict[str, Any]]:
    """Convert argparse parser arguments to gymutil custom_parameters format."""
    custom_parameters = []
    for action in parser._actions:
        if action.dest != "help":  # Skip the help action
            param = {
                "name": "--" + action.dest if not action.option_strings else action.option_strings[0],
                "type": action.type,
                "default": action.default,
                "help": action.help,
            }
            if isinstance(action, argparse._StoreTrueAction):
                param.pop("type")  # Remove the type for store_true actions
                param["action"] = "store_true"
            elif isinstance(action, argparse._StoreAction):
                if action.choices:
                    param["choices"] = action.choices
            custom_parameters.append(param)
    return custom_parameters


def parse_args_with_extras(extra_args_fn=None):
    """Parse arguments using both base parser and any extra arguments provided."""
    parser = create_base_parser()

    if extra_args_fn is not None:
        extra_args_fn(parser)

    # Store which arguments were meant to be True by default
    true_by_default = {action.dest for action in parser._actions if action.default is True}

    custom_parameters = convert_to_gymutil_format(parser)
    args = gymutil.parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    # Restore default=True values that weren't explicitly set to False
    for arg_name in true_by_default:
        # Check if this argument wasn't explicitly provided in command line
        if not any(arg.lstrip("-").replace("-", "_") == arg_name for arg in sys.argv[1:]):
            setattr(args, arg_name, True)

    # Add the sim device arguments
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"

    return args


def print_args(args):
    """Pretty print arguments."""
    print("\nArguments:")
    print("-" * 30)
    for arg, value in sorted(vars(args).items()):
        print(f"{arg:>20}: {value}")
    print("-" * 30 + "\n")
