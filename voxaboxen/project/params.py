"""
Parameters for project setup
"""

import argparse
import os
from typing import List, Union

import yaml


def save_params(args: argparse.Namespace) -> None:
    """Save a copy of the params used for this experiment"""
    params_file = os.path.join(args.project_dir, "project_config.yaml")

    args_dict = {}
    for name in sorted(vars(args)):
        val = getattr(args, name)
        args_dict[name] = val

    with open(params_file, "w") as f:
        yaml.dump(args_dict, f)

    print(
        f"Saved config to {params_file}. "
        "You may now edit this file if you want some classes "
        "to be omitted or treated as Unknown"
    )


def parse_project_args(
    args: Union[argparse.Namespace, List[str]],
) -> argparse.Namespace:
    """
    Parse project-level parameters from command-line arguments or a namespace.

    Parameters
    ----------
    args : Union[argparse.Namespace, List[str]]
        Either a pre-existing argparse.Namespace or a list of command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with project configuration.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-info-fp",
        type=str,
        required=True,
        help="filepath of csv with train info",
    )
    parser.add_argument(
        "--val-info-fp", type=str, default=None, help="filepath of csv with val info"
    )
    parser.add_argument(
        "--test-info-fp", type=str, required=True, help="filepath of csv with test info"
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        required=True,
        help="directory where project will be stored",
    )

    args = parser.parse_args(args)

    for split in ["train", "val", "test"]:
        if getattr(args, f"{split}_info_fp") is None:
            setattr(
                args,
                f"{split}_info_fp",
                os.path.join(args.data_dir, f"{split}_info.csv"),
            )
    return args
