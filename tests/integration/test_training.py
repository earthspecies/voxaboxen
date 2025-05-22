"""
This is a minimal example to check that training and evaluation runs properly
"""

from voxaboxen.project.project_setup import project_setup
from voxaboxen.training.train_model import train_model


def main() -> None:
    args = [
        "--train-info-fp=datasets/integration_test_data/train_info.csv",
        "--val-info-fp=datasets/integration_test_data/val_info.csv",
        "--test-info-fp=datasets/integration_test_data/test_info.csv",
        "--project-dir=projects/integration_test",
    ]
    project_setup(args)

    args = [
        "--project-config-fp=projects/integration_test/project_config.yaml",
        "--name=test_bidir",
        "--lr=.00005",
        "--batch-size=1",
        "--n-epochs",
        "1",
        "--bidirectional",
        "-t",
        "--exists-strategy=overwrite",
    ]
    train_model(args)


if __name__ == "__main__":
    main()


def test_error() -> None:
    main()
