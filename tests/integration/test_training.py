"""
This is a minimal example to check that training and evaluation runs properly
"""

from voxaboxen.project.project_setup import project_setup
from voxaboxen.training.train_model import train_model


def main() -> None:
    args = [
        "--data-dir=datasets/integration_test_data",
        "--project-dir=projects/integration_test",
    ]
    project_setup(args)

    args = [
        "--project-config-fp=projects/integration_test/project_config.yaml",
        "--name=test_bidir",
        "--lr=.00005",
        "--batch-size=1",
        "--n-map=10",
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
