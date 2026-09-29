"""Run one named MNIST experiment and save its results."""

import sys
from pathlib import Path

import numpy as np

from . import experiments


def main() -> None:
    name = sys.argv[1]
    np.random.seed(42)
    experiment = getattr(experiments, name).instantiate()
    output = Path("results/all_experiments") / name

    for model in experiment.models:
        model.train(nb_print=0)
        model.save_loss(output / model.get_folder_name())

    experiment.save_df(output)
    figure = experiment.plot_losses(name.replace("_", " "), min(len(experiment.config.variations), 1), 0.05)
    figure.savefig(output / "training_loss.png")
    print(f"Saved {name} to {output}")


if __name__ == "__main__":
    main()
