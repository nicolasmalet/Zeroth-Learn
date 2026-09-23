from pathlib import Path

import numpy as np

from . import experiments


RESULTS_DIR = Path("results/nb_perturbations_adam_sgd")


def main() -> None:
    np.random.seed(42)
    experiment = experiments.nb_perturbations_adam_sgd.instantiate()
    experiment.train_models(nb_print=3)
    experiment.save_df(RESULTS_DIR)
    for model in experiment.models:
        model.save_loss(RESULTS_DIR / model.get_folder_name())
    figure = experiment.plot_losses(title="MNIST: perturbation count, Adam vs SGD",
                                    plot_dimension=1,
                                    smooth_fraction=0.05)
    figure.savefig(RESULTS_DIR / "training_loss.png")
    print(f"Results saved to {RESULTS_DIR}")
