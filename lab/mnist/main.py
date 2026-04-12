from . import experiments


def main() -> None:
    experiment = experiments.adam_vs_sgd.instantiate()  # choose the experiment to run
    #experiment.train_models(nb_print=3)
    experiment.plot_losses(title="",
                           plot_dimension=0,
                           smooth_fraction=0.05)
    experiment.plot_losses(title="",
                           plot_dimension=1,
                           smooth_fraction=0.05)
