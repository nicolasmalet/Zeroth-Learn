from zeroth.core.dataclasses_utils import get_catalog_values
from .experiments import EXPERIMENTS


def main(do_train, do_test, nb_print_train, do_plot_train, do_save):
    experiment = EXPERIMENTS.small_sizes.instantiate()  # choose the experiment to run
    experiment.launch(do_train, do_test, nb_print_train,
                      do_plot_train, do_save)


def run_all_experiments(do_train, do_test, nb_print_train, do_plot_train, do_save):
    experiments = get_catalog_values(EXPERIMENTS)
    for experiment in experiments:
        experiment.instantiate().launch(do_train, do_test, nb_print_train, do_plot_train, do_save)