from zeroth.utils.dataclasses_utils import get_catalog_values
from .experiments import EXPERIMENTS


def main(do_train: bool, do_test: bool, nb_print_train: int, do_plot_train: bool, do_save: bool) -> None:
    # run_all_experiments(do_train, do_test, nb_print_train, do_plot_train, do_save)
    experiment = EXPERIMENTS.nb_perturbations.instantiate()  # choose the experiment to run
    experiment.launch(do_train, do_test, nb_print_train,
                      do_plot_train, do_save)


def run_all_experiments(do_train: bool, do_test: bool, nb_print_train: int, do_plot_train: bool, do_save: bool) -> None:
    experiments = get_catalog_values(EXPERIMENTS)[1:]
    for experiment in experiments:
        experiment.instantiate().launch(do_train, do_test, nb_print_train, do_plot_train, do_save)
