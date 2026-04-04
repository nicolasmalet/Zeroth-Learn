from . import experiments


def main(do_train: bool, do_test: bool, nb_print_train: int, do_plot_train: bool, do_save: bool) -> None:
    experiment = experiments.nb_perturbations.instantiate()  # choose the experiment to run
    experiment.launch(do_train, do_test, nb_print_train,
                      do_plot_train, do_save)
