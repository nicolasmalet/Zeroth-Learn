import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
import pandas as pd
from cycler import cycler

from zeroth.core.model import Model

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "lines.linewidth": 2.2,
        "legend.fontsize": 9,
        "legend.frameon": False,
        "grid.color": "0.85",
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
    plt.rcParams["axes.prop_cycle"] = cycler(color=[
        "#4477AA", "#EE6677", "#228833",
        "#CCBB44", "#66CCEE", "#AA3377"
    ])


def format_ax(ax: Axes):
    ax.set_axisbelow(True)
    ax.set_yscale('log')
    ax.grid(True, which="major", axis="y")
    ax.grid(False, axis="x")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def smooth_curve(loss: np.ndarray, smooth_span: int) -> np.ndarray:
    if len(loss) == 0:
        return np.array([])
    return np.exp(pd.Series(np.log(loss)).ewm(span=smooth_span, adjust=True).mean())


def plot_curve(ax: Axes, model: Model, label: str, smooth_span: int):
    loss = getattr(model, "train_loss", None)
    if loss is None or len(loss) == 0:
        return

    loss = np.asarray(loss)
    ax.plot(loss, alpha=0.25, linewidth=1.0)
    smooth = smooth_curve(loss, smooth_span)
    ax.plot(smooth, label=label, linewidth=2.5)


def plot_0d(models: list, title: str, smooth_span: int = 50):
    """
    Plots a single graph overlaying multiple models that share the same hyperparameters.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for i, model in enumerate(models):
        label = f"Model {i + 1}" if len(models) > 1 else ""
        plot_curve(ax, model, label, smooth_span)

    fig.suptitle(title, fontweight='bold', fontsize=12)
    ax.set_xlabel("Training steps")
    ax.set_ylabel("Training loss")
    format_ax(ax)

    if len(models) > 1:
        ax.legend(loc="upper right")

    plt.tight_layout()


def plot_1d(models: list, title: str, key: str, smooth_span: int = 50):
    """
    Plots a row of subplots, varying one hyperparameter (key) across columns.
    """
    n_models = len(models)
    fig, axs = plt.subplots(1, n_models, figsize=(4.5 * n_models, 3.5), sharey=True)

    if n_models == 1:
        axs = [axs]

    for i, model in enumerate(models):
        ax = axs[i]
        val = model.id.get(key, "N/A")
        plot_curve(ax, model, "Loss", smooth_span)
        ax.set_title(f"{key} = {val}")
        format_ax(ax)
        ax.legend(loc="upper right")

        if i == 0:
            ax.set_ylabel("Training loss")
        ax.set_xlabel("Training steps")

    fig.suptitle(title, fontweight='bold', fontsize=12)
    plt.tight_layout()


def plot_2d_grid(models: list, title: str, row_key: str, col_key: str, smooth_span: int = 50):
    """
    Plots a grid of subplots varying two hyperparameters: one across rows, one across columns.

    Args:
        models (list): List of model objects.
        title (str): The title of the plot.
        row_key (str): The hyperparameter key changing across rows.
        col_key (str): The hyperparameter key changing across columns.
        smooth_span (int): The span for the EWM average.
    """
    rows = list(dict.fromkeys([m.id[row_key] for m in models]))
    cols = list(dict.fromkeys([m.id[col_key] for m in models]))

    fig, axs = plt.subplots(len(rows), len(cols),
                            figsize=(4.5 * len(cols), 3.5 * len(rows)),
                            sharex=True, sharey=True, squeeze=False)

    for i, r_val in enumerate(rows):
        for j, c_val in enumerate(cols):
            ax = axs[i, j]
            cell_models = [m for m in models
                           if m.id[row_key] == r_val and m.id[col_key] == c_val]

            for model in cell_models:
                others = [f"{k}={v}" for k, v in model.id.items()
                          if k not in [row_key, col_key]]
                label = ", ".join(others) if others else "Model"
                plot_curve(ax, model, label, smooth_span)

            format_ax(ax)

            if i == 0:
                ax.set_title(f"{col_key} = {c_val}")

            if j == len(cols) - 1:
                ax.text(1.02, 0.5, f"{row_key} = {r_val}",
                        transform=ax.transAxes, rotation=-90,
                        va="center", ha="left")
    plt.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.12, wspace=0.10, hspace=0.18)
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(handles),
                   bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=9)

    fig.text(0.5, 0.07, "Training steps", ha='center', fontsize=10)
    fig.text(0.02, 0.5, "Training loss", va='center', rotation='vertical', fontsize=10)



def plot_losses(models: list, title: str, save_path: str = None, smooth_span: int = 100):
    """
    Main entry point for plotting. Automatically detects if the plot should be 0D, 1D, or 2D
    based on the number of variation parameters.

    Args:
        models (list): List of model objects.
        title (str): The title of the plot.
        save_path (str, optional): File path to save the figure (e.g., 'plot.png').
        smooth_span (int): EWM span for smoothing. Defaults to 50.
    """
    set_style()

    keys = list(models[0].id.keys())
    n_vars = len(keys)

    if n_vars == 0:
        plot_0d(models, title, smooth_span)
    elif n_vars == 1:
        plot_1d(models, title, keys[0], smooth_span)
    else:
        plot_2d_grid(models, title, keys[0], keys[1], smooth_span)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()