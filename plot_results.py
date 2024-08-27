import argparse

import matplotlib.pyplot as plt
import numpy as np


class ResultsPlotter:
    """
    Class to plot the results of a probability experiment.

    Parameters
    ----------
    filename : str
        Name of the file with the results.
    """

    def __init__(self, filename):
        self.filename = filename

    def load_results(self) -> np.ndarray:
        """
        Load the results of the experiment from a file.

        Returns
        -------
        np.ndarray
            Results of the experiment.
        """
        self.results = np.loadtxt(self.filename, delimiter=",")
        self.k = len(self.results)
        return self.results

    def plot_cumulative_probability(
        self,
        true_probability: float = None,
        xlim: tuple = (None, None),
        ylim: tuple = (None, None),
        title: str = None,
    ) -> None:
        """
        Plot the cumulative calculated probability of the experiment results.

        Parameters
        ----------
        true_probability : float, optional
            True probability of the event, by default None
        xlim : tuple, optional
            Limits for the x-axis, by default (None, None)
        ylim : tuple, optional
            Limits for the y-axis, by default (None, None)
        title : str, optional
            Title of the plot, by default None
        """
        fig, ax = plt.subplots(
            figsize=(5, 3.5), constrained_layout=True, dpi=300
        )
        ax.plot(
            np.cumsum(self.results) / np.arange(1, self.k + 1),
            label="Calculated probability",
        )
        ax.set_xlabel("Number of experiments")
        ax.set_ylabel("Probability")
        if true_probability is not None:
            ax.axhline(
                true_probability,
                color="red",
                linestyle="--",
                label="True probability",
            )
        ax.legend()
        ax.grid()
        ax.set_ylim(ylim)
        ax.set_xlim(xlim)
        if title is not None:
            ax.set_title(title)

        self.fig = fig
        self.ax = ax

    def save_plot(self, filename: str = None) -> None:
        """
        Save the plot to a file.

        Parameters
        ----------
        filename : str, optional
            Name of the file to save the plot, by default None which
            saves the plot with the same name as the results file but
            with a .png extension.
        """
        if filename is None:
            filename = (
                "".join(s for s in self.filename.split(".")[:-1]) + ".png"
            )
        self.fig.savefig(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot results of probability experiments"
    )
    parser.add_argument(
        "filename", type=str, help="Name of the file with the results"
    )
    parser.add_argument(
        "--true_probability",
        type=float,
        help="True probability of the event",
        default=None,
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        help="Limits for the x-axis",
        default=(None, None),
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        help="Limits for the y-axis",
        default=(None, None),
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title of the plot",
        default=None,
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot instead of saving it",
    )
    args = parser.parse_args()

    results_plotter = ResultsPlotter(args.filename)
    results_plotter.load_results()
    results_plotter.plot_cumulative_probability(
        true_probability=args.true_probability,
        xlim=args.xlim,
        ylim=args.ylim,
        title=args.title,
    )
    if args.show:
        plt.show()
    else:
        results_plotter.save_plot()
