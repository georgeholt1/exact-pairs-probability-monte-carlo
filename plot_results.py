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

    def load_results(self):
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
        self, true_probability=None, xlim=(None, None), ylim=(None, None)
    ):
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
    args = parser.parse_args()

    results_plotter = ResultsPlotter(args.filename)
    results_plotter.load_results()
    results_plotter.plot_cumulative_probability(
        true_probability=args.true_probability,
        xlim=args.xlim,
        ylim=args.ylim,
    )

    plt.show()
