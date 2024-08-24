import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class ProbabilityExperiment:
    """
    Class to perform probability experiments and calculate probabilities.

    Parameters
    ----------
    n : in
        Number of participants.
    m : int
        Number of items.
    k : int
        Number of experiments.
    seed : int
        Seed for the random number generator.

    Attributes
    ----------
    n : int
        Number of participants.
    m : int
        Number of items.
    k : int
        Number of experiments.
    rng : numpy.random.Generator
        Random number generator.
    results : list
        List to store the results of the experiments.
    """

    def __init__(self, n, m, k, seed=42):
        self.n = n
        self.m = m
        self.k = k
        self.rng = np.random.default_rng(seed=seed)
        self.results = []

    def run_experiment(self):
        """
        Performs a single experiment to check if all participants choose
        different items.

        Returns
        -------
        bool
            True if all participants chose different items, False
            otherwise.
        """
        choices = self.rng.integers(1, self.m + 1, self.n)
        unique_choices = np.unique(choices)
        return len(unique_choices) == self.n

    def calculate_probability(self):
        """
        Runs multiple experiments and calculates the probability of all
        participants choosing different items.

        Returns
        -------
        float
            Calculated probability.
        """
        self.results = [self.run_experiment() for _ in tqdm(range(self.k))]
        return np.mean(self.results)

    def true_probability(self):
        """
        Calculates the true probability of all participants choosing
        different items.

        Returns
        -------
        float
            True probability.
        """
        return math.factorial(self.m) / (
            self.m**self.n * math.factorial(self.m - self.n)
        )

    def plot_results(self, calculated_prob, true_prob):
        """
        Plots the calculated probability as a function of the number of
        experiments and compares it with the true probability.

        Parameters
        ----------
        calculated_prob : float
            Calculated probability from the experiments.
        true_prob : float
            True probability.
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.plot(
            np.cumsum(self.results) / np.arange(1, self.k + 1),
            label="Calculated probability",
        )
        ax.set_xlabel("Number of experiments")
        ax.set_ylabel("Probability")
        ax.axhline(
            true_prob, color="red", linestyle="--", label="True probability"
        )
        ax.legend()
        ax.grid()
        ax.set_ylim(true_prob - true_prob / 100, true_prob + true_prob / 100)
        ax.set_xlim(0, self.k)
        plt.show()


def main(n, m, k):
    """
    Main function to run the probability experiments, calculate
    probabilities, and plot the results.

    Parameters
    ----------
    n : int
        Number of participants
    m : int
        Number of items
    k : int
        Number of experiments
    """
    experiment = ProbabilityExperiment(n, m, k)

    print("Running experiments...")
    calculated_prob = experiment.calculate_probability()
    print(f"Calculated Probability: {calculated_prob:.6f}")

    true_prob = experiment.true_probability()
    print(f"True Probability: {true_prob:.6f}")

    experiment.plot_results(calculated_prob, true_prob)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact pairs probability")
    parser.add_argument(
        "--n", type=int, default=5, help="Number of participants"
    )
    parser.add_argument("--m", type=int, default=26, help="Number of items")
    parser.add_argument(
        "--k", type=int, default=1000000, help="Number of experiments"
    )
    args = parser.parse_args()
    main(args.n, args.m, args.k)
