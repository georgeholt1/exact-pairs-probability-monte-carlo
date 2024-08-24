import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class ProbabilityExperiment:
    """
    Parent class to perform probability experiments and calculate
    probabilities. The class is designed to be inherited by child
    classes that implement the run_experiment() and
    calculate_true_probability() methods.

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

    Methods
    -------
    run_experiment()
        To be implemented by child classes. Performs a single experiment
        and returns True if the experiment outcome is the case being
        tested, False otherwise.
    simulate_probability()
        Runs multiple experiments and calculates the probability of the
        case being tested.
    calculate_true_probability()
        To be implemented by child classes. Calculates the true
        probability of the case being tested.
    plot_results()
        Plots the calculated probability as a function of the number of
        experiments and compares it with the true probability.
    """

    def __init__(self, n, m, k, seed=42):
        self.n = n
        self.m = m
        self.k = k
        self.rng = np.random.default_rng(seed=seed)
        self.results = []

    def run_experiment(self):
        """
        To be implemented by child classes. Performs a single experiment
        and returns True if the experiment outcome is the case being
        tested, False otherwise.
        """
        raise NotImplementedError

    def simulate_probability(self):
        """
        Runs multiple experiments and calculates the probability of all
        participants choosing different items.

        Returns
        -------
        float
            Calculated probability.
        """
        self.results = [self.run_experiment() for _ in tqdm(range(self.k))]
        self.simulated_probability = np.mean(self.results)
        return self.simulated_probability

    def calculate_true_probability(self):
        """
        To be implemented by child classes. Calculates the true
        probability of the case being tested.
        """
        raise NotImplementedError

    def plot_results(self):
        """
        Plots the calculated probability as a function of the number of
        experiments and compares it with the true probability.
        """
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        ax.plot(
            np.cumsum(self.results) / np.arange(1, self.k + 1),
            label="Calculated probability",
        )
        ax.set_xlabel("Number of experiments")
        ax.set_ylabel("Probability")
        ax.axhline(
            self.true_probability,
            color="red",
            linestyle="--",
            label="True probability",
        )
        ax.legend()
        ax.grid()
        ax.set_ylim(
            self.true_probability - self.true_probability / 100,
            self.true_probability + self.true_probability / 100,
        )
        ax.set_xlim(0, self.k)
        plt.show()


class AllUnique(ProbabilityExperiment):
    """
    Child class to perform probability experiments and calculate
    probabilities for the case where all participants choose different
    items.

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

    Methods
    -------
    run_experiment()
        Performs a single experiment to check if any two participants
        choose the same item.
    calculate_true_probability()
        Calculates the true probability of any two participants choosing
        the same item.
    """

    def run_experiment(self):
        """
        Performs a single experiment to check if all participants choose
        different items.

        Returns
        -------
        bool
            True if any two participants chose the same item, False
            otherwise.
        """
        choices = self.rng.integers(1, self.m + 1, self.n)
        unique_choices = np.unique(choices)
        return len(unique_choices) == self.n

    def calculate_true_probability(self):
        """
        Calculates the true probability of all participants choosing
        different items.

        Returns
        -------
        float
            True probability.
        """
        self.true_probability = math.factorial(self.m) / (
            self.m**self.n * math.factorial(self.m - self.n)
        )
        return self.true_probability


def main(n, m, k, experiment_type="all_unique"):
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
    experiment_type : str
        Type of experiment to run. Default is "all_unique".
    """
    experiment = ProbabilityExperiment(n, m, k)

    print("Running experiments...")
    simulated_prob = experiment.simulate_probability()
    print(f"Simulated Probability: {simulated_prob:.6f}")

    true_prob = experiment.calculate_true_probability()
    print(f"True Probability: {true_prob:.6f}")

    experiment.plot_results()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exact pairs probability")
    parser.add_argument(
        "--n", type=int, default=5, help="Number of participants"
    )
    parser.add_argument("--m", type=int, default=26, help="Number of items")
    parser.add_argument(
        "--k", type=int, default=1000000, help="Number of experiments"
    )
    parser.add_argument(
        "--experiment_type",
        type=str,
        default="all_unique",
        help="Type of experiment to run",
    )
    args = parser.parse_args()

    main(args.n, args.m, args.k)
