# Exact pairs probability Monte Carlo simulation

This repository contains Python code to perform Monte Carlo simulations of scenarios in which a set of `n` elements is randomly drawn from a set of `m` elements, with replacement.

The code is written in such a way to enable the description of any particular scenarios of interest within the context of the problem. Implementations of the following scenarios are provided:

- Drawing all unique elements.
- Drawing no pairs, but allowing for the drawing of the same element multiple times.

## Installation

Environment management is with [poetry](https://python-poetry.org/). Once poetry is installed, clone the repository and run `poetry install` to install the dependencies.

## Usage

The simulation may be run with `exact_pairs_probability.py` by executing `exact_pairs_probability.py` with the desired options. For example:

```bash
poetry run python exact_pairs_probability.py --n 5 --m 26 --k 1000000 --experiment_type all_unique
```

This will run the simulation for the scenario in which `n=5` elements are drawn from a set of `m=26` elements, with `k=1000000` iterations, and the probability that all elements are unique is calculated.

Run `poetry run python exact_pairs_probability.py --help` for a list of available options.

### Plotting results

Results are plotted during execution of `exact_pairs_probability.py`. By passing an argument to the `--save_filename` option, the data may be saved to a file for later plotting. The `plot_results.py` script may be used to plot the results.

## For developers

Set up the development environment using `poetry install` then launch a shell with `poetry shell`.

Pre-commit hooks are available to enable automated code formatting and linting, and can be installed with `pre-commit install`.