# Exact pairs probability monte carlo simulation

Environment management is with [poetry](https://python-poetry.org/).

Install dependencies with `poetry install`.

Run the simulation with `poetry run python exact_pairs_probability.py`.

## For developers

Start a shell with `poetry shell`.

Install pre-commit hooks with `pre-commit install`.

## Plot results

```bash
python plot_results.py all_unique-n_5-m_26-k_1000000_results.txt --true_probability=0.664367 --xlim 0 1000000 --ylim 0.65772333 0.67101067 --title "All unique"
```

```bash
python plot_results.py no_exact_pairs-n_5-k_10000000_results.txt --true_probability=0.677773 --xlim 0 10000000 --ylim 0.67099526 0.68455073 --title "No exact pairs"
```