Instructions for completing `checker.py`.

You could ONLY modify the `permutation.py`, `__init__.py`, and `config.py` files.

# Goal

The goal of `permutation.py` is to implement random permutations on `y`,
then check the performances of ML pipelines on permutated data.

If there is leakage, the performance on permutated data will be higher than random guessing.

So for a given (`X`, `y`, 'covariates') and `MLPipeline`,
we will check the performance of the pipeline on permutated data,
and compare it with the random chance level.

# Design hints

An example:

BUT!! For some uers own pipelines,
they might not have the `fit()` and `evaluate()` methods,
or they might have different method names.

For a more general design, we can allow users to pass in their own functions for fitting and evaluating the model.

```python

from leakly import permute_label

# User's own machine learning pipeline, which may have leakages
test_auc = (
    userOwnPipeline(X, y, *covariates).fit()).evaluate()

# define checker
y_permuted = permute_label(
    y,
    # if 0.5, 50% of samples' labels randomly shuffled.
    perc_permutation=0.5,
    random_state=0)

test_auc_permuted = (
    userOwnPipeline(X, y_permuted, *covariates).fit()).evaluate()
```
