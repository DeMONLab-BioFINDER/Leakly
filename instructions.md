I am building a Python package `Leakly`.

Ignore git history, just focus on the structure and planning of the package.

Please help me build the structure of the package, modules, classes, and functions.

They are under `./leakly/` directory.

For empty scripts, you could write empty modules, classes, and functions with
Input/Output, Docstring, but leave the implementation empty for now.
The focus is on the structure and planning of the package.

Avoid using any external libraries that are not commonly used,
unless absolutely necessary for the functionality of the package.
If you do use external libraries,
please specify them in the `setup.py` file
and include instructions for installing them in the README.

MAKE HUMAN READABLE CODE with GOOD MODULE designs.

Avoid re-invent wheels and OVER-DESIGN!!

Try minimum implementation for each module,
and make sure the code is easy to read and understand.

# Background

Many study pipelines P(`X`, `y`, `covariates`, `config`), `X` is feature matrix,
`y`is target to predict,`covariates`are to control confounding effects.
`config` contains information such as ratio of train/test split, processing.

However, they may have leakages. For example,

- normalization/imputation before train/test split;
- select features from full samples instead only on train splits;
- Tune their model on test samples.

If there is leakage happens, we could randomly shuffle `y`, the final test
set accuracies will be better than chance. We could run many permutations,
the distribution of test set AUC should be larger than random guess level.

# Example of machine learning pipeline

For exemplar usage, we gave a ML pipeline.

PipelineClass(

    # now only supports train/test split for simplicity
    `X`, # feature matrix, N by P
    `y`, # target variable (binary or continuous), N by 1
    `covariates`, # optional, covariate variables, N by C
    `config`: # the pipeline should run in order
        {
            DataProcConfig,
            FeatureSelectionConfig,
            SplitConfig,
            MLConfig,
        }

) -> TestSet AUC

This example has many leakage items, imputation/normalization/feature_selection
should be after train/test split.

Defaults:

    - Continuous variables use Z normalization
    - Categorical variables use one-hot
    - Our default Imputer is KNNImputer,
    - Feature Selection using LinearRegressionDAA,
    - Machine learning model use RandomForest,
    - Evaluation uses AUC.

# Setup

For a typical machine learning pipeline P(`X`, `y`, `covariates`, `config`),
we could randomly shuffle `y` and run the pipeline,
if there is no leakage, the test set AUC should at chance level:
For example, 0.5 for binary classification.

An example of usage:

```python
from leakly import LeakageCheckerOneRun, SummaryPlotter

# User's own machine learning pipeline, which may have leakages
ml_pipeline = UserDefinedMLPipeline(X, y, covariates, config)

# One run of leakage check with 50% permutations
checker = LeakageCheckerOneRun(
    ml_pipeline,
    # percentage of permutations.
    # if 0.5, 50% of samples' labels randomly shuffled.
    perc_permutation=0.5,
    random_state=0,
    result_save_path="results/leakage_check_%random_state.json")

# 100 permutations to get the distribution of test set AUC under leakage
N_permutations = 100
perc_permutation = 1
test_aucs = []
for i in range(N_permutations):
    checker.perc_permutation = perc_permutation
    checker.random_state = i  # for reproducibility
    test_auc = checker.run()
    test_aucs.append(test_auc)

# Plot the distribution of test set AUC under leakage
# Also given p values from permutation test
plotter = SummaryPlotter(test_aucs)
```

# Modules

## `config.py`

- Generate configuration for all experiments
- Should be flexible to support different pipelines, but also have some default settings.
- Use `YAML` for configuration file.

## `data.py`

- Data normlaization, imputation, train/test split
- Should be flexible to support different pipelines, but also have some default settings.
- DO NOT create a data class for simplicity, just use numpy arrays for X, y, covariates.

## `stats.py`

- adjust p values for mutiple comparsions
- permutation test for AUC

## `feature_selection.py`

- feature selection methods, such as linear regression DAA
- should be flexible to support different pipelines, but also have some default settings.
- The overall design should be feature_selection_method(X, y, covariates, config) -> selected_features
- DO NOT use a data class for simplicity, just use numpy arrays for X, y, covariates.
- Please refer to my implementation in `leakly/daa.py` for details.

## `models.py`

- machine learning models, such as random forest, SVM, etc.
- user could define their own machine learning model
- Default is random forest from sklearn
- DO NOT create a data class for simplicity, just use numpy arrays for X, y, covariates.

## `ml_pipeline.py`

- the overall machine learning pipeline, which could be defined by user.
- we give an example, please read `Example of machine learning pipeline` section above for details.

## `checker.py`

- the leakage checker, which runs the machine learning pipeline with different permutations of `y`.
- The overall design should be LeakageChecker(ml_pipeline, perc_permutation, random_state, result_save_path) -> test_auc
- The `run` method should run the pipeline with the given `perc_permutation` and `random_state`, and save the results to `result_save_path`.

## `summary.py`

- the summary plotter, which plots the distribution of test set AUC under leakage.
- The overall design should be SummaryPlotter(test_aucs) -> plot
- The `plot` method should plot the distribution of test set AUC under leakage, and also given p values from permutation test.

## `simulation.py`

- Simulate data for testing the pipeline.
- Please refer to my implementation in `leakly/simulation.py` for details.
