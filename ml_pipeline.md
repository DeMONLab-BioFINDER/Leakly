Instructions for completing `ml_pipeline.py`.

You could modify the `ml_pipeline.py` and `config.py` files.

# Goal

The goal of `ml_pipeline.py` is to implement a typical machine learning pipeline.

Current we support binary classification and regression tasks.

The are two main functions in `ml_pipeline.py`:

1. User could define their pipeline **in order** like sckit-learn pipelines, but with some flexibility.
   For example, the user could choose to do feature selection before or after data normalization.

2. For main steps, there should be config files like other files has defined, but the user could also overwrite the config settings in the code.

# Design hints

configs = create_config() or load_default_config()

example_pipeline = ML_Pipeline(
X, y, covariates [optional],
problem_type=['binary_classification' or 'regression'],
pipeline=[
"imputer": configs["ImputationConfig"],
"normalizer": configs["NormalizationConfig"],
"feature_selection": configs["FeatureSelectionConfig"],
"data_split": configs["SplitConfig"],
"model": configs["ModelConfig"],
])

example_pipeline.fit()
test_predictions = example_pipeline.predict(X_test)
test_probabilities = example_pipeline.predict_proba(X_test)
test_auc = example_pipeline.evaluate(
y_true=y_test, y_pred=test_probabilities, metric="auc")

# save configuration and

example_pipeline.save("path/to/save/model_and_config")
