<p align="center">
  <img src="assets/leakly-logo.svg" alt="Leakly logo" width="520">
</p>

<p align="center">
  <a href="https://pypi.org/project/Leakly/"><img alt="PyPI" src="https://img.shields.io/pypi/v/Leakly.svg"></a>
  <a href="https://github.com/DeMONLab-BioFINDER/Leakly/actions/workflows/ci.yml?query=branch%3Amain"><img alt="Build" src="https://github.com/DeMONLab-BioFINDER/Leakly/actions/workflows/ci.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/DeMONLab-BioFINDER/Leakly/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-MIT-green.svg"></a>
</p>

<p align="center">
  <a href="https://colab.research.google.com/github/DeMONLab-BioFINDER/Leakly/blob/main/example.ipynb"><img alt="Open in Google Colab" src="https://colab.research.google.com/assets/colab-badge.svg" height="32"></a>
</p>

# Leakly: Leakage checks for any machine-learning pipeline

`Leakly` uses label permutation to test whether a pipeline still performs above
chance when no true signal is present.

If it does, the pipeline may be leaking test-set information
through preprocessing, feature selection, tuning, or another step.

![Example permutation AUC summary](https://raw.githubusercontent.com/DeMONLab-BioFINDER/Leakly/main/assets/AUC.png)

## Install

```bash
pip install Leakly
```

For the current GitHub checkout:

```bash
git clone https://github.com/DeMONLab-BioFINDER/Leakly.git
cd Leakly
pip install -e .
```

## Quick Start

### <a href="https://colab.research.google.com/github/DeMONLab-BioFINDER/Leakly/blob/main/example.ipynb"><img alt="Open example.ipynb in Colab" src="https://img.shields.io/badge/Open-example.ipynb-F9AB00?logo=googlecolab&logoColor=white" height="28"></a>

Run the first install cell, then run the notebook from top to bottom.

### Run in Python

```python
from leakly import (
    MLPipeline,
    SimulationConfig,
    SummaryPlotter,
    load_example_leakage_config,
    permute_label,
    simulate_dataset,
)

data = simulate_dataset(
    SimulationConfig(
        n_samples=200,
        n_features=1000,
        n_covariates=3,
        effect_fraction=0.1,
        effect_size=0.5,
        random_state=42,
    )
)

config = load_example_leakage_config()
scores = []

for seed in range(100):
    permuted_y = permute_label(data.y, random_state=seed)
    score = (
        # user could replace with any pipeline
        MLPipeline(
            data.X,
            permuted_y,
            covariates=data.covariates,
            config=config,
        )
        .fit()
        .evaluate()
    )
    scores.append(score)

SummaryPlotter(scores, chance_level=0.5).plot("assets/AUC.png")
```

## Principle

1. Permute labels to remove real signal.
2. Run the full pipeline exactly as a user would run it.
3. Compare the score distribution with chance level.
4. Above-chance permuted performance suggests possible leakage.

Leakly includes example configurations for a leaky pipeline and a non-leaky
pipeline so users can see the effect immediately.

## FAQ

**Can Leakly check my own pipeline?**

Yes. Leakly can evaluate any pipeline that takes `X`, `y`, optional `covariates`, and returns a test score. The key is to run the full pipeline exactly as in the real analysis, including preprocessing, feature selection, tuning, and evaluation.

**Why can a leaky pipeline score well on permuted labels?**

After labels are permuted, there should be no real biological, clinical, or statistical signal linking the features to the outcome. Therefore, a properly designed pipeline should not be able to predict the permuted labels better than chance.

A leaky pipeline may still score well because information from the full dataset has entered the analysis before the train/test split or outside the cross-validation loop. For example, leakage can occur when feature selection, scaling, imputation, covariate adjustment, dimensionality reduction, or hyperparameter tuning is performed using all samples before the data are split. In that situation, the test set has already influenced one or more earlier steps of the pipeline.

This is especially dangerous in high-dimensional settings, such as neuroimaging, omics, or biomarker discovery, where there may be many more features than samples. Even after label permutation, random patterns can appear statistically meaningful by chance. If preprocessing or feature selection is allowed to see the full dataset, the pipeline may accidentally select or preserve these random label-specific patterns. The final model can then appear to perform well on the test set, even though the performance is driven by leakage rather than true predictive signal.

Leakly detects this failure mode by asking a simple question: does the pipeline still perform above chance when the labels are meaningless? If yes, the result does not automatically prove exactly where the leakage occurs, but it provides strong evidence that the pipeline should be inspected.

**How many permutations should I run?**

Use 100 for a quick check. Use 1,000 or more for publication-level evidence.

**Why are Colab and local results different?**

Run the first notebook cell after pulling the latest code. Exact matching also
requires the same Leakly and dependency versions.

## License

MIT. See [LICENSE](LICENSE).
