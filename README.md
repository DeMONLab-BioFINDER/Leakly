<p align="center">
  <img src="https://raw.githubusercontent.com/DeMONLab-BioFINDER/Leakly/main/assets/leakly-logo.svg" alt="Leakly logo" width="520">
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

## Principle

1. Permute labels to remove real signal.
2. Run the full pipeline exactly as a user would run it.
3. Compare the score distribution with chance level.
4. Above-chance permuted performance suggests possible leakage.

Leakly includes example configurations for a leaky pipeline and a non-leaky
pipeline so users can see the effect immediately.

![Example permutation AUC summary](https://raw.githubusercontent.com/DeMONLab-BioFINDER/Leakly/main/assets/AUC.png)

## Install

```bash
pip install Leakly
```

For notebook environments that need the optional notebook dependencies:

```bash
pip install "Leakly[notebook]"
```

For the current GitHub checkout:

```bash
git clone https://github.com/DeMONLab-BioFINDER/Leakly.git
cd Leakly
pip install -e .
```

## Quick Start

### <a href="https://colab.research.google.com/github/DeMONLab-BioFINDER/Leakly/blob/main/example.ipynb"><img alt="Open example.ipynb in Colab" src="https://img.shields.io/badge/Open-example.ipynb-F9AB00?logo=googlecolab&logoColor=white" height="28"></a>

### Key Python snippet

```python
from leakly import (
    MLPipeline,
    SummaryPlotter,
    load_example_leakage_config,
    permute_label,
)

scores = []
for seed in range(100):
    permuted_y = permute_label(data.y, random_state=seed)
    score = (
        # user could replace with any pipeline
        MLPipeline(
            data.X,
            permuted_y,
            covariates=data.covariates,
            config=load_example_leakage_config(),
        ).fit()).evaluate()
    scores.append(score)

SummaryPlotter(scores, chance_level=0.5).plot("assets/AUC.png")
```

## FAQ

**Can Leakly check my own pipeline?**

Yes. Leakly can evaluate any pipeline that takes `X`, `y`, optional `covariates`, and returns a test score. The key is to run the full pipeline exactly as in the real analysis, including preprocessing, feature selection, tuning, and evaluation.

**Why can a leaky pipeline score well on permuted labels?**

After label permutation, there should be no real biological, clinical, or statistical link between features and outcomes. A valid pipeline should therefore perform near chance.

A leaky pipeline may still score well if information from the full dataset enters the analysis before the train/test split or outside the cross-validation loop. Common sources include feature selection, scaling, imputation, covariate adjustment, dimensionality reduction, or hyperparameter tuning performed on all samples.

This is especially problematic in high-dimensional data, such as neuroimaging, omics, or biomarker studies, where random label-specific patterns can appear meaningful by chance. If the test set influences preprocessing or feature selection, the model may "remember" these random patterns and show inflated performance.

**How many permutations should I run?**

Use 100 for a quick check. Use 1,000 or more for publication-level evidence.

## License

MIT. See [LICENSE](LICENSE).
