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

# Leakly

**Leakage checks for any machine-learning pipeline.**

Leakly uses label permutation to test whether a pipeline still performs above
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

Yes. Any pipeline that takes `X`, `y`, optional covariates, and returns a test
score can be evaluated with the same permutation idea.

**Why can a leaky pipeline score well on permuted labels?**

Because information from the full dataset can enter preprocessing or feature
selection before the split, so that the pipeline could "remember" random
patterns, therefore performing above chance.

**How many permutations should I run?**

Use 100 for a quick check. Use 1,000 or more for publication-level evidence.

**Why are Colab and local results different?**

Run the first notebook cell after pulling the latest code. Exact matching also
requires the same Leakly and dependency versions.

## License

MIT. See [LICENSE](LICENSE).
