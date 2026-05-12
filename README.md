# Leakly: Leakage check in machine-learning pipelines
---

Leakly helps test whether a machine-learning pipeline may be leaking target
information across train/test boundaries. It does this by rerunning a pipeline
with randomly permuted labels and inspecting the held-out test scores.

## Installation

Leakly uses the standard scientific Python stack:

```bash
pip install numpy pandas scipy scikit-learn pyyaml matplotlib tqdm
```

## Minimal Example

```python
from leakly import (
    ExampleMLPipeline,
    LeakageCheckerOneRun,
    MLConfig,
    PipelineConfig,
    SimulationConfig,
    SplitConfig,
    simulate_dataset,
)

simulated = simulate_dataset(
    SimulationConfig(
        n_samples=100,
        n_features=20,
        n_covariates=2,
        random_state=0,
    )
)

config = PipelineConfig(
    split=SplitConfig(test_fraction=0.25, random_state=0),
    ml=MLConfig(random_state=0),
)

pipeline = ExampleMLPipeline(
    simulated.X,
    simulated.y,
    covariates=simulated.covariates,
    config=config,
    feature_names=simulated.feature_names,
)

observed_auc = pipeline.run()

checker = LeakageCheckerOneRun(
    pipeline,
    perc_permutation=1.0,
    random_state=1,
)
permuted_auc = checker.run()
```
