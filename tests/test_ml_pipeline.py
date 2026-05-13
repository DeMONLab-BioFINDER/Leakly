import numpy as np
import pytest

from leakly import (
    FeatureSelectionConfig,
    MLPipeline,
    ModelConfig,
    PipelineConfig,
    SimulationConfig,
    SplitConfig,
    load_config_yaml,
    load_example_leakage_config,
    load_example_nonleakage_config,
    simulate_dataset,
)


def _small_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        data_split=SplitConfig(
            test_fraction=0.25,
            random_state=11,
            stratify=True,
        ),
        feature_selection=FeatureSelectionConfig(
            selected_feature_names=["feature_1", "feature_2"],
        ),
        model=ModelConfig(
            model="random_forest",
            random_state=11,
            model_params={"n_estimators": 10, "max_depth": 3},
        ),
    )


def test_ml_pipeline_fit_evaluate_predict_and_refit_idempotence():
    data = simulate_dataset(
        SimulationConfig(
            n_samples=80,
            n_features=8,
            n_covariates=2,
            random_state=11,
        )
    )
    pipeline = MLPipeline(
        data.X,
        data.y,
        covariates=data.covariates,
        config=_small_pipeline_config(),
    )

    pipeline.fit()
    first_fitted_steps = list(pipeline.fitted_steps)
    score = pipeline.evaluate()
    predictions = pipeline.predict(data.X[:5])

    assert 0.0 <= score <= 1.0
    assert predictions.shape == (5,)
    assert pipeline.selected_feature_names == ["feature_1", "feature_2"]
    assert pipeline.selected_feature_indices == [0, 1]

    pipeline.fit()

    assert len(pipeline.fitted_steps) == len(first_fitted_steps)
    assert pipeline.selected_feature_indices == [0, 1]
    assert 0.0 <= pipeline.evaluate() <= 1.0


def test_ml_pipeline_evaluates_user_supplied_predictions():
    pipeline = MLPipeline(
        [[0], [1], [2], [3]],
        [0, 1, 0, 1],
        config=_small_pipeline_config(),
    )
    pipeline.is_fitted = True
    pipeline.model = object()

    assert pipeline.evaluate(
        y_true=[0, 1, 1, 1],
        y_pred=[0, 1, 0, 1],
        metric="accuracy",
    ) == 0.75


def test_ml_pipeline_uses_pipeline_order_from_yaml_config_object():
    config = load_config_yaml("leakly/Example_LeakgePipeline.yaml")
    pipeline = MLPipeline(
        [[0], [1], [2], [3]],
        [0, 1, 0, 1],
        config=config,
    )

    assert pipeline.pipeline == [
        "imputation",
        "normalization",
        "feature_selection",
        "data_split",
        "model",
    ]


def test_ml_pipeline_uses_pipeline_order_from_mapping_config():
    leakage_pipeline = MLPipeline(
        [[0], [1], [2], [3]],
        [0, 1, 0, 1],
        config=load_example_leakage_config(),
    )
    no_leakage_pipeline = MLPipeline(
        [[0], [1], [2], [3]],
        [0, 1, 0, 1],
        config=load_example_nonleakage_config(),
    )

    assert leakage_pipeline.pipeline[:2] == ["imputation", "normalization"]
    assert no_leakage_pipeline.pipeline[:2] == ["data_split", "imputation"]


@pytest.mark.parametrize(
    "pipeline_steps, message",
    [
        (["imputation", "model"], "data_split"),
        (["data_split", "imputation"], "model"),
        (["data_split", "model", "normalization"], "final"),
        (["data_split", "unknown", "model"], "Unsupported"),
    ],
)
def test_ml_pipeline_rejects_invalid_pipeline_steps(pipeline_steps, message):
    with pytest.raises(ValueError, match=message):
        MLPipeline(
            np.zeros((4, 2)),
            np.array([0, 1, 0, 1]),
            config=PipelineConfig(pipeline=pipeline_steps),
        )
