import pytest
import pandas as pd
import numpy as np
from app.schemas.dataset import TaskType
from app.schemas.pipeline import PipelineBlueprint, NumericImputerStrategy, NumericScalerStrategy, CategoricalImputerStrategy, CategoricalEncoderStrategy
from app.schemas.benchmark import BenchmarkResult, FoldMetric
from app.schemas.decision import DecisionWeights
from app.core.decision.ranker import DecisionEngine
from app.core.explainability.shap_explainer import ShapExplainer
from app.core.explainability.llm_explainer import LLMExplainer
from app.core.profiling.profiler import DatasetProfiler
from app.core.meta_features.extractor import MetaFeatureExtractor
from app.core.pipelines.builder import PipelineBuilder

@pytest.fixture
def mock_benchmark_results():
    return [
        BenchmarkResult(
            pipeline_id="pipe_baseline",
            pipeline_name="Standard Baseline",
            task_type="classification",
            mean_primary_metric=0.82,
            std_primary_metric=0.02,
            metrics_summary={"primary": 0.82, "mean_accuracy": 0.83},
            mean_fit_time_seconds=0.10,
            peak_memory_mb=12.0,
            folds=[]
        ),
        BenchmarkResult(
            pipeline_id="pipe_robust",
            pipeline_name="Robust Median Scaling",
            task_type="classification",
            mean_primary_metric=0.91,
            std_primary_metric=0.01,
            metrics_summary={"primary": 0.91, "mean_accuracy": 0.90},
            mean_fit_time_seconds=0.15,
            peak_memory_mb=14.0,
            folds=[]
        )
    ]

def test_decision_engine_ranks_properly(mock_benchmark_results):
    res = DecisionEngine.rank_candidates(mock_benchmark_results)

    assert res.winning_pipeline_id == "pipe_robust"
    assert len(res.leaderboard) == 2
    assert res.leaderboard[0].rank == 1
    assert res.leaderboard[1].rank == 2
    assert res.leaderboard[0].utility_score > res.leaderboard[1].utility_score
    assert "Selected 'Robust Median Scaling'" in res.selection_reason

def test_decision_engine_respects_extreme_weights(mock_benchmark_results):
    # If runtime is weighted 100% and performance 0%, the faster baseline should win
    weights = DecisionWeights(weight_performance=0.0, weight_runtime=1.0)
    res = DecisionEngine.rank_candidates(mock_benchmark_results, weights=weights)

    assert res.winning_pipeline_id == "pipe_baseline"
    assert res.leaderboard[0].pipeline_id == "pipe_baseline"

def test_shap_explainer_extracts_feature_importance(synthetic_clean_classification):
    df = synthetic_clean_classification
    num_cols = ["num_1", "num_2"]
    cat_cols = ["cat_1"]

    blueprint = PipelineBlueprint(
        id="test_pipe",
        name="Test Pipeline",
        description="",
        num_imputer=NumericImputerStrategy.MEDIAN,
        num_scaler=NumericScalerStrategy.STANDARD,
        cat_imputer=CategoricalImputerStrategy.MOST_FREQUENT,
        cat_encoder=CategoricalEncoderStrategy.ONEHOT
    )

    pipe = PipelineBuilder.build_pipeline(
        blueprint=blueprint,
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        task_type=TaskType.CLASSIFICATION
    )

    X = df.drop(columns=["target"])
    y = df["target"]
    pipe.fit(X, y)

    shap_res = ShapExplainer.explain_winning_pipeline(
        fitted_pipeline=pipe,
        X_sample=X,
        top_k=5
    )

    assert len(shap_res.top_features) > 0
    assert shap_res.top_features[0].importance_score >= 0.0
    assert shap_res.explanation_type == "TreeExplainer"

def test_llm_explainer_deterministic_fallback(mock_benchmark_results, synthetic_dirty_regression):
    profile = DatasetProfiler.profile(synthetic_dirty_regression, "target", TaskType.REGRESSION)
    meta = MetaFeatureExtractor.extract(profile)
    decision = DecisionEngine.rank_candidates(mock_benchmark_results)

    explanation = LLMExplainer.generate_explanation(meta, decision, task_type="regression")

    assert explanation.is_fallback is True
    assert "DataLens evaluated" in explanation.narrative
    assert "Robust Median Scaling" in explanation.narrative
    assert f"{decision.leaderboard[0].utility_score:.4f}" in explanation.narrative