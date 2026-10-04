import pytest
import numpy as np
import pandas as pd
from app.schemas.dataset import TaskType
from app.core.profiling.profiler import DatasetProfiler
from app.core.meta_features.extractor import MetaFeatureExtractor
from app.core.rules.candidate_rules import RuleEngine
from app.core.pipelines.builder import PipelineBuilder
from app.core.benchmarking.cross_validator import BenchmarkEngine

def test_rule_engine_candidate_generation(synthetic_dirty_regression):
    profile = DatasetProfiler.profile(synthetic_dirty_regression, "target", TaskType.REGRESSION)
    meta = MetaFeatureExtractor.extract(profile)
    candidates = RuleEngine.generate_candidates(meta)

    assert len(candidates) >= 2
    # Verify candidate with robust scaling was generated due to outliers
    has_robust = any("Robust" in c.name for c in candidates)
    assert has_robust

def test_pipeline_builder_construction(synthetic_clean_classification):
    profile = DatasetProfiler.profile(synthetic_clean_classification, "target", TaskType.CLASSIFICATION)
    meta = MetaFeatureExtractor.extract(profile)
    candidates = RuleEngine.generate_candidates(meta)

    num_cols = ["num_1", "num_2"]
    cat_cols = ["cat_1"]

    pipeline = PipelineBuilder.build_pipeline(
        blueprint=candidates[0],
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        task_type=TaskType.CLASSIFICATION
    )

    assert hasattr(pipeline, "fit")
    assert hasattr(pipeline, "predict")
    assert "preprocessor" in pipeline.named_steps
    assert "model" in pipeline.named_steps

def test_benchmark_engine_cv_evaluation(synthetic_clean_classification):
    profile = DatasetProfiler.profile(synthetic_clean_classification, "target", TaskType.CLASSIFICATION)
    meta = MetaFeatureExtractor.extract(profile)
    candidates = RuleEngine.generate_candidates(meta)

    num_cols = ["num_1", "num_2"]
    cat_cols = ["cat_1"]

    res = BenchmarkEngine.evaluate_pipeline(
        blueprint=candidates[0],
        df=synthetic_clean_classification,
        target_col="target",
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        task_type=TaskType.CLASSIFICATION,
        n_splits=5
    )

    assert res.pipeline_id == candidates[0].id
    assert len(res.folds) == 5
    assert 0.0 <= res.mean_primary_metric <= 1.0
    assert res.mean_fit_time_seconds > 0.0
    assert "mean_accuracy" in res.metrics_summary

def test_pipeline_no_data_leakage(synthetic_dirty_regression):
    """
    Verify that transformers inside the pipeline do not leak information:
    Parameters fitted during CV are restricted to the training subset.
    """
    num_cols = ["outlier_feat", "missing_feat"]
    cat_cols = ["cat_high_card"]

    profile = DatasetProfiler.profile(synthetic_dirty_regression, "target", TaskType.REGRESSION)
    meta = MetaFeatureExtractor.extract(profile)
    candidates = RuleEngine.generate_candidates(meta)

    # Candidate 1: Mean Imputation
    pipe = PipelineBuilder.build_pipeline(
        blueprint=candidates[0],
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        task_type=TaskType.REGRESSION
    )

    # Train on first 80 rows, validate on remaining 20
    train_df = synthetic_dirty_regression.iloc[:80]
    val_df = synthetic_dirty_regression.iloc[80:]

    pipe.fit(train_df.drop(columns=["target"]), train_df["target"])

    # Extract fitted mean imputer from inside the ColumnTransformer
    preprocessor = pipe.named_steps["preprocessor"]
    fitted_imputer = preprocessor.named_transformers_["numeric"].named_steps["num_imputer"]
    
    # Calculate expected mean solely on train_df
    expected_means = train_df[num_cols].mean().values

    # Assert the fitted imputer statistics match the training split and not the full dataset
    np.testing.assert_allclose(fitted_imputer.statistics_, expected_means, rtol=1e-4)