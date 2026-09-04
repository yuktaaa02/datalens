import pytest
from app.schemas.dataset import TaskType
from app.core.validation.dataset_validator import DatasetValidator
from app.core.profiling.profiler import DatasetProfiler
from app.core.meta_features.extractor import MetaFeatureExtractor

def test_validator_rejects_empty_dataset():
    import pandas as pd
    res = DatasetValidator.validate(pd.DataFrame(), "target", TaskType.CLASSIFICATION)
    assert not res.is_valid
    assert "Dataset is empty." in res.errors

def test_validator_detects_missing_target(synthetic_clean_classification):
    res = DatasetValidator.validate(synthetic_clean_classification, "non_existent_col", TaskType.CLASSIFICATION)
    assert not res.is_valid
    assert any("not found" in err for err in res.errors)

def test_validator_accepts_valid_classification(synthetic_clean_classification):
    res = DatasetValidator.validate(synthetic_clean_classification, "target", TaskType.CLASSIFICATION)
    assert res.is_valid
    assert len(res.errors) == 0

def test_profiler_and_meta_extraction(synthetic_dirty_regression):
    # 1. Profile
    profile = DatasetProfiler.profile(
        df=synthetic_dirty_regression,
        target_col="target",
        task_type=TaskType.REGRESSION
    )
    assert profile.total_rows == 100
    assert profile.total_columns == 4
    assert profile.columns["outlier_feat"].outlier_count >= 2
    assert profile.columns["missing_feat"].missing_count == 15

    # 2. Extract Meta-Features
    meta = MetaFeatureExtractor.extract(profile)
    assert meta.n_rows == 100
    assert meta.n_features == 3
    assert meta.missing_cell_ratio > 0.0
    assert meta.max_cardinality == 20
    assert meta.class_imbalance_ratio == 1.0  # Fixed for regression