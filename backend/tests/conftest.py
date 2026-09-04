import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def synthetic_clean_classification():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "num_1": np.random.normal(loc=10, scale=2, size=n),
        "num_2": np.random.uniform(low=0, high=1, size=n),
        "cat_1": np.random.choice(["A", "B", "C"], size=n),
        "target": np.random.choice([0, 1], size=n, p=[0.5, 0.5])
    })
    return df

@pytest.fixture
def synthetic_dirty_regression():
    np.random.seed(42)
    n = 100
    num_with_outliers = np.random.normal(loc=0, scale=1, size=n)
    num_with_outliers[0] = 50.0  # Explicit outlier
    num_with_outliers[1] = -45.0 # Explicit outlier

    num_with_missing = np.random.exponential(scale=2.0, size=n)
    num_with_missing[:15] = np.nan # 15% missing

    df = pd.DataFrame({
        "outlier_feat": num_with_outliers,
        "missing_feat": num_with_missing,
        "cat_high_card": [f"id_{i % 20}" for i in range(n)],
        "target": np.random.normal(loc=100, scale=15, size=n)
    })
    return df