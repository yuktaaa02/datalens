from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ColumnProfile(BaseModel):
    name: str
    dtype: str
    is_numeric: bool
    missing_count: int
    missing_ratio: float
    unique_count: int
    cardinality_ratio: float
    # Numeric stats
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    skewness: Optional[float] = None
    outlier_count: Optional[int] = None
    outlier_ratio: Optional[float] = None
    # Categorical stats
    top_categories: Optional[Dict[str, int]] = None

class TargetProfile(BaseModel):
    name: str
    task_type: str
    missing_count: int
    # Classification
    class_counts: Optional[Dict[str, int]] = None
    imbalance_ratio: Optional[float] = None  # min_class / max_class
    # Regression
    mean: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None

class DatasetProfile(BaseModel):
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    overall_missing_ratio: float
    duplicate_rows_count: int
    columns: Dict[str, ColumnProfile]
    target: TargetProfile

class MetaFeatures(BaseModel):
    n_rows: int
    n_features: int
    num_cat_ratio: float          # numeric_features / total_features
    missing_cell_ratio: float     # null_cells / total_cells
    outlier_cell_ratio: float     # outlier_cells / total_numeric_cells
    mean_skewness: float          # avg absolute skewness across numeric features
    max_cardinality: int          # max unique values among categorical features
    class_imbalance_ratio: float  # 1.0 for regression or balanced classification