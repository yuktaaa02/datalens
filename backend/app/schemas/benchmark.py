from typing import Dict, List
from pydantic import BaseModel

class FoldMetric(BaseModel):
    fold: int
    primary_metric: float
    secondary_metrics: Dict[str, float]
    fit_time_seconds: float

class BenchmarkResult(BaseModel):
    pipeline_id: str
    pipeline_name: str
    task_type: str
    mean_primary_metric: float
    std_primary_metric: float
    metrics_summary: Dict[str, float]
    mean_fit_time_seconds: float
    peak_memory_mb: float
    folds: List[FoldMetric]