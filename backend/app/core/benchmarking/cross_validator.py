import time
import tracemalloc
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
from app.schemas.dataset import TaskType
from app.schemas.pipeline import PipelineBlueprint
from app.schemas.benchmark import BenchmarkResult, FoldMetric
from app.core.pipelines.builder import PipelineBuilder

class BenchmarkEngine:
    @classmethod
    def evaluate_pipeline(
        cls,
        blueprint: PipelineBlueprint,
        df: pd.DataFrame,
        target_col: str,
        numeric_cols: List[str],
        categorical_cols: List[str],
        task_type: TaskType,
        n_splits: int = 5,
        random_state: int = 42
    ) -> BenchmarkResult:
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if task_type == TaskType.CLASSIFICATION:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_metrics: List[FoldMetric] = []
        fit_times: List[float] = []
        peak_mems: List[float] = []

        fold_idx = 1
        for train_idx, val_idx in cv.split(X, y):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline = PipelineBuilder.build_pipeline(
                blueprint=blueprint,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                task_type=task_type,
                random_state=random_state
            )

            tracemalloc.start()
            t0 = time.perf_counter()

            pipeline.fit(X_train, y_train)

            fit_time = time.perf_counter() - t0
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            fit_times.append(fit_time)
            peak_mems.append(peak_mem / (1024 * 1024))

            y_pred = pipeline.predict(X_val)
            secondary: Dict[str, float] = {}

            if task_type == TaskType.CLASSIFICATION:
                primary = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
                secondary["accuracy"] = float(accuracy_score(y_val, y_pred))

                try:
                    if hasattr(pipeline.named_steps["model"], "predict_proba"):
                        y_proba = pipeline.predict_proba(X_val)
                        if len(np.unique(y)) == 2:
                            secondary["roc_auc"] = float(roc_auc_score(y_val, y_proba[:, 1]))
                        else:
                            secondary["roc_auc"] = float(roc_auc_score(y_val, y_proba, multi_class="ovr"))
                except Exception:
                    secondary["roc_auc"] = 0.0

            else:
                primary = float(r2_score(y_val, y_pred))
                secondary["mae"] = float(mean_absolute_error(y_val, y_pred))
                secondary["rmse"] = float(root_mean_squared_error(y_val, y_pred))

            fold_metrics.append(FoldMetric(
                fold=fold_idx,
                primary_metric=round(primary, 4),
                secondary_metrics={k: round(v, 4) for k, v in secondary.items()},
                fit_time_seconds=round(fit_time, 4)
            ))
            fold_idx += 1

        primaries = [f.primary_metric for f in fold_metrics]
        mean_primary = float(np.mean(primaries))
        std_primary = float(np.std(primaries))

        metrics_summary: Dict[str, float] = {
            "primary": round(mean_primary, 4),
            "primary_std": round(std_primary, 4)
        }

        all_sec_keys = fold_metrics[0].secondary_metrics.keys()
        for k in all_sec_keys:
            vals = [f.secondary_metrics[k] for f in fold_metrics]
            metrics_summary[f"mean_{k}"] = round(float(np.mean(vals)), 4)

        return BenchmarkResult(
            pipeline_id=blueprint.id,
            pipeline_name=blueprint.name,
            task_type=task_type.value,
            mean_primary_metric=round(mean_primary, 4),
            std_primary_metric=round(std_primary, 4),
            metrics_summary=metrics_summary,
            mean_fit_time_seconds=round(float(np.mean(fit_times)), 4),
            peak_memory_mb=round(float(np.max(peak_mems)), 2),
            folds=fold_metrics
        )