import numpy as np
import pandas as pd
from scipy import stats
from app.schemas.dataset import TaskType
from app.schemas.profile import ColumnProfile, TargetProfile, DatasetProfile

class DatasetProfiler:
    @staticmethod
    def profile(df: pd.DataFrame, target_col: str, task_type: TaskType) -> DatasetProfile:
        total_rows, total_cols = df.shape
        mem_usage = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 3)
        total_cells = total_rows * total_cols
        total_missing = int(df.isnull().sum().sum())
        overall_missing_ratio = round(total_missing / total_cells, 4) if total_cells > 0 else 0.0
        duplicates = int(df.duplicated().sum())

        feature_cols = [col for col in df.columns if col != target_col]
        columns_profile: dict[str, ColumnProfile] = {}

        for col in feature_cols:
            series = df[col]
            import pandas as pd  # ensure pandas is imported

            is_num = bool(pd.api.types.is_numeric_dtype(series))
            n_missing = int(series.isnull().sum())
            missing_ratio = round(n_missing / total_rows, 4)
            n_unique = int(series.nunique(dropna=True))
            cardinality_ratio = round(n_unique / total_rows, 4)

            col_prof = ColumnProfile(
                name=col,
                dtype=str(series.dtype),
                is_numeric=is_num,
                missing_count=n_missing,
                missing_ratio=missing_ratio,
                unique_count=n_unique,
                cardinality_ratio=cardinality_ratio
            )

            if is_num:
                clean_series = series.dropna()
                if len(clean_series) > 0:
                    col_prof.mean = round(float(clean_series.mean()), 4)
                    col_prof.median = round(float(clean_series.median()), 4)
                    col_prof.std = round(float(clean_series.std()), 4) if len(clean_series) > 1 else 0.0
                    col_prof.min = round(float(clean_series.min()), 4)
                    col_prof.max = round(float(clean_series.max()), 4)
                    
                    # Skewness
                    col_prof.skewness = round(float(stats.skew(clean_series, bias=False)), 4) if len(clean_series) > 2 else 0.0
                    
                    # Outliers via IQR
                    q25, q75 = np.percentile(clean_series, [25, 75])
                    iqr = q75 - q25
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    outliers = clean_series[(clean_series < lower_bound) | (clean_series > upper_bound)]
                    col_prof.outlier_count = int(len(outliers))
                    col_prof.outlier_ratio = round(len(outliers) / len(clean_series), 4)
            else:
                top_cats = series.value_counts().head(5).to_dict()
                col_prof.top_categories = {str(k): int(v) for k, v in top_cats.items()}

            columns_profile[col] = col_prof

        # Target column profiling
        target_series = df[target_col]
        target_missing = int(target_series.isnull().sum())

        if task_type == TaskType.CLASSIFICATION:
            counts = target_series.value_counts()
            class_counts_dict = {str(k): int(v) for k, v in counts.items()}
            imb_ratio = round(float(counts.min() / counts.max()), 4) if len(counts) > 1 and counts.max() > 0 else 1.0
            target_profile = TargetProfile(
                name=target_col,
                task_type=task_type.value,
                missing_count=target_missing,
                class_counts=class_counts_dict,
                imbalance_ratio=imb_ratio
            )
        else:
            clean_target = target_series.dropna()
            target_profile = TargetProfile(
                name=target_col,
                task_type=task_type.value,
                missing_count=target_missing,
                mean=round(float(clean_target.mean()), 4) if len(clean_target) > 0 else 0.0,
                std=round(float(clean_target.std()), 4) if len(clean_target) > 1 else 0.0,
                skewness=round(float(stats.skew(clean_target, bias=False)), 4) if len(clean_target) > 2 else 0.0
            )

        return DatasetProfile(
            total_rows=total_rows,
            total_columns=total_cols,
            memory_usage_mb=mem_usage,
            overall_missing_ratio=overall_missing_ratio,
            duplicate_rows_count=duplicates,
            columns=columns_profile,
            target=target_profile
        )