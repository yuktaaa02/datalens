import pandas as pd
import numpy as np
from app.schemas.dataset import TaskType, DatasetValidationResult

class DatasetValidator:
    MIN_ROWS_REQUIRED: int = 25

    @classmethod
    def validate(
        cls, 
        df: pd.DataFrame, 
        target_column: str, 
        task_type: TaskType
    ) -> DatasetValidationResult:
        errors = []
        warnings = []

        if df.empty:
            return DatasetValidationResult(
                is_valid=False,
                row_count=0,
                column_count=0,
                target_column=target_column,
                task_type=task_type,
                errors=["Dataset is empty."]
            )

        row_count, col_count = df.shape

        if row_count < cls.MIN_ROWS_REQUIRED:
            errors.append(
                f"Dataset must have at least {cls.MIN_ROWS_REQUIRED} rows for 5-fold CV; found {row_count}."
            )

        if target_column not in df.columns:
            errors.append(f"Target column '{target_column}' not found in dataset columns.")
            return DatasetValidationResult(
                is_valid=False,
                row_count=row_count,
                column_count=col_count,
                target_column=target_column,
                task_type=task_type,
                errors=errors
            )

        # Check target integrity
        target_series = df[target_column]
        if target_series.isnull().all():
            errors.append(f"Target column '{target_column}' contains only null values.")

        if task_type == TaskType.CLASSIFICATION:
            unique_classes = target_series.dropna().nunique()
            if unique_classes < 2:
                errors.append(f"Classification target must have >= 2 unique classes; found {unique_classes}.")
            # Check minimum instances per class for stratified splitting
            class_counts = target_series.value_counts()
            if (class_counts < 5).any():
                warnings.append("Some classes have fewer than 5 samples; Stratified 5-Fold CV may require adjustment.")

        elif task_type == TaskType.REGRESSION:
            if not np.issubdtype(target_series.dtype, np.number):
                errors.append(f"Regression target '{target_column}' must be numerical.")

        # Check features
        all_null_cols = [col for col in df.columns if df[col].isnull().all()]
        if all_null_cols:
            warnings.append(f"Columns with 100% missing values detected: {all_null_cols}")

        if df.columns.duplicated().any():
            errors.append("Duplicate column names detected.")

        return DatasetValidationResult(
            is_valid=len(errors) == 0,
            row_count=row_count,
            column_count=col_count,
            target_column=target_column,
            task_type=task_type,
            warnings=warnings,
            errors=errors
        )