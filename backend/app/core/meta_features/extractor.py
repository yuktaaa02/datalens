import numpy as np
from app.schemas.profile import DatasetProfile, MetaFeatures

class MetaFeatureExtractor:
    @staticmethod
    def extract(profile: DatasetProfile) -> MetaFeatures:
        feature_cols = list(profile.columns.values())
        n_features = len(feature_cols)

        if n_features == 0:
            return MetaFeatures(
                n_rows=profile.total_rows,
                n_features=0,
                num_cat_ratio=0.0,
                missing_cell_ratio=profile.overall_missing_ratio,
                outlier_cell_ratio=0.0,
                mean_skewness=0.0,
                max_cardinality=0,
                class_imbalance_ratio=1.0
            )

        numeric_cols = [c for c in feature_cols if c.is_numeric]
        categorical_cols = [c for c in feature_cols if not c.is_numeric]

        num_cat_ratio = round(len(numeric_cols) / n_features, 4)

        # Outlier calculation across numeric features
        total_num_cells = sum(c.unique_count for c in numeric_cols)
        total_outliers = sum(c.outlier_count or 0 for c in numeric_cols)
        outlier_cell_ratio = round(total_outliers / max(1, total_num_cells), 4)

        # Skewness aggregation (absolute mean of non-null skew values)
        skews = [abs(c.skewness) for c in numeric_cols if c.skewness is not None]
        mean_skewness = round(float(np.mean(skews)), 4) if skews else 0.0

        # Maximum cardinality in categoricals
        max_cardinality = max([c.unique_count for c in categorical_cols], default=0)

        # Class imbalance ratio (1.0 for regression)
        imbalance = profile.target.imbalance_ratio if profile.target.imbalance_ratio is not None else 1.0

        return MetaFeatures(
            n_rows=profile.total_rows,
            n_features=n_features,
            num_cat_ratio=num_cat_ratio,
            missing_cell_ratio=profile.overall_missing_ratio,
            outlier_cell_ratio=outlier_cell_ratio,
            mean_skewness=mean_skewness,
            max_cardinality=max_cardinality,
            class_imbalance_ratio=imbalance
        )