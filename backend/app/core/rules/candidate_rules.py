from typing import List
from app.schemas.profile import MetaFeatures
from app.schemas.pipeline import (
    PipelineBlueprint,
    NumericImputerStrategy,
    NumericScalerStrategy,
    CategoricalImputerStrategy,
    CategoricalEncoderStrategy
)

class RuleEngine:
    MISSING_THRESHOLD_LOW = 0.05
    MISSING_THRESHOLD_HIGH = 0.20
    OUTLIER_THRESHOLD = 0.05
    SKEW_THRESHOLD = 1.0
    HIGH_CARDINALITY_THRESHOLD = 15

    @classmethod
    def generate_candidates(cls, meta: MetaFeatures) -> List[PipelineBlueprint]:
        candidates: List[PipelineBlueprint] = []

        has_num = meta.num_cat_ratio > 0.0
        has_cat = meta.num_cat_ratio < 1.0 and meta.n_features > 0
        has_missing = meta.missing_cell_ratio > 0.0
        has_outliers = meta.outlier_cell_ratio > cls.OUTLIER_THRESHOLD
        has_skew = meta.mean_skewness > cls.SKEW_THRESHOLD
        is_high_card = meta.max_cardinality > cls.HIGH_CARDINALITY_THRESHOLD

        default_cat_imp = CategoricalImputerStrategy.MOST_FREQUENT if (has_cat and has_missing) else (
            CategoricalImputerStrategy.NONE if not has_missing else CategoricalImputerStrategy.MOST_FREQUENT
        )
        default_cat_enc = (
            CategoricalEncoderStrategy.ORDINAL if is_high_card else CategoricalEncoderStrategy.ONEHOT
        ) if has_cat else CategoricalEncoderStrategy.NONE

        # Candidate 1: Standard Baseline
        c1_rules = ["baseline_strategy"]
        c1_num_imp = NumericImputerStrategy.MEAN if has_missing and has_num else NumericImputerStrategy.NONE
        c1_num_scale = NumericScalerStrategy.STANDARD if has_num else NumericScalerStrategy.NONE

        candidates.append(PipelineBlueprint(
            id="pipe_baseline",
            name="Standard Baseline",
            description="Mean Imputation + StandardScaler + OneHot/Ordinal Encoding",
            num_imputer=c1_num_imp,
            num_scaler=c1_num_scale,
            cat_imputer=default_cat_imp,
            cat_encoder=default_cat_enc,
            activated_rules=c1_rules
        ))

        # Candidate 2: Robust Median Scaling
        c2_rules = []
        c2_num_imp = NumericImputerStrategy.MEDIAN if has_missing and has_num else NumericImputerStrategy.NONE
        c2_num_scale = NumericScalerStrategy.STANDARD if has_num else NumericScalerStrategy.NONE

        if has_missing:
            c2_rules.append("median_imputation_rule")
        if has_outliers:
            c2_num_scale = NumericScalerStrategy.ROBUST
            c2_rules.append("outlier_robust_scaling_rule")

        candidates.append(PipelineBlueprint(
            id="pipe_robust",
            name="Robust Median Scaling",
            description="Median Imputation + RobustScaler (tolerant to outliers)",
            num_imputer=c2_num_imp,
            num_scaler=c2_num_scale,
            cat_imputer=default_cat_imp,
            cat_encoder=default_cat_enc,
            activated_rules=c2_rules or ["standard_median_fallback"]
        ))

        # Candidate 3: KNN Imputation (if missing values exist)
        if has_missing and has_num:
            candidates.append(PipelineBlueprint(
                id="pipe_knn_impute",
                name="KNN Imputation Pipeline",
                description="KNN Imputer (k=5) + MinMax Scaling",
                num_imputer=NumericImputerStrategy.KNN,
                num_scaler=NumericScalerStrategy.MINMAX,
                cat_imputer=default_cat_imp,
                cat_encoder=default_cat_enc,
                activated_rules=["knn_imputation_rule", "minmax_scaling_rule"]
            ))

        # Candidate 4: Power Transformation (if high skewness)
        if has_skew and has_num:
            candidates.append(PipelineBlueprint(
                id="pipe_power_transform",
                name="Power Transformation (Yeo-Johnson)",
                description="Median Imputation + Yeo-Johnson Power Transform for skewed distributions",
                num_imputer=NumericImputerStrategy.MEDIAN if has_missing else NumericImputerStrategy.NONE,
                num_scaler=NumericScalerStrategy.POWER,
                cat_imputer=default_cat_imp,
                cat_encoder=default_cat_enc,
                activated_rules=["high_skewness_power_transform_rule"]
            ))

        # Candidate 5: MinMax Scaler Alternative (if outliers are low)
        if not has_outliers and has_num and len(candidates) < 4:
            candidates.append(PipelineBlueprint(
                id="pipe_minmax",
                name="Bounded MinMax Scaling",
                description="Median Imputation + MinMaxScaler",
                num_imputer=NumericImputerStrategy.MEDIAN if has_missing else NumericImputerStrategy.NONE,
                num_scaler=NumericScalerStrategy.MINMAX,
                cat_imputer=default_cat_imp,
                cat_encoder=default_cat_enc,
                activated_rules=["low_outlier_minmax_rule"]
            ))

        return candidates