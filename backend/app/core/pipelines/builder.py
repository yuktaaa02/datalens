from typing import List
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    PowerTransformer,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from app.schemas.dataset import TaskType
from app.schemas.pipeline import (
    PipelineBlueprint,
    NumericImputerStrategy,
    NumericScalerStrategy,
    CategoricalImputerStrategy,
    CategoricalEncoderStrategy
)

class PipelineBuilder:
    @staticmethod
    def build_pipeline(
        blueprint: PipelineBlueprint,
        numeric_cols: List[str],
        categorical_cols: List[str],
        task_type: TaskType,
        random_state: int = 42
    ) -> Pipeline:
        transformers = []

        # 1. Numeric Branch
        if numeric_cols:
            num_steps = []
            if blueprint.num_imputer == NumericImputerStrategy.MEAN:
                num_steps.append(("num_imputer", SimpleImputer(strategy="mean")))
            elif blueprint.num_imputer == NumericImputerStrategy.MEDIAN:
                num_steps.append(("num_imputer", SimpleImputer(strategy="median")))
            elif blueprint.num_imputer == NumericImputerStrategy.KNN:
                num_steps.append(("num_imputer", KNNImputer(n_neighbors=5)))

            if blueprint.num_scaler == NumericScalerStrategy.STANDARD:
                num_steps.append(("num_scaler", StandardScaler()))
            elif blueprint.num_scaler == NumericScalerStrategy.ROBUST:
                num_steps.append(("num_scaler", RobustScaler()))
            elif blueprint.num_scaler == NumericScalerStrategy.MINMAX:
                num_steps.append(("num_scaler", MinMaxScaler()))
            elif blueprint.num_scaler == NumericScalerStrategy.POWER:
                num_steps.append(("num_scaler", PowerTransformer(method="yeo-johnson")))

            if num_steps:
                transformers.append(("numeric", Pipeline(steps=num_steps), numeric_cols))
            else:
                transformers.append(("numeric", "passthrough", numeric_cols))

        # 2. Categorical Branch
        if categorical_cols:
            cat_steps = []
            if blueprint.cat_imputer == CategoricalImputerStrategy.MOST_FREQUENT:
                cat_steps.append(("cat_imputer", SimpleImputer(strategy="most_frequent")))
            elif blueprint.cat_imputer == CategoricalImputerStrategy.CONSTANT:
                cat_steps.append(("cat_imputer", SimpleImputer(strategy="constant", fill_value="missing")))

            if blueprint.cat_encoder == CategoricalEncoderStrategy.ONEHOT:
                cat_steps.append(("cat_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
            elif blueprint.cat_encoder == CategoricalEncoderStrategy.ORDINAL:
                cat_steps.append(("cat_encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )))

            if cat_steps:
                transformers.append(("categorical", Pipeline(steps=cat_steps), categorical_cols))
            else:
                transformers.append(("categorical", "passthrough", categorical_cols))

        # 3. ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop"
        )

        # 4. Fixed Baseline Estimator
        if task_type == TaskType.CLASSIFICATION:
            estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=random_state)

        return Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])