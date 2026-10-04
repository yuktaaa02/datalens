from enum import Enum
from typing import List
from pydantic import BaseModel, Field

class NumericImputerStrategy(str, Enum):
    NONE = "none"
    MEAN = "mean"
    MEDIAN = "median"
    KNN = "knn"

class CategoricalImputerStrategy(str, Enum):
    NONE = "none"
    MOST_FREQUENT = "most_frequent"
    CONSTANT = "constant"

class NumericScalerStrategy(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    ROBUST = "robust"
    MINMAX = "minmax"
    POWER = "power"  # Yeo-Johnson

class CategoricalEncoderStrategy(str, Enum):
    NONE = "none"
    ONEHOT = "onehot"
    ORDINAL = "ordinal"

class PipelineBlueprint(BaseModel):
    id: str
    name: str
    description: str
    num_imputer: NumericImputerStrategy
    num_scaler: NumericScalerStrategy
    cat_imputer: CategoricalImputerStrategy
    cat_encoder: CategoricalEncoderStrategy
    activated_rules: List[str] = Field(default_factory=list)