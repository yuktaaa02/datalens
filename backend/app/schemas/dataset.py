from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

class DatasetValidationResult(BaseModel):
    is_valid: bool
    row_count: int
    column_count: int
    target_column: str
    task_type: TaskType
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)