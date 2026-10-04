from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class CandidateScore(BaseModel):
    pipeline_id: str
    pipeline_name: str
    rank: int
    utility_score: float
    raw_primary_metric: float
    raw_fit_time_seconds: float
    normalized_primary_metric: float
    normalized_fit_time: float

class DecisionWeights(BaseModel):
    weight_performance: float = Field(default=0.80, ge=0.0, le=1.0)
    weight_runtime: float = Field(default=0.20, ge=0.0, le=1.0)

class DecisionResult(BaseModel):
    winning_pipeline_id: str
    winning_pipeline_name: str
    weights_applied: DecisionWeights
    leaderboard: List[CandidateScore]
    selection_reason: str