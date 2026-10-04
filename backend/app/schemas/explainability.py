from typing import Dict, List, Optional
from pydantic import BaseModel

class FeatureImportance(BaseModel):
    feature_name: str
    importance_score: float

class ShapExplanationResult(BaseModel):
    top_features: List[FeatureImportance]
    explanation_type: str = "TreeExplainer"

class LLMExplanationResult(BaseModel):
    narrative: str
    is_fallback: bool