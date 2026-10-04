import os
import uuid
import pandas as pd
from typing import Dict, Optional, Any
from app.schemas.dataset import TaskType
from app.schemas.profile import DatasetProfile, MetaFeatures
from app.schemas.pipeline import PipelineBlueprint
from app.schemas.benchmark import BenchmarkResult
from app.schemas.decision import DecisionResult, DecisionWeights
from app.schemas.explainability import ShapExplanationResult, LLMExplanationResult
from app.core.validation.dataset_validator import DatasetValidator
from app.core.profiling.profiler import DatasetProfiler
from app.core.meta_features.extractor import MetaFeatureExtractor
from app.core.rules.candidate_rules import RuleEngine
from app.core.pipelines.builder import PipelineBuilder
from app.core.benchmarking.cross_validator import BenchmarkEngine
from app.core.decision.ranker import DecisionEngine
from app.core.explainability.shap_explainer import ShapExplainer
from app.core.explainability.llm_explainer import LLMExplainer
from app.core.exporter.code_generator import CodeGenerator

class AnalysisState:
    UPLOADED = "UPLOADED"
    PROFILING = "PROFILING"
    BENCHMARKING = "BENCHMARKING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class AnalysisRecord:
    def __init__(
        self,
        analysis_id: str,
        df: pd.DataFrame,
        target_col: str,
        task_type: TaskType,
        filename: str
    ):
        self.analysis_id = analysis_id
        self.df = df
        self.target_col = target_col
        self.task_type = task_type
        self.filename = filename
        self.status = AnalysisState.UPLOADED
        self.progress_percent = 10
        self.error_message: Optional[str] = None
        
        # Results
        self.profile: Optional[DatasetProfile] = None
        self.meta_features: Optional[MetaFeatures] = None
        self.candidates: list[PipelineBlueprint] = []
        self.benchmarks: list[BenchmarkResult] = []
        self.decision: Optional[DecisionResult] = None
        self.shap_result: Optional[ShapExplanationResult] = None
        self.llm_explanation: Optional[LLMExplanationResult] = None

class AnalysisService:
    _registry: Dict[str, AnalysisRecord] = {}

    @classmethod
    def create_analysis(
        cls,
        df: pd.DataFrame,
        target_col: str,
        task_type: TaskType,
        filename: str
    ) -> AnalysisRecord:
        val_res = DatasetValidator.validate(df, target_col, task_type)
        if not val_res.is_valid:
            raise ValueError(f"Dataset validation failed: {'; '.join(val_res.errors)}")

        analysis_id = str(uuid.uuid4())
        record = AnalysisRecord(
            analysis_id=analysis_id,
            df=df,
            target_col=target_col,
            task_type=task_type,
            filename=filename
        )
        cls._registry[analysis_id] = record
        return record

    @classmethod
    def get_record(cls, analysis_id: str) -> Optional[AnalysisRecord]:
        return cls._registry.get(analysis_id)

    @classmethod
    def run_pipeline_sync(cls, analysis_id: str, weights: DecisionWeights = DecisionWeights()):
        record = cls.get_record(analysis_id)
        if not record:
            return

        try:
            # 1. Profiling & Meta-Features
            record.status = AnalysisState.PROFILING
            record.progress_percent = 25
            record.profile = DatasetProfiler.profile(record.df, record.target_col, record.task_type)
            record.meta_features = MetaFeatureExtractor.extract(record.profile)

            # 2. Rule Candidate Generation
            record.candidates = RuleEngine.generate_candidates(record.meta_features)

            # 3. Controlled Cross-Validation Benchmarking
            record.status = AnalysisState.BENCHMARKING
            record.progress_percent = 50
            
            numeric_cols = [c.name for c in record.profile.columns.values() if c.is_numeric]
            categorical_cols = [c.name for c in record.profile.columns.values() if not c.is_numeric]

            benchmarks = []
            for candidate in record.candidates:
                bench = BenchmarkEngine.evaluate_pipeline(
                    blueprint=candidate,
                    df=record.df,
                    target_col=record.target_col,
                    numeric_cols=numeric_cols,
                    categorical_cols=categorical_cols,
                    task_type=record.task_type,
                    n_splits=5
                )
                benchmarks.append(bench)
            record.benchmarks = benchmarks
            record.progress_percent = 80

            # 4. Multi-Objective Decision Engine
            record.decision = DecisionEngine.rank_candidates(record.benchmarks, weights=weights)

            # 5. Dual Explainability
            winner_blueprint = next(
                c for c in record.candidates if c.id == record.decision.winning_pipeline_id
            )
            winning_pipe = PipelineBuilder.build_pipeline(
                blueprint=winner_blueprint,
                numeric_cols=numeric_cols,
                categorical_cols=categorical_cols,
                task_type=record.task_type
            )
            X = record.df.drop(columns=[record.target_col])
            y = record.df[record.target_col]
            winning_pipe.fit(X, y)

            record.shap_result = ShapExplainer.explain_winning_pipeline(winning_pipe, X, top_k=8)
            record.llm_explanation = LLMExplainer.generate_explanation(
                meta=record.meta_features,
                decision=record.decision,
                task_type=record.task_type.value
            )

            record.status = AnalysisState.COMPLETED
            record.progress_percent = 100

        except Exception as e:
            record.status = AnalysisState.FAILED
            record.error_message = str(e)

    @classmethod
    def export_python_script(cls, analysis_id: str) -> str:
        record = cls.get_record(analysis_id)
        if not record or not record.decision:
            raise ValueError("Analysis not found or benchmark not completed.")

        winner = next(c for c in record.candidates if c.id == record.decision.winning_pipeline_id)
        numeric_cols = [c.name for c in record.profile.columns.values() if c.is_numeric]
        categorical_cols = [c.name for c in record.profile.columns.values() if not c.is_numeric]

        return CodeGenerator.generate_pipeline_script(
            blueprint=winner,
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            target_col=record.target_col,
            task_type=record.task_type
        )