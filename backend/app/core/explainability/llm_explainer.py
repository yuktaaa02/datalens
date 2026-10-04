import os
from app.schemas.profile import MetaFeatures
from app.schemas.decision import DecisionResult
from app.schemas.explainability import LLMExplanationResult

class LLMExplainer:
    SYSTEM_PROMPT = (
        "You are an expert Data Science and Machine Learning engineer. "
        "Explain the preprocessing benchmark recommendation accurately using ONLY the provided structured JSON data. "
        "Do NOT invent metrics or make claims outside the provided facts. "
        "Highlight the winning pipeline, how it handles dataset traits, and its metric advantage."
    )

    @classmethod
    def generate_explanation(
        cls,
        meta: MetaFeatures,
        decision: DecisionResult,
        task_type: str
    ) -> LLMExplanationResult:
        winner = decision.leaderboard[0]
        runner_up = decision.leaderboard[1] if len(decision.leaderboard) > 1 else None

        # Check for API key (OpenAI or similar)
        api_key = os.getenv("OPENAI_API_KEY")

        if api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)

                prompt = (
                    f"Dataset Meta-Features: Rows={meta.n_rows}, Features={meta.n_features}, "
                    f"MissingCells={meta.missing_cell_ratio*100:.1f}%, Outliers={meta.outlier_cell_ratio*100:.1f}%, "
                    f"MeanSkewness={meta.mean_skewness:.2f}, MaxCardinality={meta.max_cardinality}.\n"
                    f"Task Type: {task_type}.\n"
                    f"Recommended Pipeline: {winner.pipeline_name} (Rank 1, Utility={winner.utility_score:.4f}, "
                    f"PrimaryMetric={winner.raw_primary_metric:.4f}, Runtime={winner.raw_fit_time_seconds:.3f}s).\n"
                )
                if runner_up:
                    prompt += (
                        f"Runner-up Pipeline: {runner_up.pipeline_name} (Rank 2, Utility={runner_up.utility_score:.4f}, "
                        f"PrimaryMetric={runner_up.raw_primary_metric:.4f}, Runtime={runner_up.raw_fit_time_seconds:.3f}s).\n"
                    )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": cls.SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=250,
                    temperature=0.2
                )
                narrative = response.choices[0].message.content.strip()
                return LLMExplanationResult(narrative=narrative, is_fallback=False)
            except Exception:
                pass  # Fall through to deterministic generator on any API error

        # Deterministic Rule-Based Fallback
        traits = []
        if meta.missing_cell_ratio > 0.0:
            traits.append(f"{meta.missing_cell_ratio*100:.1f}% missing values")
        if meta.outlier_cell_ratio > 0.05:
            traits.append(f"{meta.outlier_cell_ratio*100:.1f}% outlier incidence")
        if meta.mean_skewness > 1.0:
            traits.append(f"significant positive skewness ({meta.mean_skewness:.2f})")

        traits_str = ", ".join(traits) if traits else "standard continuous/categorical distributions"

        narrative = (
            f"DataLens evaluated {len(decision.leaderboard)} candidate preprocessing pipelines under controlled 5-fold cross-validation. "
            f"The dataset exhibited {traits_str}. "
            f"'{winner.pipeline_name}' was selected as the optimal pipeline with an overall utility score of {winner.utility_score:.4f}. "
            f"It attained a primary evaluation metric of {winner.raw_primary_metric:.4f} with a total fit time of {winner.raw_fit_time_seconds:.3f}s."
        )

        if runner_up:
            metric_delta = winner.raw_primary_metric - runner_up.raw_primary_metric
            narrative += (
                f" In comparison, runner-up '{runner_up.pipeline_name}' achieved a primary metric of {runner_up.raw_primary_metric:.4f} "
                f"({metric_delta:+.4f} difference). "
                f"The benchmark demonstrates that the preprocessing choices in '{winner.pipeline_name}' provided superior balance between "
                f"predictive accuracy and computational efficiency."
            )

        return LLMExplanationResult(narrative=narrative, is_fallback=True)