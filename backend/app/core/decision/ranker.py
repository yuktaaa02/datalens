import numpy as np
from typing import List
from app.schemas.benchmark import BenchmarkResult
from app.schemas.decision import CandidateScore, DecisionWeights, DecisionResult

class DecisionEngine:
    EPSILON = 1e-6

    @classmethod
    def rank_candidates(
        cls,
        benchmarks: List[BenchmarkResult],
        weights: DecisionWeights = DecisionWeights()
    ) -> DecisionResult:
        if not benchmarks:
            raise ValueError("Cannot rank an empty list of benchmark results.")

        # Extract vectors
        primary_metrics = np.array([b.mean_primary_metric for b in benchmarks])
        fit_times = np.array([b.mean_fit_time_seconds for b in benchmarks])

        # Min-Max Normalization for Performance (higher is better)
        min_p, max_p = np.min(primary_metrics), np.max(primary_metrics)
        if max_p - min_p < cls.EPSILON:
            norm_p = np.ones_like(primary_metrics)
        else:
            norm_p = (primary_metrics - min_p) / ((max_p - min_p) + cls.EPSILON)

        # Min-Max Normalization for Runtime (lower is better)
        min_t, max_t = np.min(fit_times), np.max(fit_times)
        if max_t - min_t < cls.EPSILON:
            norm_t = np.ones_like(fit_times)
        else:
            norm_t = (max_t - fit_times) / ((max_t - min_t) + cls.EPSILON)

        # Multi-Objective Utility Calculation
        w_perf = weights.weight_performance
        w_time = weights.weight_runtime
        total_w = w_perf + w_time
        if total_w > 0:
            w_perf /= total_w
            w_time /= total_w

        utility_scores = (w_perf * norm_p) + (w_time * norm_t)

        candidate_scores: List[CandidateScore] = []
        for i, b in enumerate(benchmarks):
            candidate_scores.append(CandidateScore(
                pipeline_id=b.pipeline_id,
                pipeline_name=b.pipeline_name,
                rank=0,
                utility_score=round(float(utility_scores[i]), 4),
                raw_primary_metric=round(float(primary_metrics[i]), 4),
                raw_fit_time_seconds=round(float(fit_times[i]), 4),
                normalized_primary_metric=round(float(norm_p[i]), 4),
                normalized_fit_time=round(float(norm_t[i]), 4)
            ))

        # Sort descending by utility score
        candidate_scores.sort(key=lambda x: x.utility_score, reverse=True)

        # Assign ordinal ranks (1 to K)
        for idx, score_item in enumerate(candidate_scores):
            score_item.rank = idx + 1

        winner = candidate_scores[0]
        runner_up = candidate_scores[1] if len(candidate_scores) > 1 else None

        if runner_up:
            delta_metric = round(winner.raw_primary_metric - runner_up.raw_primary_metric, 4)
            delta_time = round(runner_up.raw_fit_time_seconds - winner.raw_fit_time_seconds, 4)
            reason = (
                f"Selected '{winner.pipeline_name}' as Rank 1 with utility score {winner.utility_score:.4f}. "
                f"It achieved {winner.raw_primary_metric:.4f} primary metric (Δ={delta_metric:+.4f} vs runner-up) "
                f"and completed in {winner.raw_fit_time_seconds:.3f}s (Δ={delta_time:+.3f}s vs runner-up)."
            )
        else:
            reason = (
                f"Selected '{winner.pipeline_name}' as the sole evaluated candidate "
                f"with primary metric {winner.raw_primary_metric:.4f} and fit time {winner.raw_fit_time_seconds:.3f}s."
            )

        return DecisionResult(
            winning_pipeline_id=winner.pipeline_id,
            winning_pipeline_name=winner.pipeline_name,
            weights_applied=weights,
            leaderboard=candidate_scores,
            selection_reason=reason
        )