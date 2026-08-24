"""
Subjective MOS / CMOS / SMOS Evaluation Harness.
Provides templates and scoring protocols for human evaluation of voice cloning quality.
"""

from typing import Dict, List, Optional
import numpy as np


class HumanEvaluationHarness:
    """Manages crowdsourced/expert subjective listening tests."""

    def __init__(self, experiment_id: str = "exp_zero_shot_v1"):
        self.experiment_id = experiment_id
        self.ratings: List[Dict[str, object]] = []

    def create_evaluation_task(
        self,
        task_id: str,
        sample_a_path: str,
        sample_b_path: Optional[str] = None,
        reference_path: Optional[str] = None,
        transcript: str = "",
        eval_type: str = "MOS" # "MOS", "CMOS", "SMOS"
    ) -> Dict[str, object]:
        """
        Creates an evaluation question template:
        - MOS (1 to 5): Naturalness rating
        - CMOS (-3 to +3): Comparative preference vs Baseline
        - SMOS (1 to 5): Speaker identity similarity vs Reference
        """
        return {
            "task_id": task_id,
            "eval_type": eval_type,
            "sample_a_path": sample_a_path,
            "sample_b_path": sample_b_path,
            "reference_path": reference_path,
            "transcript": transcript,
            "instructions": (
                "Rate naturalness on a 1-5 scale" if eval_type == "MOS"
                else "Rate which audio sounds closer in speaker timbre to the reference" if eval_type == "SMOS"
                else "Rate comparative preference between Audio A and Audio B from -3 to +3"
            )
        }

    def record_rating(self, task_id: str, rater_id: str, score: float):
        self.ratings.append({
            "task_id": task_id,
            "rater_id": rater_id,
            "score": score
        })

    def aggregate_results(self) -> Dict[str, float]:
        """Compute mean score and 95% confidence intervals."""
        if not self.ratings:
            return {"count": 0, "mean": 0.0, "ci_95": 0.0}

        scores = np.array([r["score"] for r in self.ratings], dtype=np.float32)
        mean_score = float(np.mean(scores))
        std_err = float(np.std(scores) / np.sqrt(len(scores)))
        ci_95 = 1.96 * std_err

        return {
            "count": len(scores),
            "mean_score": mean_score,
            "ci_95": float(ci_95),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores))
        }
