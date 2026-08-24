"""
Batched Stage 2 Flow-Matching Serving Worker.
Batches ODE trajectory integration steps for concurrent synthesis tasks.
"""

from typing import Dict, List, Optional
import numpy as np


class Stage2ServerWorker:
    """Microservice worker for DiT flow-matching generation."""

    def __init__(self, stage2_model):
        self.model = stage2_model

    def batch_sample_ode(
        self,
        batch_semantic_cond: List[np.ndarray],
        batch_speaker_embeddings: List[np.ndarray],
        ode_steps: int = 16,
        solver: str = "midpoint"
    ) -> List[np.ndarray]:
        """Runs parallel ODE integration across batched conditioning inputs."""
        results = []
        for sem_cond, spk_emb in zip(batch_semantic_cond, batch_speaker_embeddings):
            latents = self.model.sample_ode(
                semantic_cond=sem_cond,
                speaker_emb=spk_emb,
                num_steps=ode_steps,
                solver=solver
            )
            results.append(latents)
        return results
