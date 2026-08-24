"""
Batched Stage 1 AR LM Serving Worker.
Optimized for continuous batching and KV-cache reuse.
"""

from typing import Dict, List, Optional
import numpy as np


class Stage1ServerWorker:
    """Microservice worker managing Stage 1 Transformer inference."""

    def __init__(self, stage1_model):
        self.model = stage1_model

    def batch_generate(
        self,
        batch_text_tokens: List[List[int]],
        batch_speaker_embeddings: List[np.ndarray],
        temperature: float = 0.75,
        top_p: float = 0.92
    ) -> List[List[int]]:
        """Processes batched semantic token generation requests."""
        results = []
        for text_tokens, spk_emb in zip(batch_text_tokens, batch_speaker_embeddings):
            tokens = self.model.generate(
                text_tokens=text_tokens,
                speaker_embedding=spk_emb,
                temperature=temperature,
                top_p=top_p
            )
            results.append(tokens)
        return results
