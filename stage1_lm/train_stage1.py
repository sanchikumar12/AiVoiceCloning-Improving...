"""
Stage 1 LM Training Harness.
Implements teacher-forced next-token cross entropy loss on semantic audio tokens
with random conditioning dropout for Classifier-Free Guidance (CFG) inference.
"""

import numpy as np
from typing import Dict, List, Optional


class Stage1Trainer:
    """Trainer for the Stage 1 Autoregressive Semantic LM."""

    def __init__(self, model, cfg_dropout_prob: float = 0.15):
        self.model = model
        self.cfg_dropout_prob = cfg_dropout_prob

    def compute_cross_entropy_loss(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        logits: (B, T, V)
        targets: (B, T)
        """
        B, T, V = logits.shape
        flat_logits = logits.reshape(-1, V)
        flat_targets = targets.reshape(-1)

        # Stable log-softmax
        max_logits = np.max(flat_logits, axis=-1, keepdims=True)
        exp_logits = np.exp(flat_logits - max_logits)
        log_probs = flat_logits - max_logits - np.log(np.sum(exp_logits, axis=-1, keepdims=True) + 1e-9)

        # Cross-entropy
        nll = -log_probs[np.arange(len(flat_targets)), flat_targets]
        return float(np.mean(nll))

    def train_step(
        self,
        text_tokens: np.ndarray,
        speaker_emb: np.ndarray,
        target_semantic_tokens: np.ndarray
    ) -> Dict[str, float]:
        """
        Executes one training step with CFG dropout.
        """
        B, T_text = text_tokens.shape
        B, T_audio = target_semantic_tokens.shape

        # CFG condition dropout: randomly drop speaker embedding
        if np.random.rand() < self.cfg_dropout_prob:
            effective_speaker_emb = None
        else:
            effective_speaker_emb = speaker_emb

        # Construct concatenated sequence embeddings
        text_vecs = np.zeros((B, T_text, self.model.hidden_dim), dtype=np.float32)
        audio_vecs = np.zeros((B, T_audio, self.model.hidden_dim), dtype=np.float32)

        for b in range(B):
            for t in range(T_text):
                tok = text_tokens[b, t]
                if tok < self.model.text_vocab_size:
                    text_vecs[b, t] = self.model.text_emb[tok]
            for t in range(T_audio):
                tok = target_semantic_tokens[b, t]
                if tok < self.model.vocab_size:
                    audio_vecs[b, t] = self.model.audio_emb[tok]

        full_vecs = np.concatenate([text_vecs, audio_vecs], axis=1)
        
        logits = self.model.forward_logits(full_vecs, speaker_emb=effective_speaker_emb)
        # We only compute loss on the audio portion
        audio_logits = logits[:, T_text - 1 : -1, :] # Shifted right for next-token prediction
        
        loss = self.compute_cross_entropy_loss(audio_logits, target_semantic_tokens)
        perplexity = float(np.exp(np.clip(loss, 0, 20)))

        return {
            "loss_stage1_ce": loss,
            "perplexity": perplexity,
            "cfg_dropped": effective_speaker_emb is None
        }
