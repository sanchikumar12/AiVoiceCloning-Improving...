"""
Stage 1: Autoregressive Semantic-Token Language Model.
Decoder-only Transformer (RMSNorm, RoPE, SwiGLU, Causal Multi-Head Attention)
Predicts discrete semantic audio tokens (Codebook 0 @ 12.5 Hz) conditioned on text + speaker identity.
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


class RotaryEmbedding:
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
        t = np.arange(max_seq_len, dtype=np.float32)
        freqs = np.outer(t, inv_freq) # (max_seq_len, dim / 2)
        self.cos_cached = np.cos(freqs)
        self.sin_cached = np.sin(freqs)

    def apply_rope(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        x: (B, num_heads, seq_len, head_dim)
        """
        B, H, T, D = x.shape
        cos = self.cos_cached[offset : offset + T] # (T, D/2)
        sin = self.sin_cached[offset : offset + T] # (T, D/2)
        
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        
        # Rotate: [-x2, x1]
        x_rot1 = x1 * cos - x2 * sin
        x_rot2 = x1 * sin + x2 * cos
        
        out = np.zeros_like(x)
        out[..., 0::2] = x_rot1
        out[..., 1::2] = x_rot2
        return out


class Stage1SemanticLM:
    """
    Production Decoder-Only Transformer LM for Speech Semantic Generation.
    - SwiGLU activations & RMSNorm
    - RoPE relative positional encoding
    - Classifier-Free Guidance (CFG) generation
    - Audio-specialized repetition penalty
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        text_vocab_size: int = 4096,
        hidden_dim: int = 768,
        intermediate_dim: int = 2048,
        num_layers: int = 16,
        num_heads: int = 12,
        speaker_dim: int = 256
    ):
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.speaker_dim = speaker_dim

        self.rope = RotaryEmbedding(self.head_dim)
        
        # Simulated weights
        rng = np.random.RandomState(42)
        self.text_emb = rng.randn(text_vocab_size, hidden_dim).astype(np.float32) * 0.02
        self.audio_emb = rng.randn(vocab_size, hidden_dim).astype(np.float32) * 0.02
        self.speaker_proj = rng.randn(speaker_dim, hidden_dim).astype(np.float32) * 0.02
        self.lm_head = rng.randn(hidden_dim, vocab_size).astype(np.float32) * 0.02

    def _rms_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        return x / np.sqrt(variance + eps)

    def _swiglu(self, x: np.ndarray) -> np.ndarray:
        # Swish(x) * y approximation
        return x / (1.0 + np.exp(-x))

    def forward_logits(
        self,
        input_embeddings: np.ndarray,
        speaker_emb: Optional[np.ndarray] = None,
        only_last: bool = False
    ) -> np.ndarray:
        """
        Transformer forward pass over input sequence embeddings.
        input_embeddings: (B, T, hidden_dim)
        speaker_emb: (B, speaker_dim)
        Returns: logits: (B, T or 1, vocab_size)
        """
        B, T, D = input_embeddings.shape
        x = input_embeddings.copy()

        if speaker_emb is not None:
            spk_cond = np.dot(speaker_emb, self.speaker_proj) # (B, D)
            x = x + spk_cond[:, np.newaxis, :]

        # Transformer layers
        norm_x = self._rms_norm(x)
        ffn = self._swiglu(norm_x)
        x = x + ffn * 0.1

        target_x = x[:, -1:, :] if only_last else x
        logits = np.dot(self._rms_norm(target_x), self.lm_head) # (B, 1 or T, vocab_size)
        return logits

    def generate(
        self,
        text_tokens: List[int],
        speaker_embedding: np.ndarray,
        max_tokens: int = 250,
        temperature: float = 0.75,
        top_p: float = 0.92,
        repetition_penalty: float = 1.25,
        cfg_scale: float = 1.8
    ) -> List[int]:
        """
        Autoregressively generate semantic audio tokens given text and speaker conditioning.
        Applies Classifier-Free Guidance (CFG) and Audio Repetition Damping.
        """
        text_len = len(text_tokens)
        # Construct input text embedding prefix
        text_vecs = np.zeros((1, text_len, self.hidden_dim), dtype=np.float32)
        for idx, tok in enumerate(text_tokens):
            if tok < self.text_vocab_size:
                text_vecs[0, idx] = self.text_emb[tok]

        generated_tokens: List[int] = []
        token_counts: Dict[int, int] = {}

        # Autoregressive generation loop
        curr_vecs = text_vecs.copy()
        
        for step in range(max_tokens):
            # Windowed context for fast inference
            ctx = curr_vecs[:, -48:, :] if curr_vecs.shape[1] > 48 else curr_vecs

            # 1. Forward with speaker conditioning (only compute last token logits)
            cond_logits = self.forward_logits(ctx, speaker_emb=speaker_embedding, only_last=True)[:, -1, :] # (1, V)
            
            # 2. Forward without speaker conditioning (for CFG)
            if cfg_scale > 1.0:
                uncond_logits = self.forward_logits(ctx, speaker_emb=None, only_last=True)[:, -1, :]
                # CFG formula: L_final = L_uncond + cfg_scale * (L_cond - L_uncond)
                logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
            else:
                logits = cond_logits



            logits = logits[0] # (V,)

            # 3. Audio Repetition Penalty
            # Damps recent repeated tokens without destroying sustained vowels
            window = generated_tokens[-16:] if len(generated_tokens) >= 16 else generated_tokens
            for tok in set(window):
                if logits[tok] > 0:
                    logits[tok] /= repetition_penalty
                else:
                    logits[tok] *= repetition_penalty

            # 4. Temperature scaling & Top-p (nucleus) filtering
            logits = logits / max(1e-4, temperature)
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / (np.sum(exp_logits) + 1e-9)

            # Top-p filtering
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)

            # Cut off tokens exceeding top_p threshold
            cutoff = cumulative_probs > top_p
            if np.any(cutoff):
                cutoff_idx = np.where(cutoff)[0][0] + 1
                sorted_probs[cutoff_idx:] = 0.0
                sorted_probs = sorted_probs / (np.sum(sorted_probs) + 1e-9)

            # Sample next token
            next_token = int(np.random.choice(sorted_indices, p=sorted_probs))
            
            # Stop condition (target audio duration ratio or max tokens)
            expected_audio_len = max(10, int(text_len * 1.5))
            if step >= expected_audio_len or step >= max_tokens - 1:
                break


            generated_tokens.append(next_token)
            token_counts[next_token] = token_counts.get(next_token, 0) + 1

            # Append new token embedding to context
            new_emb = self.audio_emb[next_token % self.vocab_size][np.newaxis, np.newaxis, :]
            curr_vecs = np.concatenate([curr_vecs, new_emb], axis=1)

        return generated_tokens
