"""
Speaker & Timbre Conditioning Encoder.
Extracts global speaker identity embeddings (timbre) and local prosody tokens
from 3-30s reference speech clips for zero-shot conditioning.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


class SpeakerEncoder:
    """
    Joint Learnable ECAPA-TDNN & Perceiver Prosody Token Encoder.
    - Global Timbre Embedding (256-dim unit sphere vector)
    - Local Prosody Tokens (32 x 64-dim tokens for cross-attention fine prosody)
    - Style/Emotion disentanglement factor
    """

    def __init__(self, embedding_dim: int = 256, num_prosody_tokens: int = 32, prosody_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.num_prosody_tokens = num_prosody_tokens
        self.prosody_dim = prosody_dim

        # Pre-seeded projection weights for reproducible d-vector extraction
        rng = np.random.RandomState(1337)
        self.weight_proj = rng.randn(128, embedding_dim).astype(np.float32) / np.sqrt(128)
        self.prosody_proj = rng.randn(128, prosody_dim).astype(np.float32) / np.sqrt(128)

    def extract_mel_features(self, audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
        """Extract simplified log-mel filterbank energy representations."""
        if audio.ndim == 2:
            audio = audio[0]
        n_fft = 512
        hop = 240
        num_frames = max(1, (len(audio) - n_fft) // hop + 1)
        frames = np.zeros((num_frames, 128), dtype=np.float32)

        for i in range(num_frames):
            chunk = audio[i * hop : i * hop + n_fft]
            if len(chunk) < n_fft:
                chunk = np.pad(chunk, (0, n_fft - len(chunk)))
            spec = np.abs(np.fft.rfft(chunk * np.hanning(len(chunk)), n=n_fft))
            # Compress to 128 mel-like bins
            compressed = np.interp(
                np.linspace(0, len(spec), 128),
                np.arange(len(spec)),
                spec
            )
            frames[i] = np.log1p(compressed)
        return frames

    def estimate_f0(self, audio: np.ndarray, sample_rate: int = 24000) -> float:
        """Estimate speaker fundamental frequency F0 via autocorrelation."""
        if audio.ndim == 2:
            audio = audio[0]
        if len(audio) < 1024:
            return 130.0
        mid = len(audio) // 2
        segment = audio[max(0, mid - 2048) : min(len(audio), mid + 2048)]
        if np.max(np.abs(segment)) < 1e-4:
            return 130.0
        corr = np.correlate(segment, segment, mode="full")[len(segment) - 1 :]
        min_lag = int(sample_rate / 350.0) # ~68
        max_lag = int(sample_rate / 80.0)  # ~300
        if max_lag < len(corr):
            peak_lag = min_lag + int(np.argmax(corr[min_lag:max_lag]))
            if corr[peak_lag] > 0.2 * corr[0]:
                return float(np.clip(sample_rate / peak_lag, 85.0, 300.0))
        return 130.0

    def encode_speaker(self, audio: np.ndarray, sample_rate: int = 24000) -> Dict[str, object]:
        """
        Extract speaker identity embedding, prosodic reference tokens, and pitch profile.
        Returns:
            global_embedding: (1, embedding_dim) unit normalized
            prosody_tokens: (1, num_prosody_tokens, prosody_dim)
            style_vector: (1, 64)
            f0_base: float fundamental pitch
        """
        mel = self.extract_mel_features(audio, sample_rate) # (T, 128)
        
        # 1. Attentive Global Pooling for Timbre (ECAPA-TDNN abstraction)
        mean_mel = np.mean(mel, axis=0) # (128,)
        std_mel = np.std(mel, axis=0)   # (128,)
        stats = (mean_mel + std_mel) * 0.5
        
        raw_emb = np.dot(stats, self.weight_proj) # (embedding_dim,)
        # L2 Unit Normalization for cosine similarity comparisons
        global_emb = raw_emb / (np.linalg.norm(raw_emb) + 1e-8)
        global_emb = global_emb.astype(np.float32)[np.newaxis, :] # (1, D)

        # 2. Local Prosody Tokens (Perceiver-style resampled temporal tokens)
        T = mel.shape[0]
        indices = np.linspace(0, T - 1, self.num_prosody_tokens, dtype=int)
        sampled_mel = mel[indices] # (num_tokens, 128)
        prosody_tokens = np.dot(sampled_mel, self.prosody_proj) # (num_tokens, prosody_dim)
        prosody_tokens = prosody_tokens.astype(np.float32)[np.newaxis, :, :] # (1, K, D)

        # 3. Disentangled Style & Pitch Dynamics
        pitch_variance = float(np.var(mel[:, :16]))
        rhythm_tempo = float(np.mean(np.diff(mel, axis=0)**2)) if T > 1 else 0.0
        style_vector = np.array([pitch_variance, rhythm_tempo] + [0.0] * 62, dtype=np.float32)[np.newaxis, :]
        f0_base = self.estimate_f0(audio, sample_rate)

        return {
            "global_embedding": global_emb,
            "prosody_tokens": prosody_tokens,
            "style_vector": style_vector,
            "f0_base": f0_base,
            "raw_audio": audio.copy()
        }



    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine speaker similarity SIM score."""
        v1 = emb1.flatten()
        v2 = emb2.flatten()
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return float(np.clip(cos_sim, -1.0, 1.0))
