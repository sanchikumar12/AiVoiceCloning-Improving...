"""
Stage 2: Conditional Flow-Matching Decoder (DiT / ConvNeXt Backbone).
Converts discrete semantic tokens + speaker conditioning into continuous acoustic latents
using Optimal Transport Conditional Flow Matching (OT-CFM) and Fast ODE sampling.
"""

import math
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union


class ConditionalFlowMatchingDiT:
    """
    Diffusion Transformer (DiT) for Conditional Flow Matching.
    Generates high-fidelity acoustic latents from semantic tokens and speaker identity.
    """

    def __init__(
        self,
        in_channels: int = 512, # Latent dimension
        cond_dim: int = 512,
        speaker_dim: int = 256,
        hidden_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        sigma_min: float = 1e-4
    ):
        self.in_channels = in_channels
        self.cond_dim = cond_dim
        self.speaker_dim = speaker_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.num_heads = num_heads
        self.sigma_min = sigma_min

        # Layer projections
        rng = np.random.RandomState(88)
        self.time_embed = rng.randn(128, hidden_dim).astype(np.float32) * 0.02
        self.cond_proj = rng.randn(cond_dim + speaker_dim, hidden_dim).astype(np.float32) * 0.02
        self.in_proj = rng.randn(in_channels, hidden_dim).astype(np.float32) * 0.02
        self.out_proj = rng.randn(hidden_dim, in_channels).astype(np.float32) * 0.02

    def _get_timestep_embedding(self, timesteps: np.ndarray, dim: int = 128) -> np.ndarray:
        """Sinusoidal timestep embedding for continuous time t in [0, 1]."""
        half_dim = dim // 2
        freqs = np.exp(-np.log(10000.0) * np.arange(0, half_dim, dtype=np.float32) / half_dim)
        args = np.outer(timesteps, freqs)
        emb = np.concatenate([np.cos(args), np.sin(args)], axis=-1)
        return emb

    def forward_velocity(
        self,
        x_t: np.ndarray,
        t: float,
        semantic_cond: np.ndarray,
        speaker_emb: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Predict vector field velocity v_t(x_t, t | condition).
        x_t: (B, T, in_channels)
        t: scalar in [0, 1]
        semantic_cond: (B, T, cond_dim)
        speaker_emb: (B, speaker_dim)
        Returns: predicted velocity v_t: (B, T, in_channels)
        """
        B, T, C = x_t.shape
        t_arr = np.array([t] * B, dtype=np.float32)
        t_emb = np.dot(self._get_timestep_embedding(t_arr, 128), self.time_embed) # (B, H)

        # Combine semantic condition with speaker embedding
        if speaker_emb is not None:
            spk_tiled = np.repeat(speaker_emb[:, np.newaxis, :], T, axis=1) # (B, T, spk_dim)
            joint_cond = np.concatenate([semantic_cond, spk_tiled], axis=-1)
        else:
            dummy_spk = np.zeros((B, T, self.speaker_dim), dtype=np.float32)
            joint_cond = np.concatenate([semantic_cond, dummy_spk], axis=-1)

        c_proj = np.dot(joint_cond, self.cond_proj) # (B, T, H)
        h = np.dot(x_t, self.in_proj) + c_proj + t_emb[:, np.newaxis, :]

        # Non-linear flow layers (DiT AdaLN feed-forward approximation)
        h = np.tanh(h)
        v_pred = np.dot(h, self.out_proj) # (B, T, in_channels)
        return v_pred

    def sample_ode(
        self,
        semantic_cond: np.ndarray,
        speaker_emb: np.ndarray,
        num_steps: int = 16,
        solver: str = "midpoint",
        cfg_scale: float = 2.0
    ) -> np.ndarray:
        """
        Integrates the probability flow ODE from t=0 (prior noise) to t=1 (target acoustic latents).
        Optimal Transport path: x_t = (1 - (1 - sigma_min)*t)*x_0 + t*x_1
        """
        B, T, _ = semantic_cond.shape
        # Initialize x_0 ~ N(0, I) standard Gaussian noise
        x_t = np.random.randn(B, T, self.in_channels).astype(np.float32)
        dt = 1.0 / num_steps

        # Precompute static condition projections outside ODE loop
        spk_tiled = np.repeat(speaker_emb[:, np.newaxis, :], T, axis=1) # (B, T, spk_dim)
        joint_cond = np.concatenate([semantic_cond, spk_tiled], axis=-1)
        c_proj_cond = np.dot(joint_cond, self.cond_proj) # (B, T, H)

        if cfg_scale > 1.0:
            dummy_spk = np.zeros((B, T, self.speaker_dim), dtype=np.float32)
            joint_uncond = np.concatenate([semantic_cond, dummy_spk], axis=-1)
            c_proj_uncond = np.dot(joint_uncond, self.cond_proj) # (B, T, H)

        def eval_v(x, t_val):
            t_arr = np.array([t_val] * B, dtype=np.float32)
            t_emb = np.dot(self._get_timestep_embedding(t_arr, 128), self.time_embed)[:, np.newaxis, :]
            x_in = np.dot(x, self.in_proj)
            
            h_cond = np.tanh(x_in + c_proj_cond + t_emb)
            v_cond = np.dot(h_cond, self.out_proj)

            if cfg_scale > 1.0:
                h_uncond = np.tanh(x_in + c_proj_uncond + t_emb)
                v_uncond = np.dot(h_uncond, self.out_proj)
                return v_uncond + cfg_scale * (v_cond - v_uncond)
            return v_cond

        for step in range(num_steps):
            t = step * dt
            v_t = eval_v(x_t, t)

            if solver == "midpoint" and step < num_steps - 1:
                x_mid = x_t + 0.5 * dt * v_t
                t_mid = t + 0.5 * dt
                v_mid = eval_v(x_mid, t_mid)
                x_t = x_t + dt * v_mid
            else:
                x_t = x_t + dt * v_t

        return x_t

