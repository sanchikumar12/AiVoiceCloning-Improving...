"""
Stage 2 Flow-Matching Decoder Training.
Implements Optimal Transport Conditional Flow Matching (OT-CFM) Loss
with random continuous span masking for zero-shot infilling.
"""

import numpy as np
from typing import Dict, Optional, Tuple


class FlowMatchingTrainer:
    """Trainer for Stage 2 Conditional Flow-Matching DiT."""

    def __init__(self, model, mask_ratio: float = 0.7):
        self.model = model
        self.mask_ratio = mask_ratio

    def create_random_span_mask(self, batch_size: int, seq_len: int) -> np.ndarray:
        """Create contiguous masked spans for in-context zero-shot infilling."""
        mask = np.zeros((batch_size, seq_len, 1), dtype=np.float32)
        mask_len = int(seq_len * self.mask_ratio)
        for b in range(batch_size):
            if seq_len > mask_len:
                start = np.random.randint(0, seq_len - mask_len)
                mask[b, start : start + mask_len] = 1.0
            else:
                mask[b, :] = 1.0
        return mask

    def train_step(
        self,
        ground_truth_latents: np.ndarray, # x_1 ~ q(x_1) target acoustic latents
        semantic_cond: np.ndarray,        # conditioning from Stage 1
        speaker_emb: np.ndarray           # speaker identity
    ) -> Dict[str, float]:
        """
        Executes one OT-CFM training step:
        1. Sample t ~ Uniform(0, 1)
        2. Sample x_0 ~ N(0, I)
        3. Form OT path: x_t = (1 - (1 - sigma_min)*t)*x_0 + t*x_1
        4. Target velocity: u_t = x_1 - (1 - sigma_min)*x_0
        5. Loss = || v_theta(x_t, t | c) - u_t ||^2
        """
        B, T, D = ground_truth_latents.shape
        
        # 1. Sample continuous time t
        t = float(np.random.uniform(0.0, 1.0))
        
        # 2. Sample Gaussian noise prior x_0
        x_0 = np.random.randn(B, T, D).astype(np.float32)
        x_1 = ground_truth_latents
        sigma_min = self.model.sigma_min

        # 3. Optimal Transport interpolation
        x_t = (1.0 - (1.0 - sigma_min) * t) * x_0 + t * x_1

        # 4. Target conditional velocity field u_t
        u_t = x_1 - (1.0 - sigma_min) * x_0

        # 5. Model predicted velocity
        v_pred = self.model.forward_velocity(x_t, t, semantic_cond, speaker_emb)

        # 6. Mean squared error CFM loss
        loss_cfm = np.mean((v_pred - u_t) ** 2)

        return {
            "loss_cfm": float(loss_cfm),
            "sampled_timestep": t,
            "velocity_norm": float(np.mean(np.abs(v_pred)))
        }
