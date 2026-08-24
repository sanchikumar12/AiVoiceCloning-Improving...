"""
Neural Codec Decoder Serving Service.
High-throughput acoustic latent-to-waveform vocoder decoding.
"""

from typing import List, Optional
import numpy as np


class CodecDecodeWorker:
    """Microservice worker executing causal vocoder synthesis."""

    def __init__(self, codec_model):
        self.codec = codec_model

    def decode_batch(self, batch_latents: List[np.ndarray]) -> List[np.ndarray]:
        results = []
        for latents in batch_latents:
            wav = self.codec.decode_latents(latents)[0]
            results.append(wav)
        return results
