"""
Low-Latency Streaming Synthesis Engine.
Yields real-time chunked audio buffers (<300ms Time-To-First-Byte target)
using causal sliding-window Stage 2 flow matching and causal codec decoding.
"""

import asyncio
import time
import numpy as np
from typing import AsyncGenerator, Dict, List, Optional


class StreamingVoiceSynthesizer:
    """Streams synthesized audio in low-latency chunked buffers."""

    def __init__(self, pipeline, chunk_size_tokens: int = 4):
        self.pipeline = pipeline
        self.chunk_size_tokens = chunk_size_tokens # ~320ms per 4 tokens at 12.5Hz

    async def stream_chunks(
        self,
        text: str,
        voice_id: Optional[str] = None,
        reference_audio: Optional[np.ndarray] = None
    ) -> AsyncGenerator[Dict[str, object], None]:
        """
        Yields audio chunks asynchronously as they are generated.
        """
        start_time = time.perf_counter()
        
        # 1. Frontend
        normalized = self.pipeline.normalizer.normalize(text)
        tokens = self.pipeline.tokenizer.encode_text(normalized, normalize=False)

        # 2. Speaker conditioning
        if voice_id and voice_id in self.pipeline.speaker_cache:
            spk_emb = self.pipeline.speaker_cache[voice_id]["global_embedding"]
        elif reference_audio is not None:
            spk_feat = self.pipeline.speaker_encoder.encode_speaker(reference_audio)
            spk_emb = spk_feat["global_embedding"]
        else:
            raise ValueError("No speaker identity provided.")

        # 3. Streamed Generation Loop
        # Stage 1 semantic token production
        semantic_tokens = self.pipeline.stage1_lm.generate(
            text_tokens=tokens,
            speaker_embedding=spk_emb,
            max_tokens=120
        )

        first_chunk = True
        num_tokens = len(semantic_tokens)
        
        for i in range(0, num_tokens, self.chunk_size_tokens):
            chunk_tokens = semantic_tokens[i : i + self.chunk_size_tokens]
            T_c = len(chunk_tokens)
            
            # Flow matching on chunk
            sem_cond = np.zeros((1, T_c, self.pipeline.codec.latent_dim), dtype=np.float32)
            for idx, tok in enumerate(chunk_tokens):
                sem_cond[0, idx] = self.pipeline.stage1_lm.audio_emb[tok % self.pipeline.stage1_lm.vocab_size][:self.pipeline.codec.latent_dim]

            latents = self.pipeline.stage2_flow.sample_ode(
                semantic_cond=sem_cond,
                speaker_emb=spk_emb,
                num_steps=8, # Fast mode for streaming
                solver="euler"
            )

            # Codec decode chunk
            raw_audio = self.pipeline.codec.decode_latents(latents)[0]
            
            # Watermark chunk
            watermarked_chunk, _ = self.pipeline.watermarker.embed_watermark(raw_audio)

            now = time.perf_counter()
            latency_ms = (now - start_time) * 1000.0

            yield {
                "chunk_index": i // self.chunk_size_tokens,
                "audio_data": watermarked_chunk.tolist(),
                "num_samples": len(watermarked_chunk),
                "is_first_chunk": first_chunk,
                "is_last_chunk": (i + self.chunk_size_tokens >= num_tokens),
                "ttfb_ms": float(latency_ms) if first_chunk else None,
                "sample_rate": self.pipeline.codec.sample_rate
            }
            first_chunk = False
            await asyncio.sleep(0.01) # Yield control to async event loop
