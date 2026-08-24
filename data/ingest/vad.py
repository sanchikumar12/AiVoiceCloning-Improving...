"""
Voice Activity Detection (VAD) and Utterance Segmenter.
Segments continuous audio files into 2-20s utterance-level training clips.
"""

import numpy as np
from typing import Dict, List, Tuple


class UtteranceSegmenter:
    """Energy-based & Silero-compatible Voice Activity Detector."""

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_ms: int = 30,
        energy_threshold: float = 0.015,
        min_speech_ms: int = 1500,
        max_speech_ms: int = 18000,
        silence_pad_ms: int = 200
    ):
        self.sample_rate = sample_rate
        self.frame_len = int(sample_rate * (frame_ms / 1000.0))
        self.energy_threshold = energy_threshold
        self.min_samples = int(sample_rate * (min_speech_ms / 1000.0))
        self.max_samples = int(sample_rate * (max_speech_ms / 1000.0))
        self.pad_samples = int(sample_rate * (silence_pad_ms / 1000.0))

    def detect_speech_frames(self, audio: np.ndarray) -> np.ndarray:
        """Compute frame-level speech activity mask."""
        num_frames = len(audio) // self.frame_len
        speech_mask = np.zeros(num_frames, dtype=bool)

        for i in range(num_frames):
            frame = audio[i * self.frame_len : (i + 1) * self.frame_len]
            energy = np.sqrt(np.mean(frame ** 2) + 1e-9)
            if energy > self.energy_threshold:
                speech_mask[i] = True
        return speech_mask

    def segment_audio(self, audio: np.ndarray) -> List[Dict[str, object]]:
        """
        Split audio into clean utterance chunks between min_speech_ms and max_speech_ms.
        """
        speech_mask = self.detect_speech_frames(audio)
        segments = []
        in_speech = False
        start_frame = 0

        for idx, is_speech in enumerate(speech_mask):
            if is_speech and not in_speech:
                in_speech = True
                start_frame = idx
            elif not is_speech and in_speech:
                in_speech = False
                end_frame = idx
                
                # Convert to sample indices with padding
                start_sample = max(0, start_frame * self.frame_len - self.pad_samples)
                end_sample = min(len(audio), end_frame * self.frame_len + self.pad_samples)
                duration = end_sample - start_sample

                if self.min_samples <= duration <= self.max_samples:
                    clip = audio[start_sample:end_sample]
                    segments.append({
                        "start_sample": start_sample,
                        "end_sample": end_sample,
                        "duration_sec": duration / self.sample_rate,
                        "audio": clip
                    })

        # If entire audio is active or no silence gaps detected, fallback to chunking
        if not segments and len(audio) >= self.min_samples:
            limit = min(len(audio), self.max_samples)
            segments.append({
                "start_sample": 0,
                "end_sample": limit,
                "duration_sec": limit / self.sample_rate,
                "audio": audio[:limit]
            })

        return segments
