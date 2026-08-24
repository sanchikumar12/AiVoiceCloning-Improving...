"""
Speech Dataset & Batch Loader for Zero-Shot Voice Training.
Supports audio-text-speaker triplets, dynamic padding, and sharded streaming.
"""

import numpy as np
from typing import Dict, Iterator, List, Optional, Tuple


class VoiceCloningDataset:
    """Dataset storing audio utterances, transcripts, and speaker metadata."""

    def __init__(self, samples: Optional[List[Dict[str, object]]] = None):
        self.samples = samples or []

    def __len__(self) -> int:
        return len(self.samples)

    def add_sample(self, audio: np.ndarray, transcript: str, speaker_id: str, sample_rate: int = 24000):
        self.samples.append({
            "audio": audio,
            "transcript": transcript,
            "speaker_id": speaker_id,
            "sample_rate": sample_rate
        })

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.samples[idx]

    def iterate_batches(self, batch_size: int = 8, shuffle: bool = True) -> Iterator[Dict[str, object]]:
        """Yield padded training batches."""
        indices = np.arange(len(self.samples))
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(self.samples), batch_size):
            batch_idx = indices[start : start + batch_size]
            batch_samples = [self.samples[i] for i in batch_idx]
            
            audios = [s["audio"] for s in batch_samples]
            transcripts = [s["transcript"] for s in batch_samples]
            speaker_ids = [s["speaker_id"] for s in batch_samples]

            # Dynamic audio padding
            max_audio_len = max(len(a) for a in audios)
            padded_audios = np.zeros((len(audios), max_audio_len), dtype=np.float32)
            for i, a in enumerate(audios):
                padded_audios[i, : len(a)] = a

            yield {
                "audio": padded_audios,
                "transcripts": transcripts,
                "speaker_ids": speaker_ids,
                "batch_size": len(batch_samples)
            }
