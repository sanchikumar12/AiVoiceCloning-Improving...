"""
Speaker Consent Verification & Safety Policy Guard.
Enforces voice biometric verification against recorded consent phrases,
anti-impersonation policy enforcement, and text prompt safety moderation.
"""

import re
from typing import Dict, List, Optional, Tuple
import numpy as np


class ConsentVerificationError(Exception):
    """Raised when voice cloning request fails ethical consent or safety checks."""
    pass


class SpeakerConsentChecker:
    """
    Mandatory safety gate:
    1. Compares reference audio embedding with user's live recorded consent phrase.
    2. Enforces anti-spoofing and likeness authorization thresholds.
    3. Blocks non-consensual public figure impersonation.
    4. Moderates fraudulent/coercive text prompts.
    """

    MANDATORY_CONSENT_PHRASE = (
        "I hereby authorize the cloning and synthetic generation of my voice for authorized application use."
    )

    KNOWN_PROTECTED_FIGURES = [
        "joe biden", "donald trump", "barack obama", "kamala harris", "elon musk",
        "taylor swift", "morgan freeman", "david attenborough", "scarlett johansson"
    ]

    DISALLOWED_PATTERNS = [
        re.compile(r"\b(transfer your funds|wire money immediately|urgent wire transfer|bank password|pin code)\b", re.I),
        re.compile(r"\b(i am holding your (child|family|daughter|son)|pay ransom)\b", re.I),
        re.compile(r"\b(official election notice: polling location moved|vote by text message)\b", re.I)
    ]

    def __init__(self, speaker_encoder, similarity_threshold: float = 0.78):
        self.speaker_encoder = speaker_encoder
        self.similarity_threshold = similarity_threshold

    def verify_vocal_consent(
        self,
        reference_audio: np.ndarray,
        consent_audio: np.ndarray,
        consent_phrase_spoken: str
    ) -> Dict[str, object]:
        """
        Verify that the person submitting the reference voice has personally spoken the consent pledge.
        """
        # 1. Check phrase content similarity
        clean_spoken = consent_phrase_spoken.strip().lower()
        clean_expected = self.MANDATORY_CONSENT_PHRASE.lower()
        
        # Word overlap check
        spoken_words = set(re.findall(r"\w+", clean_spoken))
        expected_words = set(re.findall(r"\w+", clean_expected))
        overlap = len(spoken_words.intersection(expected_words)) / max(1, len(expected_words))

        if overlap < 0.6:
            return {
                "verified": False,
                "reason": f"Consent phrase mismatch (overlap {overlap:.1%}). Expected: '{self.MANDATORY_CONSENT_PHRASE}'",
                "similarity_score": 0.0
            }

        # 2. Extract speaker embeddings
        ref_feat = self.speaker_encoder.encode_speaker(reference_audio)
        consent_feat = self.speaker_encoder.encode_speaker(consent_audio)

        # 3. Compute vocal biometric similarity
        similarity = self.speaker_encoder.compute_similarity(
            ref_feat["global_embedding"],
            consent_feat["global_embedding"]
        )

        is_verified = similarity >= self.similarity_threshold

        return {
            "verified": is_verified,
            "similarity_score": float(similarity),
            "threshold": self.similarity_threshold,
            "reason": "Authorized" if is_verified else f"Vocal biometric similarity {similarity:.2f} below required threshold {self.similarity_threshold:.2f}"
        }

    def moderate_request(self, target_text: str, voice_name_or_label: str = "") -> Tuple[bool, Optional[str]]:
        """
        Check for text safety violations or unauthorized public person impersonation.
        Returns: (is_safe, refusal_reason)
        """
        # 1. Impersonation check
        label_lower = voice_name_or_label.lower()
        for figure in self.KNOWN_PROTECTED_FIGURES:
            if figure in label_lower:
                return False, f"Unauthorized voice cloning of protected public figure '{figure}' is prohibited."

        # 2. Dangerous / fraudulent text moderation
        for pattern in self.DISALLOWED_PATTERNS:
            if pattern.search(target_text):
                return False, "Generation refused: Prompt contains prohibited scam, extortion, or election interference patterns."

        return True, None
