"""
Comprehensive Test Suite for ElevenLabs-Class Voice Cloning Architecture.
Tests Frontend, Speaker Conditioning, Stage 1 LM, Stage 2 Flow DiT, Causal Codec,
AudioSeal Watermarking, Consent Verification, and Objective Metrics.
"""

import os
import sys
import unittest
import numpy as np

# Ensure root on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from frontend.normalize import TextNormalizer
from frontend.g2p import G2PConverter
from frontend.tokenizer import VoiceTokenizer
from speaker_encoder.model import SpeakerEncoder
from speaker_encoder.train_speaker import CrossSampleTrainer
from codec.model import NeuralAudioCodec, SplitRVQQuantizer
from codec.train_codec import MultiScaleSTFTLoss, CodecTrainer
from stage1_lm.model import Stage1SemanticLM
from stage1_lm.train_stage1 import Stage1Trainer
from stage2_flow.model import ConditionalFlowMatchingDiT
from stage2_flow.train_stage2 import FlowMatchingTrainer
from watermark.audioseal_wrapper import AudioSealWatermark
from inference.consent_check import SpeakerConsentChecker, ConsentVerificationError
from inference.pipeline import VoiceCloner
from data.filter.filter_pipeline import QualityFilterPipeline
from eval.objective_metrics import ObjectiveEvaluator


class TestVoiceCloningSystem(unittest.TestCase):

    def setUp(self):
        self.sr = 24000
        self.normalizer = TextNormalizer(locale="en-US")
        self.g2p = G2PConverter()
        self.tokenizer = VoiceTokenizer()
        self.speaker_encoder = SpeakerEncoder(embedding_dim=256)
        self.codec = NeuralAudioCodec(sample_rate=self.sr, frame_rate=12.5)
        self.watermarker = AudioSealWatermark(watermark_bits=16, sample_rate=self.sr)

    def test_text_normalization(self):
        """Verify currency, numbers, abbreviations, and time expansion."""
        raw = "Dr. Smith paid $25.50 on 2026-08-24 at 10:30 AM for 50 kg of AI hardware."
        normalized = self.normalizer.normalize(raw)
        self.assertIn("Doctor", normalized)
        self.assertIn("twenty-five dollars and fifty cents", normalized)
        self.assertIn("ten thirty A M", normalized)
        self.assertIn("kilograms", normalized)
        self.assertIn("A I", normalized)

    def test_g2p_and_tokenizer(self):
        """Verify phonetic decomposition and token ID encoding."""
        text = "Hello neural speech synthesis"
        tokens = self.tokenizer.encode_text(text)
        self.assertTrue(len(tokens) > 0)
        self.assertTrue(all(isinstance(t, int) for t in tokens))
        
        prompt_seq = self.tokenizer.build_prompt_sequence(tokens, speaker_prompt_len=4)
        self.assertEqual(prompt_seq[0], self.tokenizer.bos_id)
        self.assertIn(self.tokenizer.token2id[self.tokenizer.TEXT_START], prompt_seq)
        self.assertIn(self.tokenizer.token2id[self.tokenizer.AUDIO_START], prompt_seq)

    def test_codec_split_rvq(self):
        """Verify causal split-RVQ quantizer and waveform roundtrip."""
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        
        tokens, latents = self.codec.encode(audio)
        # tokens: (1, num_codebooks, num_frames)
        self.assertEqual(tokens.ndim, 3)
        self.assertEqual(tokens.shape[1], 8) # 8 codebooks
        
        # Codebook 0 is semantic token sequence
        semantic_tokens = tokens[0, 0, :]
        self.assertTrue(len(semantic_tokens) > 0)

        # Decode latents to waveform
        decoded_wave = self.codec.decode_latents(latents, target_samples=len(audio))
        self.assertEqual(decoded_wave.shape[-1], len(audio))
        self.assertTrue(np.max(np.abs(decoded_wave)) > 0.0)

    def test_speaker_encoder_and_similarity(self):
        """Verify 256-dim embedding extraction and cosine similarity."""
        t = np.linspace(0, 2.0, self.sr * 2, endpoint=False)
        audio1 = (0.4 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
        audio2 = (0.4 * np.sin(2 * np.pi * 150 * t) + 0.01 * np.random.randn(len(t))).astype(np.float32) # Same voice
        audio3 = (0.4 * np.sin(2 * np.pi * 400 * t)).astype(np.float32) # Different pitch/timbre

        feat1 = self.speaker_encoder.encode_speaker(audio1)
        feat2 = self.speaker_encoder.encode_speaker(audio2)
        feat3 = self.speaker_encoder.encode_speaker(audio3)

        self.assertEqual(feat1["global_embedding"].shape, (1, 256))
        
        sim_same = self.speaker_encoder.compute_similarity(feat1["global_embedding"], feat2["global_embedding"])
        sim_diff = self.speaker_encoder.compute_similarity(feat1["global_embedding"], feat3["global_embedding"])

        self.assertGreater(sim_same, 0.90)
        self.assertGreater(sim_same, sim_diff)

    def test_stage1_lm_generation(self):
        """Verify Stage 1 AR semantic token generation and CFG."""
        stage1 = Stage1SemanticLM(vocab_size=2048, hidden_dim=256, num_layers=4)
        spk_emb = np.random.randn(1, 256).astype(np.float32)
        text_tokens = [10, 25, 30, 45, 12]

        tokens = stage1.generate(
            text_tokens=text_tokens,
            speaker_embedding=spk_emb,
            max_tokens=20,
            temperature=0.7,
            cfg_scale=1.5
        )
        self.assertTrue(len(tokens) > 0)
        self.assertTrue(all(0 <= tok < 2048 for tok in tokens))

    def test_stage2_flow_matching(self):
        """Verify Conditional Flow Matching DiT and Midpoint ODE sampler."""
        dit = ConditionalFlowMatchingDiT(in_channels=128, cond_dim=128, speaker_dim=256, hidden_dim=128)
        sem_cond = np.random.randn(1, 10, 128).astype(np.float32)
        spk_emb = np.random.randn(1, 256).astype(np.float32)

        latents = dit.sample_ode(
            semantic_cond=sem_cond,
            speaker_emb=spk_emb,
            num_steps=8,
            solver="midpoint",
            cfg_scale=1.8
        )
        self.assertEqual(latents.shape, (1, 10, 128))
        self.assertFalse(np.isnan(latents).any())

    def test_watermark_audioseal(self):
        """Verify imperceptible AudioSeal embedding, SNR >= 38 dB, and detector accuracy."""
        t = np.linspace(0, 1.5, int(self.sr * 1.5), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

        watermarked_audio, meta = self.watermarker.embed_watermark(audio, target_snr_db=40.0)
        self.assertEqual(len(watermarked_audio), len(audio))
        self.assertGreaterEqual(meta["snr_db"], 38.0)

        # Detect watermark
        det = self.watermarker.detect_watermark(watermarked_audio)
        self.assertTrue(det["is_watermarked"])
        self.assertGreater(det["confidence"], 0.6)

    def test_consent_verification_gate(self):
        """Verify spoken consent validation and refusal of mismatched vocal biometric signatures."""
        checker = SpeakerConsentChecker(self.speaker_encoder, similarity_threshold=0.75)
        
        t = np.linspace(0, 2.0, self.sr * 2, endpoint=False)
        user_voice = (0.4 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
        imposter_voice = (0.4 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

        # Matching voice + correct phrase
        res_valid = checker.verify_vocal_consent(
            reference_audio=user_voice,
            consent_audio=user_voice,
            consent_phrase_spoken=checker.MANDATORY_CONSENT_PHRASE
        )
        self.assertTrue(res_valid["verified"])

        # Mismatched imposter voice -> Must Reject
        res_imposter = checker.verify_vocal_consent(
            reference_audio=user_voice,
            consent_audio=imposter_voice,
            consent_phrase_spoken=checker.MANDATORY_CONSENT_PHRASE
        )
        self.assertFalse(res_imposter["verified"])

        # Mismatched phrase -> Must Reject
        res_wrong_phrase = checker.verify_vocal_consent(
            reference_audio=user_voice,
            consent_audio=user_voice,
            consent_phrase_spoken="Hello I want to test this model."
        )
        self.assertFalse(res_wrong_phrase["verified"])

    def test_data_filter_pipeline(self):
        """Verify SNR and quality threshold filtering."""
        pipe = QualityFilterPipeline(min_snr_db=15.0, min_quality_score=3.0)
        clean_audio = (0.4 * np.sin(2 * np.pi * 220 * np.linspace(0, 2.0, self.sr * 2))).astype(np.float32)
        res = pipe.evaluate_clip(clean_audio, self.sr, transcript="clean audio")
        self.assertTrue(res["accepted"])


    def test_objective_evaluator(self):
        """Verify WER, CER, SIM-o, and UTMOS evaluation."""
        evaluator = ObjectiveEvaluator(self.speaker_encoder)
        t = np.linspace(0, 1.0, self.sr, endpoint=False)
        audio = (0.4 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

        metrics = evaluator.evaluate_sample(
            generated_audio=audio,
            reference_audio=audio,
            target_text="test voice cloning",
            asr_hypothesis_text="test voice cloning",
            generation_time_sec=0.2,
            sample_rate=self.sr
        )
        self.assertEqual(metrics["wer"], 0.0)
        self.assertAlmostEqual(metrics["sim_o"], 1.0, places=2)
        self.assertGreater(metrics["utmos_score"], 4.0)


if __name__ == "__main__":
    unittest.main()
