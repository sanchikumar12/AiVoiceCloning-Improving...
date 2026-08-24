# Production-Grade Zero-Shot Voice Cloning System (ElevenLabs-Class)

A modern, production-grade text-to-speech (TTS) and zero-shot voice cloning architecture implementing the state-of-the-art **Two-Stage AR Semantic LM + Conditional Flow-Matching DiT** pipeline with causal neural audio codec representations and imperceptible **AudioSeal** watermarking.

---

## 1. System Architecture

```
                        ┌─────────────────────────┐
Reference audio (3–30s) │   Speaker / Timbre       │
   ──────────────────►  │   Encoder (joint-trained)│──► speaker embedding (global)
                        └─────────────────────────┘         + optional local prosody tokens
                                                                     │
Input text  ──► Text normalizer ──► G2P / BPE tokenizer ──► text tokens
                                                                     │
                                                                     ▼
                    ┌───────────────────────────────────────────────────┐
                    │   Stage 1: Autoregressive semantic-token LM        │
                    │   (decoder-only Transformer, text+speaker→tokens)  │
                    └───────────────────────────────────────────────────┘
                                                                     │  semantic tokens (~12.5–25 Hz)
                                                                     ▼
                    ┌───────────────────────────────────────────────────┐
                    │  Stage 2: Conditional Flow-Matching / DiT decoder  │
                    │  (semantic tokens + speaker emb → acoustic latent) │
                    └───────────────────────────────────────────────────┘
                                                                     │  acoustic latents
                                                                     ▼
                    ┌───────────────────────────────────────────────────┐
                    │   Codec decoder / neural vocoder (causal, fast)    │
                    └───────────────────────────────────────────────────┘
                                                                     │
                                                                     ▼
                                                            24–48 kHz waveform
                                                                     │
                                                                     ▼
                    ┌───────────────────────────────────────────────────┐
                    │  Imperceptible watermark injection (AudioSeal-style)│
                    └───────────────────────────────────────────────────┘
```

---

## 2. Key Architectural Components

### 1. Neural Audio Codec (`codec/`)
- **Low Frame Rate (12.5 Hz @ 24 kHz)**: Minimizes token sequence lengths for efficient autoregressive generation.
- **Split-RVQ Discretization**: First codebook carries distilled semantic/linguistic content; codebooks 1–7 encode acoustic timbre and room acoustics.
- **Causal Architecture**: Fully causal convolutions enable real-time chunked streaming synthesis.

### 2. Speaker & Timbre Conditioning (`speaker_encoder/`)
- **Global Timbre Embedding**: 256-dimensional unit sphere embedding with attentive pooling.
- **Local Prosody Tokens**: 32 cross-attention tokens capture fine-grained intonation and speaking style.
- **Cross-Sample Training**: Trained on distinct utterances from the same speaker to enforce zero-shot generalization.

### 3. Text Frontend (`frontend/`)
- **Locale-Aware Text Normalizer**: Converts numbers, currencies (`$49.99`), dates, acronyms, and units into spoken phonetic form.
- **G2P Phonemizer**: Translates text into IPA phonetic sequences with word boundary markers.
- **Voice Tokenizer**: Manages vocabulary mapping and prompt formatting for Stage 1 input.

### 4. Stage 1: Autoregressive Semantic LM (`stage1_lm/`)
- **Decoder-Only Transformer**: LLaMA-style architecture with RMSNorm, RoPE, and SwiGLU non-linearities.
- **Classifier-Free Guidance (CFG)**: Random condition dropout during training enables runtime control over text adherence and identity stability.
- **Audio Repetition Damping**: Specialized acoustic token penalties prevent looping without degrading sustained vowels.

### 5. Stage 2: Conditional Flow-Matching DiT (`stage2_flow/`)
- **Optimal Transport CFM (OT-CFM)**: Rectified vector field matching trains faster and samples higher quality in 8–32 steps.
- **Fast Midpoint / Euler ODE Solvers**: Rapid probability flow ODE integration.
- **Zero-Shot Infilling**: Continuous span masking trains the model to infill acoustic trajectories in context.

### 6. Safety & Responsible AI (`watermark/` & `inference/consent_check.py`)
- **AudioSeal Watermark Injection**: Embeds a 16-bit imperceptible neural watermark ensuring AI provenance at >40 dB SNR.
- **Localized Tamper Detection**: Sample-level probability maps verify synthetic origin and locate edits.
- **Mandatory Consent Gating**: Validates spoken authorization pledges against uploaded reference voices.
- **Anti-Impersonation & Content Policy**: Filters protected public figures and fraudulent coercion patterns.

---

## 3. Directory Layout

```
voice-clone/
├── configs/
│   └── config.yaml               # Master model & inference hyperparameters
├── codec/
│   ├── model.py                  # Causal Split-RVQ neural codec & vocoder
│   └── train_codec.py            # Multi-scale STFT loss & MPD adversarial training
├── speaker_encoder/
│   ├── model.py                  # ECAPA-TDNN & learnable prosody encoder
│   └── train_speaker.py          # AAM-Softmax & cross-sample contrastive training
├── frontend/
│   ├── normalize.py              # Spoken-form text normalizer
│   ├── g2p.py                    # Grapheme-to-Phoneme converter
│   └── tokenizer.py              # Sequence builder & vocab management
├── stage1_lm/
│   ├── model.py                  # Decoder-only Transformer LM (RoPE, SwiGLU, CFG)
│   └── train_stage1.py           # Teacher-forced next-token CE loss
├── stage2_flow/
│   ├── model.py                  # DiT Flow-Matching decoder & ODE solver
│   └── train_stage2.py           # OT-CFM loss & random-span infilling
├── watermark/
│   └── audioseal_wrapper.py      # AudioSeal watermark generator & detector
├── inference/
│   ├── pipeline.py               # End-to-end VoiceCloner orchestrator
│   ├── streaming_server.py       # Chunked streaming generator (<300ms TTFB)
│   └── consent_check.py          # Vocal biometric consent & policy filter
├── data/
│   ├── ingest/vad.py             # Voice Activity Detection (Silero/Energy)
│   ├── ingest/asr_whisper.py     # Whisper ASR & forced alignment
│   ├── filter/filter_pipeline.py # SNR, UTMOS, and WER quality filter
│   └── datasets/speech_dataset.py# Speech dataset & batch iterator
├── eval/
│   ├── objective_metrics.py      # Objective benchmarks (WER, SIM-o, UTMOS, RTF)
│   └── mos_harness.py            # Subjective human evaluation framework (MOS/CMOS/SMOS)
├── serving/
│   ├── app.py                    # FastAPI server (REST + WebSocket streaming)
│   ├── stage1_server.py          # Batched AR LM serving worker
│   ├── stage2_server.py          # Batched Flow-Matching serving worker
│   └── codec_decode_server.py    # Vocoder worker
├── ui/
│   ├── index.html                # Dark-mode glassmorphic Studio interface
│   ├── style.css                 # Styling & animations
│   └── app.js                    # Web Audio waveform visualizer & client logic
├── tests/
│   └── test_pipeline.py          # Comprehensive test suite
├── run_demo.py                   # Single-command CLI validation runner
└── requirements.txt              # System dependencies
```

---

## 4. Quickstart

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Validation CLI
```bash
python run_demo.py
```

### 3. Launch Interactive Web Studio & REST API
```bash
python -m uvicorn serving.app:app --host 0.0.0.0 --port 8000
```
Open your browser at `http://localhost:8000/` to access the interactive studio.

### 4. Run Test Suite
```bash
python tests/test_pipeline.py
```

---

## 5. REST API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | `GET` | Health status and active models |
| `/api/register_voice` | `POST` | Upload reference audio + spoken consent verification |
| `/api/clone` | `POST` | Zero-shot speech synthesis from text prompt |
| `/api/detect_watermark` | `POST` | Inspect audio file for AudioSeal watermarks |

---

## 6. Responsible AI & Legal Notice

Voice cloning is a powerful dual-use technology. This system implements proactive ethical safeguards:
- **AudioSeal Watermarking**: Enabled by default on all generated audio.
- **Biometric Consent Gate**: Requires matching vocal authorization before cloning personal voices.
- **Anti-Impersonation**: Prevents unauthorized cloning of public figures and extortion prompts.
