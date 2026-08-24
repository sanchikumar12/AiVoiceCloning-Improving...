"""
FastAPI Serving Application for Zero-Shot Voice Cloning System.
Exposes REST and Streaming APIs for voice registration, consent verification,
speech synthesis, and AudioSeal watermark inspection.
"""

import base64
import io
import os
import tempfile
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import soundfile as sf
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from ..inference.pipeline import VoiceCloner
    from ..inference.consent_check import ConsentVerificationError
    from ..inference.streaming_server import StreamingVoiceSynthesizer
except (ImportError, ValueError):
    from inference.pipeline import VoiceCloner
    from inference.consent_check import ConsentVerificationError
    from inference.streaming_server import StreamingVoiceSynthesizer


# Initialize FastAPI App
app = FastAPI(
    title="ElevenLabs-Class Zero-Shot Voice Cloning API",
    description="Production-grade 2-stage AR + Flow-Matching TTS system with AudioSeal watermarking & consent gating.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load master configuration
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
config = {}
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

# Initialize Cloner pipeline
cloner = VoiceCloner(config=config)
streamer = StreamingVoiceSynthesizer(cloner)

# Request Models
class CloneRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    emotion_style: Optional[str] = "conversational"
    gender_tone: Optional[str] = "auto"
    temperature: float = 0.75
    top_p: float = 0.92
    repetition_penalty: float = 1.25
    ode_steps: int = 16
    apply_watermark: bool = True


class ConsentRequest(BaseModel):
    consent_phrase_spoken: str


def audio_to_base64_wav(audio_data: np.ndarray, sample_rate: int) -> str:
    """Convert float numpy array to base64-encoded WAV buffer."""
    buf = io.BytesIO()
    sf.write(buf, audio_data, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def decode_uploaded_audio(upload_file: UploadFile, default_sr: int = 24000) -> Tuple[np.ndarray, int]:
    """
    Bulletproof universal audio decoder supporting ALL containers and codecs:
    M4A, AAC, MP3, WAV, FLAC, OGG, OPUS, WebM, MP4, WMA, AIFF.
    """
    upload_file.file.seek(0)
    file_bytes = upload_file.file.read()
    if not file_bytes:
        raise ValueError("Uploaded audio file is empty.")

    raw_filename = getattr(upload_file, "filename", "audio.wav") or "audio.wav"
    ext = os.path.splitext(raw_filename)[1].lower() or ".wav"

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Tier 1: PyAV (FFmpeg engine) - decodes M4A, AAC, MP3, WAV, FLAC, OGG, WebM, MP4
        try:
            import av
            container = av.open(tmp_path)
            frames = []
            file_sr = default_sr
            for frame in container.decode(audio=0):
                file_sr = frame.sample_rate
                arr = frame.to_ndarray()
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=0)
                frames.append(arr.astype(np.float32))
            container.close()

            if frames:
                data = np.concatenate(frames)
                max_abs = float(np.max(np.abs(data)))
                if max_abs > 2.0:
                    data = data / (32768.0 if max_abs <= 32768.0 else 2147483648.0)

                if file_sr != default_sr:
                    from scipy.signal import resample_poly
                    import math
                    gcd = math.gcd(default_sr, file_sr)
                    data = resample_poly(data, default_sr // gcd, file_sr // gcd).astype(np.float32)

                return data, default_sr
        except Exception:
            pass

        # Tier 2: miniaudio in-memory decode
        try:
            import miniaudio
            decoded = miniaudio.decode(file_bytes, nchannels=1, sample_rate=default_sr)
            samples = np.array(decoded.samples, dtype=np.float32)
            if decoded.sample_width == 2:
                samples = samples / 32768.0
            elif decoded.sample_width == 4:
                samples = samples / 2147483648.0
            return samples, decoded.sample_rate
        except Exception:
            pass

        # Tier 3: soundfile fallback
        data, sr = sf.read(tmp_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        return data, sr

    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass




@app.get("/health")
async def health_check():
    """Health and system metrics check."""
    return {
        "status": "healthy",
        "service": "voice-clone-engine",
        "registered_voices": list(cloner.speaker_cache.keys()),
        "sample_rate": cloner.codec.sample_rate,
        "causal_codec": cloner.codec.causal,
        "watermark_enabled": cloner.watermarker is not None
    }


@app.post("/api/register_voice")
async def register_voice(
    voice_id: str = Form(...),
    reference_file: UploadFile = File(...),
    consent_file: Optional[UploadFile] = File(None),
    consent_phrase: Optional[str] = Form(None),
    bypass_consent: bool = Form(False)
):
    """
    Register a new voice profile from uploaded reference audio and consent verification.
    Supports WAV, MP3, M4A, OGG, FLAC, AAC, WebM.
    """
    try:
        # Read reference audio with universal format decoder
        ref_data, sr = decode_uploaded_audio(reference_file, default_sr=cloner.codec.sample_rate)

        consent_data = None
        if consent_file:
            consent_data, _ = decode_uploaded_audio(consent_file, default_sr=cloner.codec.sample_rate)

        bypass_flag = str(bypass_consent).lower() in ("true", "1", "yes")

        res = cloner.register_voice(
            voice_id=voice_id,
            reference_audio=ref_data,
            consent_audio=consent_data,
            consent_phrase=consent_phrase,
            bypass_consent=bypass_flag
        )

        return res
    except ConsentVerificationError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/api/clone")
def clone_voice_endpoint(req: CloneRequest):
    """
    Synthesize speech for input text matching the target voice identity.
    """
    try:
        # Default voice creation if cache is empty
        if not cloner.speaker_cache and req.voice_id is None:
            # Generate a clean synthetic reference voice profile
            t = np.linspace(0, 3.0, 24000 * 3, endpoint=False)
            dummy_ref = (np.sin(2 * np.pi * 140 * t) * 0.4 + np.sin(2 * np.pi * 280 * t) * 0.2).astype(np.float32)
            cloner.register_voice("default_voice", dummy_ref, bypass_consent=True)
            voice_id_to_use = "default_voice"
        else:
            voice_id_to_use = req.voice_id or list(cloner.speaker_cache.keys())[0]

        result = cloner.clone_voice(
            text=req.text,
            voice_id=voice_id_to_use,
            emotion_style=req.emotion_style or "conversational",
            gender_tone=req.gender_tone or "auto",
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
            ode_steps=req.ode_steps,
            apply_watermark=req.apply_watermark
        )


        # Convert waveform to Base64 WAV
        b64_audio = audio_to_base64_wav(result["audio"], result["sample_rate"])

        return {
            "status": "success",
            "audio_base64": b64_audio,
            "duration_sec": result["duration_sec"],
            "normalized_text": result["normalized_text"],
            "num_semantic_tokens": result["num_semantic_tokens"],
            "watermark_meta": result["watermark_meta"],
            "profiling": result["profiling"]
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/detect_watermark")
async def detect_watermark_endpoint(audio_file: UploadFile = File(...)):
    """
    Inspect an uploaded audio file for AudioSeal watermarks and tamper localization.
    """
    try:
        audio_data, sr = decode_uploaded_audio(audio_file, default_sr=cloner.codec.sample_rate)

        det_result = cloner.watermarker.detect_watermark(audio_data)
        return {
            "is_ai_generated": det_result["is_watermarked"],
            "confidence_score": det_result["confidence"],
            "recovered_payload": det_result["recovered_payload"],
            "sample_rate": sr,
            "duration_sec": len(audio_data) / sr
        }
    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


# Mount UI & Outputs static folders
UI_DIR = os.path.join(os.path.dirname(__file__), "..", "ui")
OUTPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

if os.path.exists(UI_DIR):
    app.mount("/ui", StaticFiles(directory=UI_DIR, html=True), name="ui")
    
    @app.get("/")
    async def index():
        return FileResponse(os.path.join(UI_DIR, "index.html"))

