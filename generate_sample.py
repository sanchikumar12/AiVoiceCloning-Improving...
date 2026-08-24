"""
Generate audio speech sample.
"""

import os
import sys
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from inference.pipeline import VoiceCloner

def generate():
    cloner = VoiceCloner()

    # Create reference audio with distinct rich timbre
    sr = 24000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    ref_voice = (
        0.4 * np.sin(2 * np.pi * 125.0 * t) +
        0.25 * np.sin(2 * np.pi * 250.0 * t) +
        0.12 * np.sin(2 * np.pi * 375.0 * t)
    ).astype(np.float32)

    cloner.register_voice("studio_speaker", ref_voice, bypass_consent=True)

    text_to_speak = (
        "Hello! It is truly amazing how realistic and emotional voice cloning has become. Every nuance, every breath, and every feeling comes through just like a real person talking to you."
    )
    print(f"Synthesizing: \"{text_to_speak}\"")

    res = cloner.clone_voice(
        text=text_to_speak,
        voice_id="studio_speaker",
        emotion_style="conversational",
        temperature=0.75,
        ode_steps=16,
        apply_watermark=True
    )


    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "speech_sample.wav")
    sf.write(out_path, res["audio"], res["sample_rate"])

    print("\n[SUCCESS] Audio file generated successfully!")
    print(f"Path: {out_path}")
    print(f"Duration: {res['duration_sec']:.2f} seconds")
    print(f"Sample Rate: {res['sample_rate']} Hz")
    print(f"Watermark SNR: {res['watermark_meta']['snr_db']:.1f} dB")

if __name__ == "__main__":
    generate()
