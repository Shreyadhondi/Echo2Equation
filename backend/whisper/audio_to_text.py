# backend/whisper/audio_to_text.py
from __future__ import annotations
import os
import sys
import tempfile
from datetime import datetime
from functools import lru_cache
from typing import Tuple, Optional
import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "base.en"             # Model we decided to use
DOWNLOAD_ROOT = "models/whisper"   # Cached model path
SAVE_DIR = "data/test_recordings"  # Permanent save location for test recordings

os.makedirs(SAVE_DIR, exist_ok=True)  # Ensure folder exists


@lru_cache(maxsize=1)
def _get_model() -> WhisperModel:
    """Load and cache the Whisper model."""
    return WhisperModel(
        MODEL_NAME,
        device="cpu",
        compute_type="int8",
        download_root=DOWNLOAD_ROOT,
    )


def record_until_enter(
    out_path: Optional[str] = None,
    samplerate: int = 16000,
    channels: int = 1,
) -> str:
    """
    Record from microphone until Enter is pressed.
    Saves audio to out_path (WAV) and returns the path.
    """
    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=".wav", prefix="echo2eq_")
        os.close(fd)

    print("Press ENTER to start recording...")
    input()
    print("🎙️ Recording... Press ENTER to stop.")
    
    rec_data = []

    def callback(indata, frames, time, status):
        if status:
            print(status, file=sys.stderr)
        rec_data.append(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        callback=callback,
    )

    with stream:
        input()  # Wait for Enter to stop
    print("⏹️ Recording stopped.")

    # Combine chunks and save
    audio_np = np.concatenate(rec_data, axis=0)
    sf.write(out_path, audio_np, samplerate, subtype="PCM_16")
    print(f"💾 Temp saved recording → {out_path}")

    # Save a permanent copy with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(SAVE_DIR, f"recording_{timestamp}.wav")
    sf.write(save_path, audio_np, samplerate, subtype="PCM_16")
    print(f"📂 Permanent copy saved → {save_path}")

    return out_path


def transcribe_file(path: str, beam_size: int = 5) -> Tuple[str, float, str]:
    """Transcribe an audio file using Whisper."""
    model = _get_model()
    segments, info = model.transcribe(
        path,
        beam_size=beam_size,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
    )
    text = "".join(seg.text for seg in segments).strip()
    return text, float(info.duration or 0.0), str(info.language or "en")


def transcribe_live_interactive(
    samplerate: int = 16000,
    beam_size: int = 5,
    keep_temp_wav: bool = False,
) -> Tuple[str, float, str]:
    """
    Record from mic interactively (Enter to start/stop) and transcribe.
    Always keeps a copy in SAVE_DIR; temp file optionally removed.
    """
    tmp_path = None
    try:
        tmp_path = record_until_enter(samplerate=samplerate)
        text, dur, lang = transcribe_file(tmp_path, beam_size=beam_size)
        return text, dur, lang
    finally:
        if tmp_path and not keep_temp_wav:
            try:
                os.remove(tmp_path)
            except Exception:
                pass


# -------------------------------
# CLI usage
# -------------------------------
if __name__ == "__main__":
    try:
        text, dur, lang = transcribe_live_interactive(keep_temp_wav=False)
        print("\n📝 Transcription:")
        print(text if text else "[EMPTY]")
        print(f"\nℹ️ Detected language: {lang} | Duration: {dur:.2f}s")
    except KeyboardInterrupt:
        print("\n❌ Recording cancelled.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Hint: pip install sounddevice soundfile ; and install portaudio19-dev on Ubuntu.")

#test
#LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 venv/bin/python backend/whisper/audio_to_text.py
#