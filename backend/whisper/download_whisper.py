#!/usr/bin/env python3
"""
Pre-download a Whisper model (faster-whisper) so it’s ready for offline use.

WHAT THIS DOES
--------------
- Initializes a faster-whisper WhisperModel once.
- This triggers an automatic download of model files (if not already cached).
- Prints where the model is stored on disk and basic model info.

WHY THIS IS USEFUL
------------------
- Avoids the first-run latency during your app’s /asr calls.
- Lets you control the download location (e.g., inside your repo).
- Confirms your environment can load the model successfully on CPU.

REQUIREMENTS
------------
    pip install faster-whisper

(You do NOT need ffmpeg just to download the model. ffmpeg is only needed
for decoding/transcoding audio files when you actually transcribe.)

USAGE EXAMPLES
--------------
    # Download the recommended CPU model (English-only), using int8:
    python backend/whisper/download_whisper.py

    # Download small.en (better accuracy, slower on CPU), cache under ./models/whisper:
    python backend/whisper/download_whisper.py --model small.en --out-dir models/whisper

    # Reuse default cache (~/.cache/ctranslate2) but verbose logs:
    python backend/whisper/download_whisper.py --verbose

NOTES
-----
- Faster-whisper caches models under ~/.cache/ctranslate2 by default.
- You can override with --out-dir to keep models inside your project folder
  (handy for Docker builds or air-gapped machines after first download).
"""

import argparse
import os
from pathlib import Path
from faster_whisper import WhisperModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-download a faster-whisper model for CPU/GPU use."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base.en",   # good balance for CPU-only English
        help="Model size to download: tiny(.en), base(.en), small(.en), medium, large-v3, etc. (default: base.en)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device for initialization (cpu or cuda). Only affects init time; download is the same.",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help=(
            "Compute type for initialization. CPU tips: int8 (fastest), int8_float16 (balanced), float32 (slow). "
            "GPU tips: float16 or int8_float16."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Optional directory to store/download the model (overrides default cache at ~/.cache/ctranslate2). "
            "Example: models/whisper"
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra information."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve download/cache directory (if provided)
    download_root = None
    if args.out_dir:
        download_root = Path(args.out_dir).expanduser().resolve()
        download_root.mkdir(parents=True, exist_ok=True)

    print("===========================================")
    print(" Whisper Model Pre-Downloader (faster-whisper)")
    print("===========================================")
    print(f"• Model name     : {args.model}")
    print(f"• Device         : {args.device}")
    print(f"• Compute type   : {args.compute_type}")
    print(f"• Download root  : {download_root if download_root else '~/.cache/ctranslate2 (default)'}")
    print("• Action         : Initialize model once to trigger download/cache")
    print("")

    try:
        # Initializing WhisperModel triggers the download if not present.
        # After the first run, files are reused from cache/download_root.
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            download_root=str(download_root) if download_root else None,
        )

        # Grab some metadata by doing a zero-cost call (no audio) via model attributes.
        # Note: Not all attrs are public; keep this minimal and robust.
        print("✅ Model initialized successfully.")
        if args.verbose:
            # The model object exposes internal info like transcribe options when running,
            # but we avoid accessing private internals. We just confirm init and cache path.
            pass

        # Best-effort: print where the model ended up on disk.
        # Faster-whisper stores models in subfolders named like: "<model>-ct2"
        # We’ll search typical locations.
        potential_roots = [
            download_root if download_root else Path.home() / ".cache" / "ctranslate2"
        ]

        found_paths = []
        for root in potential_roots:
            if not root:
                continue
            root = Path(root)
            if root.exists():
                for p in root.rglob("*"):
                    # Heuristic: directories containing model.bin / tokenizer.json etc.
                    if p.is_dir() and any((p / fname).exists() for fname in ["model.bin", "tokenizer.json", "vocabulary.txt", "config.json"]):
                        if args.model.replace(".en", "") in p.name or "whisper" in p.name.lower() or "ct2" in p.name.lower():
                            found_paths.append(p)
        if found_paths:
            print("📦 Cached model directories (potential matches):")
            for p in sorted(set(found_paths)):
                print(f"   - {p}")
        else:
            print("ℹ Could not enumerate cache paths reliably, but initialization succeeded. "
                  "Your model is cached under the chosen root.")

        print("\nYou’re all set! Next step: call transcribe() from your ASR code.")
        print("Tip: On CPU, start with: device='cpu', compute_type='int8', model='base.en'.")

    except Exception as e:
        print("❌ Failed to initialize/download the model.")
        print(f"   Error: {e}")
        print("\nCommon fixes:")
        print("  • Check your internet connection (first run needs to download weights).")
        print("  • If behind a proxy/firewall, configure environment (e.g., HTTPS_PROXY).")
        print("  • Try another model size (e.g., tiny.en) or omit --out-dir.")
        print("  • Ensure Python has write permissions to the cache/out directory.")
        raise


if __name__ == "__main__":
    main()
