#!/usr/bin/env python3
"""
Example 1: Programmatic multi-format batch transcription using TranscriberEngine.
Demonstrates transcribing a mix of .wav, .mp3, .mp4, and .m4a files on auto-detected
hardware (NVIDIA CUDA, AMD ROCm/DirectML, or multi-core CPU) with far-field speech
enhancement (AGC), automated Word (.docx) conversation reports, and stage timing tables.
"""

import os
import sys

# Ensure 'src' is importable when running directly from a downloaded ZIP archive
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from audio_transcriber import TranscriberEngine, detect_hardware_profile


def main():
    # 1. Inspect detected CPU/GPU hardware profile
    hw = detect_hardware_profile()
    print("Detected Hardware Profile:")
    print(" ", hw.summary_line())

    if len(sys.argv) < 2:
        print("\nUsage: python examples/batch_transcribe_example.py <file_or_folder_1> [file_or_folder_2 ...]")
        return

    # 2. Initialize the stateful TranscriberEngine (loads model once across all files)
    engine = TranscriberEngine(
        model_size="base.en",
        device="auto",               # Auto-selects CUDA, ROCm, DirectML, or multi-core CPU
        compute_type="auto",         # Auto-selects int8_float16/float16 on GPU or int8 on CPU
        enhance_speech=True,         # O(1)-memory High-Pass + 50 ms Block-RMS AGC
        mix_all_audio_streams=False, # Set True to mix multiple audio tracks in .mp4/.mkv videos
    )

    # 3. Run batch transcription with producer-consumer pre-decode pipelining
    summary = engine.transcribe_batch(
        inputs=sys.argv[1:],
        recursive=True,
        generate_docx=True,
        generate_srt=True,
        generate_json=True,
    )

    print("\nBatch Complete!")
    print(f"Processed {len(summary.results)} file(s) at {summary.overall_speedup_x_rt:.1f}x Real-Time.")


if __name__ == "__main__":
    main()

