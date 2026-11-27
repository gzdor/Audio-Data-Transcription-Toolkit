#!/usr/bin/env python3
"""
Command-line interface for the Audio Data Transcription Toolkit: scalable
multi-recording audio/video transcription (.wav, .mp3, .mp4, etc.), hardware probing
(CPU / NVIDIA CUDA / AMD ROCm & DirectML), speech enhancement, conversation report
(.docx) generation, and GUI launching.
"""

import argparse
import os
import sys
from .audio_io import probe_media_file
from .hardware import detect_hardware_profile
from .postprocess import DEFAULT_TECHNICAL_REPLACEMENTS, load_custom_replacements
from .transcribe import DEFAULT_INITIAL_PROMPT, TranscriberEngine


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Scalable multi-format audio & video transcription and conversation report toolkit "
            "(Multi-CPU, NVIDIA CUDA, AMD ROCm/DirectML, Batch Processing, Timing Telemetry & GUI)."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="One or more audio/video files (.wav, .mp3, .mp4, .m4a, .flac, .ogg, .mkv, .mov, .webm), directories, or glob patterns",
    )
    parser.add_argument(
        "-o",
        "--output-txt",
        default=None,
        help="Explicit output .txt path (when transcribing a single file)",
    )
    parser.add_argument(
        "-d",
        "--output-dir",
        default=None,
        help="Output directory for batch transcription deliverables (defaults to each input file's directory)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively scan input directories for supported audio and video files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files whose <stem>-Transcript.txt already exists in the output directory",
    )
    parser.add_argument(
        "--model",
        default="base.en",
        help="Whisper model size or path (default: base.en; options: tiny.en, base.en, small.en, medium.en, large-v3-turbo)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "rocm", "directml"],
        help="Execution device (default: auto — probes NVIDIA CUDA, AMD ROCm/DirectML, then multi-core CPU)",
    )
    parser.add_argument(
        "--compute-type",
        default="auto",
        choices=["auto", "int8", "float16", "int8_float16", "float32"],
        help="Quantization / precision compute type (default: auto)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel CTranslate2 workers for multi-chunk / multi-file scaling (default: auto from CPU/GPU topology)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Number of intra-op CPU threads per worker (default: auto from physical core count)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for BatchedInferencePipeline (default: auto — 16 on CPU, 24 on GPU)",
    )
    parser.add_argument(
        "--enhance",
        action="store_true",
        help="Enable O(1)-memory High-Pass + 50 ms Block-RMS AGC speech enhancement for far-field / across-table voices",
    )
    parser.add_argument(
        "--mix-streams",
        action="store_true",
        help="Mix all audio tracks when transcribing multi-track video containers (.mp4/.mkv)",
    )
    parser.add_argument(
        "--docx",
        dest="docx",
        action="store_true",
        default=True,
        help="Automatically generate a formatted Word (.docx) Executive Summary & Conversation Report for each recording (enabled by default)",
    )
    parser.add_argument(
        "--no-docx",
        dest="docx",
        action="store_false",
        help="Disable automatic Word (.docx) Executive Summary & Conversation Report generation",
    )
    parser.add_argument(
        "--srt",
        action="store_true",
        help="Export SubRip (.srt) subtitle file for each recording",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Export per-file structured JSON transcript and stage timing telemetry",
    )
    parser.add_argument(
        "--telemetry-json",
        default=None,
        help="Write aggregate batch timing benchmark summary to the specified JSON path",
    )
    parser.add_argument(
        "--replacements-json",
        default=None,
        help="Optional JSON file containing custom regex terminology replacement pairs",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_INITIAL_PROMPT,
        help="Domain vocabulary initial prompt for Whisper conditioning",
    )
    parser.add_argument(
        "--probe-hardware",
        action="store_true",
        help="Probe and display CPU / NVIDIA CUDA / AMD ROCm & DirectML hardware profile and exit",
    )
    parser.add_argument(
        "--probe-media",
        action="store_true",
        help="Probe and display audio/video container, codec, sample rate, and stream info for input files and exit",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the Desktop Graphical User Interface (tkinter/ttk)",
    )
    parser.add_argument(
        "--web",
        action="store_true",
        help="Launch the Local Browser Web GUI on http://127.0.0.1:8765",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for --web GUI (default: 8765)",
    )

    args = parser.parse_args()

    if args.probe_hardware:
        hw = detect_hardware_profile(
            preferred_device=args.device,
            preferred_compute_type=args.compute_type,
            user_threads=args.threads,
            user_workers=args.workers,
            user_batch_size=args.batch_size,
        )
        print(hw.summary_line())
        for note in hw.notes:
            print(f"  * {note}")
        return

    if args.probe_media:
        for p in args.inputs:
            info = probe_media_file(p)
            print(f"{os.path.basename(p)}: {info.summary_str()} | Duration: {info.duration_sec:.2f}s")
        return

    if args.gui or args.web:
        from .gui import launch_desktop_gui, launch_web_gui

        if args.web:
            launch_web_gui(port=args.port, initial_files=args.inputs)
        else:
            try:
                import tkinter as tk
                r = tk.Tk()
                r.destroy()
                launch_desktop_gui(initial_files=args.inputs)
            except Exception as e:
                print(f"[GUI] Desktop display unavailable ({e}); launching Web GUI on port {args.port}...")
                launch_web_gui(port=args.port, initial_files=args.inputs, open_browser=False)
        return

    if not args.inputs:
        parser.print_help()
        sys.exit(1)

    replacements = (
        load_custom_replacements(args.replacements_json)
        if args.replacements_json
        else DEFAULT_TECHNICAL_REPLACEMENTS
    )

    engine = TranscriberEngine(
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.threads,
        num_workers=args.workers,
        batch_size=args.batch_size,
        enhance_speech=args.enhance,
        mix_all_audio_streams=args.mix_streams,
        initial_prompt=args.prompt,
        replacements=replacements,
    )

    if len(args.inputs) == 1 and os.path.isfile(args.inputs[0]) and args.output_txt:
        engine.transcribe_file(
            audio_path=args.inputs[0],
            output_dir=args.output_dir,
            output_txt_path=args.output_txt,
            generate_docx=args.docx,
            generate_srt=args.srt,
            generate_json=args.json,
        )
    else:
        engine.transcribe_batch(
            inputs=args.inputs,
            output_dir=args.output_dir,
            recursive=args.recursive,
            skip_existing=args.skip_existing,
            generate_docx=args.docx,
            generate_srt=args.srt,
            generate_json=args.json,
            telemetry_json_path=args.telemetry_json,
        )


if __name__ == "__main__":
    main()
