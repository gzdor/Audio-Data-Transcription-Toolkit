#!/usr/bin/env python3
"""
Example 2: Programmatic creation of a styled Word (.docx) Executive Conversation Report
using MeetingReportBuilder.
"""

import os
import sys

# Ensure 'src' is importable when running directly from a downloaded ZIP archive
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from audio_transcriber import MeetingReportBuilder


def build_sample_report(output_path: str = "Sample-Architecture-Sync-Report.docx"):
    builder = MeetingReportBuilder()

    builder.add_title("System Architecture & Signal Processing Technical Sync")
    builder.add_subtitle("Executive Conversation Summary, Chronological Topic Breakdown & Action Items")

    builder.add_metadata_table([
        ("Meeting Topic", "Distributed Sensor Array Processing & Edge Inference Pipeline"),
        ("Duration & Format", "00:45:12 | WAV [pcm_f32le, 48000 Hz, 1 ch]"),
        ("Participants", "Alex Rivera (Systems Lead), Jordan Patel (DSP Engineer), Taylor Kim (ML Architect)"),
        ("Deliverables", "Raw Transcript (.txt), Subtitles (.srt), and Executive Summary (.docx)"),
    ])

    builder.add_h1("1. Executive Summary")
    builder.add_body(
        "The engineering team reviewed end-to-end latency and signal-to-noise ratio (SNR) performance "
        "across the real-time acoustic and vibration sensor pipeline. Key discussion focused on "
        "mitigating low-frequency environmental noise via high-pass filtering and block-strided RMS "
        "automatic gain control before downstream neural network classification."
    )

    builder.add_callout(
        "Block-vectorized Automatic Gain Control (AGC) operating on 60-second chunks keeps peak memory "
        "under 8 MB while boosting far-field speaker clarity by up to +26 dB.",
        "CORE TECHNICAL TAKEAWAY",
    )

    builder.add_h1("2. Chronological Conversation Breakdown")
    builder.add_h2("Section 1 [00:00:00 – 00:15:00] — Sensor Front-End & Noise Floor Characterization")
    builder.add_bullet(
        "Reviewed baseline noise floor measurements and wind-induced micro-perturbation artifacts.",
        bold_prefix="Acoustic Coupling & Shielding: ",
    )
    builder.add_bullet(
        "Confirmed that 130 Hz cumulative-sum high-pass filtering eliminates HVAC rumble prior to gain scaling.",
        bold_prefix="Pre-Emphasis Filtering: ",
    )

    builder.add_h2("Section 2 [00:15:00 – 00:45:12] — Multi-Worker CPU & GPU Batch Scaling")
    builder.add_bullet(
        "Demonstrated producer-consumer pipelining overlapping PyAV media decoding with CTranslate2 inference.",
        bold_prefix="Throughput Optimization: ",
    )

    builder.add_h1("3. Action Items & Next Steps")
    builder.add_styled_table(
        headers=["Action Item", "Owner", "Target Milestone"],
        rows_data=[
            ("Benchmark multi-worker INT8 CPU vs. FP16 GPU batch throughput", "Jordan Patel", "Sprint 14"),
            ("Integrate automated .docx and .json telemetry export into CI", "Taylor Kim", "Sprint 14"),
            ("Publish updated operator documentation and launcher scripts", "Alex Rivera", "Complete"),
        ],
        col_widths_inches=[3.2, 1.8, 1.8],
    )

    builder.save(output_path)
    print(f"Saved sample Word report to: {output_path}")


if __name__ == "__main__":
    build_sample_report()

