#!/usr/bin/env python3
"""
Unit test suite for audio_transcriber:
  - Multi-format audio/video probing and 16 kHz mono float32 resampling (.wav, .mp3, .m4a, .flac, .ogg, .mp4)
  - O(1)-memory High-Pass + 50 ms Block-RMS AGC speech enhancement
  - Pre-compiled regex post-processing & dialogue grouping
  - Automated Word (.docx) Executive Summary & Conversation Report generation
"""

import os
import sys
import tempfile
import unittest
import av
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from audio_transcriber import (
    SUPPORTED_MEDIA_EXTENSIONS,
    build_auto_docx_report,
    clean_text,
    detect_hardware_profile,
    enhance_audio_fast,
    group_transcript_segments,
    load_and_resample_audio,
    probe_media_file,
    resolve_audio_inputs,
)


class TestAudioTranscriberToolkit(unittest.TestCase):
    def test_hardware_detection(self):
        hw = detect_hardware_profile()
        self.assertGreaterEqual(hw.logical_cores, 1)
        self.assertGreaterEqual(hw.physical_cores, 1)
        self.assertIn(hw.recommended_device, ("cpu", "cuda", "rocm", "directml"))

    def test_enhance_audio_fast_bounds_and_nan_safety(self):
        sr = 16000
        t = np.linspace(0, 3.0, sr * 3, dtype=np.float32)
        quiet_tone = 0.01 * np.sin(2 * np.pi * 440.0 * t)
        quiet_tone[100] = np.nan
        quiet_tone[200] = np.inf
        enhanced = enhance_audio_fast(quiet_tone.copy(), sr=sr)
        self.assertEqual(enhanced.dtype, np.float32)
        self.assertEqual(len(enhanced), len(quiet_tone))
        self.assertTrue(np.all(np.isfinite(enhanced)))
        self.assertLessEqual(float(np.max(np.abs(enhanced))), 0.96)
        self.assertGreater(float(np.max(np.abs(enhanced))), 0.10)

    def test_multi_format_audio_and_video_decoding(self):
        sr = 48000
        t = np.linspace(0, 1.5, int(sr * 1.5), dtype=np.float32)
        tone = (0.25 * np.sin(2 * np.pi * 440.0 * t) * 32767).astype(np.int16)

        with tempfile.TemporaryDirectory() as tmp:
            for ext, fmt, codec in [
                (".wav", "wav", "pcm_s16le"),
                (".mp3", "mp3", "libmp3lame"),
                (".m4a", "ipod", "aac"),
                (".flac", "flac", "flac"),
                (".ogg", "ogg", "libopus"),
                (".mp4", "mp4", "aac"),
            ]:
                fpath = os.path.join(tmp, f"sample{ext}")
                with av.open(fpath, "w", format=fmt) as out:
                    v_stream = None
                    if ext == ".mp4":
                        v_stream = out.add_stream("mpeg4", rate=15)
                        v_stream.width = 160
                        v_stream.height = 120
                        v_stream.pix_fmt = "yuv420p"
                    a_stream = out.add_stream(codec, rate=sr)
                    a_stream.layout = "mono"
                    if v_stream is not None:
                        for _ in range(15):
                            img = np.zeros((120, 160, 3), dtype=np.uint8)
                            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
                            for pkt in v_stream.encode(vf):
                                out.mux(pkt)
                        for pkt in v_stream.encode(None):
                            out.mux(pkt)
                    af = av.AudioFrame.from_ndarray(tone.reshape(1, -1), format="s16", layout="mono")
                    af.sample_rate = sr
                    for pkt in a_stream.encode(af):
                        out.mux(pkt)
                    for pkt in a_stream.encode(None):
                        out.mux(pkt)

                info = probe_media_file(fpath)
                self.assertGreaterEqual(info.audio_streams, 1)
                audio_16k, _, corrupt = load_and_resample_audio(fpath, sampling_rate=16000)
                self.assertEqual(audio_16k.dtype, np.float32)
                self.assertEqual(corrupt, 0)
                self.assertGreater(len(audio_16k), 20000)

            # Verify non-media files are ignored by resolve_audio_inputs
            open(os.path.join(tmp, "readme.txt"), "w").close()
            resolved = resolve_audio_inputs([os.path.join(tmp, "*")])
            self.assertEqual(len(resolved), 6)

    def test_auto_executive_summary_docx_generation(self):
        segments = [
            {"start": 0.0, "end": 12.4, "text": "Welcome to the architecture review covering multi-format audio transcription and signal processing."},
            {"start": 12.5, "end": 28.0, "text": "High-pass filtering and block-strided automatic gain control improve far-field speech recognition accuracy."},
            {"start": 28.5, "end": 45.0, "text": "Next steps include deploying the GPU batch pipeline across our recording archives."},
        ]
        blocks = group_transcript_segments(segments)
        timings = {
            "file_size_mb": 14.2,
            "container_info": "MP4 [aac, 48000 Hz, 2 ch]",
            "audio_duration_sec": 45.0,
            "speech_duration_sec": 42.0,
            "decode_sec": 0.12,
            "enhance_sec": 0.03,
            "model_load_sec": 0.40,
            "inference_sec": 1.25,
            "postprocess_sec": 0.01,
            "deliverables_sec": 0.05,
            "total_wall_sec": 1.86,
            "speedup_x_rt": 24.19,
            "rtf": 0.0413,
            "decode_pct": 6.5,
            "enhance_pct": 1.6,
            "model_load_pct": 21.5,
            "inference_pct": 67.2,
            "postprocess_pct": 0.5,
        }
        with tempfile.TemporaryDirectory() as tmp:
            docx_out = os.path.join(tmp, "Executive-Report.docx")
            build_auto_docx_report(
                output_docx_path=docx_out,
                audio_path="/recordings/design_review.mp4",
                segments=segments,
                grouped_blocks=blocks,
                timings=timings,
                hardware_summary="CPU: 8 physical / 16 logical cores",
            )
            self.assertTrue(os.path.exists(docx_out))
            self.assertGreater(os.path.getsize(docx_out), 5000)


if __name__ == "__main__":
    unittest.main()

