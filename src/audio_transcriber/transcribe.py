#!/usr/bin/env python3
"""
Stage 1 & Scalable Batch Engine: Local CPU / NVIDIA CUDA / AMD ROCm & DirectML
ASR inference using faster-whisper (WhisperModel + BatchedInferencePipeline)
with PyTorch GPU fallback for AMD GPUs, multi-format audio & video container
extraction (.wav, .mp3, .mp4, .m4a, .aac, .flac, .ogg, .opus, .wma, .mkv, .mov, .webm, .avi),
O(1)-memory block-vectorized speech enhancement (high-pass + RMS AGC),
intra-file multi-worker chunk parallelism, producer-consumer multi-file batching,
and detailed stage timing telemetry.
"""

import glob
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline

from .audio_io import (
    SUPPORTED_MEDIA_EXTENSIONS,
    MediaProbeInfo,
    load_and_resample_audio,
    probe_media_file,
)
from .hardware import HardwareProfile, detect_hardware_profile
from .postprocess import (
    DEFAULT_TECHNICAL_REPLACEMENTS,
    clean_text,
    compile_replacements,
    export_json_result,
    format_srt,
    format_transcript_sections,
    group_transcript_segments,
    sec_to_hms,
)
from .docx_builder import build_auto_docx_report

DEFAULT_INITIAL_PROMPT = (
    "Technical engineering discussion, software architecture, signal processing, "
    "acoustics, machine learning, hardware testing, action items, milestones, deliverables."
)

# Alias for backward compatibility
SUPPORTED_AUDIO_EXTENSIONS = SUPPORTED_MEDIA_EXTENSIONS


@dataclass
class SimpleSegment:
    start: float
    end: float
    text: str


@dataclass
class StageTimings:
    file_size_mb: float = 0.0
    container_info: str = ""
    audio_duration_sec: float = 0.0
    speech_duration_sec: float = 0.0
    decode_sec: float = 0.0
    enhance_sec: float = 0.0
    model_load_sec: float = 0.0
    inference_sec: float = 0.0
    postprocess_sec: float = 0.0
    deliverables_sec: float = 0.0
    total_wall_sec: float = 0.0

    @property
    def speedup_x_rt(self) -> float:
        return self.audio_duration_sec / self.total_wall_sec if self.total_wall_sec > 0 else 0.0

    @property
    def inference_x_rt(self) -> float:
        return self.audio_duration_sec / self.inference_sec if self.inference_sec > 0 else 0.0

    @property
    def rtf(self) -> float:
        return self.total_wall_sec / self.audio_duration_sec if self.audio_duration_sec > 0 else 0.0

    @property
    def decode_mb_per_sec(self) -> float:
        return self.file_size_mb / self.decode_sec if self.decode_sec > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        tot = max(self.total_wall_sec, 1e-6)
        d = asdict(self)
        d.update({
            "speedup_x_rt": round(self.speedup_x_rt, 2),
            "inference_x_rt": round(self.inference_x_rt, 2),
            "rtf": round(self.rtf, 5),
            "decode_mb_per_sec": round(self.decode_mb_per_sec, 2),
            "decode_pct": round(100.0 * self.decode_sec / tot, 1),
            "enhance_pct": round(100.0 * self.enhance_sec / tot, 1),
            "model_load_pct": round(100.0 * self.model_load_sec / tot, 1),
            "inference_pct": round(100.0 * self.inference_sec / tot, 1),
            "postprocess_pct": round(100.0 * self.postprocess_sec / tot, 1),
            "deliverables_pct": round(100.0 * self.deliverables_sec / tot, 1),
        })
        return d

    def format_table(self, filename: str, segment_count: int, word_count: int) -> str:
        d = self.to_dict()
        lines = [
            "┌" + "─" * 78 + "┐",
            f"│ TIMING & TELEMETRY SUMMARY: {filename[:48]:<48} │",
            "├" + "─" * 42 + "┬" + "─" * 16 + "┬" + "─" * 18 + "┤",
            f"│ {'Pipeline Stage':<40} │ {'Time (sec)':>14} │ {'% / Throughput':>16} │",
            "├" + "─" * 42 + "┼" + "─" * 16 + "┼" + "─" * 18 + "┤",
            f"│ {'1. Media Decode & 16 kHz Resample':<40} │ {self.decode_sec:>13.2f}s │ {d['decode_pct']:>6.1f}% ({self.decode_mb_per_sec:>5.1f}MB/s)│",
            f"│ {'2. High-Pass + RMS AGC Enhancement':<40} │ {self.enhance_sec:>13.2f}s │ {d['enhance_pct']:>15.1f}% │",
            f"│ {'3. Model Load / Initialization':<40} │ {self.model_load_sec:>13.2f}s │ {d['model_load_pct']:>15.1f}% │",
            f"│ {'4. VAD + ASR Inference':<40} │ {self.inference_sec:>13.2f}s │ {d['inference_pct']:>6.1f}% ({self.inference_x_rt:>5.1f}xRT)│",
            f"│ {'5. Regex Post-Processing & Diarization':<40} │ {self.postprocess_sec:>13.2f}s │ {d['postprocess_pct']:>15.1f}% │",
            f"│ {'6. Deliverable Output (.txt/.docx/...)':<40} │ {self.deliverables_sec:>13.2f}s │ {d['deliverables_pct']:>15.1f}% │",
            "├" + "─" * 42 + "┼" + "─" * 16 + "┼" + "─" * 18 + "┤",
            f"│ {'TOTAL END-TO-END WALL TIME':<40} │ {self.total_wall_sec:>13.2f}s │ {self.speedup_x_rt:>9.1f}x RT    │",
            "├" + "─" * 42 + "┴" + "─" * 16 + "┴" + "─" * 18 + "┤",
            f"│ Media: {self.container_info[:69]:<69} │",
            f"│ Audio Duration: {sec_to_hms(self.audio_duration_sec)} ({self.audio_duration_sec:.1f}s) | RTF: {self.rtf:.4f} | Segs: {segment_count} | Words: {word_count:,}".ljust(79) + "│",
            "└" + "─" * 78 + "┘",
        ]
        return "\n".join(lines)


@dataclass
class FileTranscriptionResult:
    audio_path: str
    raw_stream_path: str
    transcript_txt_path: str
    docx_path: Optional[str] = None
    srt_path: Optional[str] = None
    json_path: Optional[str] = None
    segment_count: int = 0
    word_count: int = 0
    timings: StageTimings = field(default_factory=StageTimings)
    segments: List[Dict] = field(default_factory=list)


@dataclass
class BatchTranscriptionSummary:
    hardware: HardwareProfile
    results: List[FileTranscriptionResult] = field(default_factory=list)
    total_batch_wall_sec: float = 0.0

    @property
    def total_audio_duration_sec(self) -> float:
        return sum(r.timings.audio_duration_sec for r in self.results)

    @property
    def overall_speedup_x_rt(self) -> float:
        return (
            self.total_audio_duration_sec / self.total_batch_wall_sec
            if self.total_batch_wall_sec > 0
            else 0.0
        )

    @property
    def overall_rtf(self) -> float:
        return (
            self.total_batch_wall_sec / self.total_audio_duration_sec
            if self.total_audio_duration_sec > 0
            else 0.0
        )

    def format_summary_table(self) -> str:
        lines = [
            "═" * 92,
            f"BATCH TRANSCRIPTION SUMMARY ({len(self.results)} recording(s))",
            f"Hardware: {self.hardware.summary_line()}",
            "─" * 92,
            f"{'File Name':<32} │ {'Audio Dur':>10} │ {'Wall Time':>10} │ {'Speedup':>9} │ {'RTF':>7} │ {'Segs':>6} │ {'Words':>7}",
            "─" * 92,
        ]
        for r in self.results:
            fname = os.path.basename(r.audio_path)[:32]
            t = r.timings
            lines.append(
                f"{fname:<32} │ {sec_to_hms(t.audio_duration_sec):>10} │ {t.total_wall_sec:>9.2f}s │ "
                f"{t.speedup_x_rt:>7.1f}x │ {t.rtf:>7.4f} │ {r.segment_count:>6} │ {r.word_count:>7,}"
            )
        lines.extend([
            "─" * 92,
            f"{'TOTAL / AGGREGATE':<32} │ {sec_to_hms(self.total_audio_duration_sec):>10} │ "
            f"{self.total_batch_wall_sec:>9.2f}s │ {self.overall_speedup_x_rt:>7.1f}x │ "
            f"{self.overall_rtf:>7.4f} │ {sum(r.segment_count for r in self.results):>6} │ "
            f"{sum(r.word_count for r in self.results):>7,}",
            "═" * 92,
        ])
        return "\n".join(lines)


def enhance_audio_fast(
    audio: np.ndarray,
    sr: int = 16000,
    chunk_sec: int = 60,
    hp_window_samples: int = 100,
    rms_block_samples: int = 800,
    target_rms: float = 0.08,
    max_gain: float = 22.0,
) -> np.ndarray:
    """
    O(1)-extra-memory block-vectorized speech enhancement for far-field recordings:
    1) Subtracts a 100-sample (~130 Hz) moving average via cumulative sum to strip
       low-frequency HVAC / table rumble without allocating giant convolution buffers.
    2) Computes 50 ms (800-sample) block RMS envelope and smoothly scales quiet
       across-the-table voices up to `target_rms` (capped at `max_gain`).
    Processes in 60-second chunks (< 8 MB temporary RAM regardless of file length).
    """
    n = len(audio)
    if n == 0:
        return audio

    out = np.empty_like(audio, dtype=np.float32)
    chunk_size = chunk_sec * sr

    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        x = np.nan_to_num(
            audio[start:end].astype(np.float32, copy=True),
            copy=False,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        m = len(x)

        if m > hp_window_samples:
            padded = np.pad(x, (hp_window_samples // 2, hp_window_samples - hp_window_samples // 2), mode="edge")
            csum = np.cumsum(padded, dtype=np.float64)
            low_pass = ((csum[hp_window_samples:] - csum[:-hp_window_samples]) / hp_window_samples).astype(np.float32)
            x_hp = x - low_pass
        else:
            x_hp = x - np.mean(x)

        n_blocks = max(1, m // rms_block_samples)
        trimmed_len = n_blocks * rms_block_samples
        blocks = x_hp[:trimmed_len].reshape(n_blocks, rms_block_samples)
        block_rms = np.sqrt(np.mean(blocks * blocks, axis=1) + 1e-8)

        if n_blocks >= 5:
            k = np.ones(5, dtype=np.float32) / 5.0
            block_rms = np.convolve(block_rms, k, mode="same")

        block_gain = np.clip(target_rms / (block_rms + 1e-8), 0.5, max_gain)
        block_centers = np.linspace(0, m - 1, num=n_blocks, dtype=np.float32)
        sample_idx = np.arange(m, dtype=np.float32)
        gain_env = np.interp(sample_idx, block_centers, block_gain).astype(np.float32)

        np.multiply(x_hp, gain_env, out=out[start:end])
        np.clip(out[start:end], -0.95, 0.95, out=out[start:end])

    return out


def resolve_audio_inputs(inputs: Sequence[str], recursive: bool = False) -> List[str]:
    """Expand a mix of file paths, directory paths, and glob patterns into sorted unique media files."""
    found: List[str] = []
    seen = set()

    for item in inputs:
        expanded_paths = glob.glob(item, recursive=recursive) or [item]
        for path in expanded_paths:
            abs_p = os.path.abspath(path)
            if os.path.isfile(abs_p) and abs_p.lower().endswith(SUPPORTED_MEDIA_EXTENSIONS):
                if abs_p not in seen:
                    seen.add(abs_p)
                    found.append(abs_p)
            elif os.path.isdir(abs_p):
                if recursive:
                    for root, _, files in os.walk(abs_p):
                        for f in sorted(files):
                            if f.lower().endswith(SUPPORTED_MEDIA_EXTENSIONS):
                                fp = os.path.abspath(os.path.join(root, f))
                                if fp not in seen:
                                    seen.add(fp)
                                    found.append(fp)
                else:
                    for f in sorted(os.listdir(abs_p)):
                        if f.lower().endswith(SUPPORTED_MEDIA_EXTENSIONS):
                            fp = os.path.abspath(os.path.join(abs_p, f))
                            if os.path.isfile(fp) and fp not in seen:
                                seen.add(fp)
                                found.append(fp)
    return found


def _split_audio_at_silence_boundaries(
    audio: np.ndarray,
    num_chunks: int,
    sr: int = 16000,
    search_window_sec: float = 15.0,
) -> List[Tuple[float, np.ndarray]]:
    """
    Partition a long 16 kHz audio array into `num_chunks` contiguous slices, choosing split
    points at the quietest 100 ms energy minimum within +/- `search_window_sec` of each boundary
    so words are never cut mid-syllable during multi-worker parallel CPU transcription.
    """
    n = len(audio)
    if num_chunks <= 1 or n < sr * 180:
        return [(0.0, audio)]

    target_step = n // num_chunks
    search_half = int(search_window_sec * sr)
    frame_len = int(0.1 * sr)

    boundaries = [0]
    for i in range(1, num_chunks):
        nominal = i * target_step
        w_start = max(boundaries[-1] + sr * 10, nominal - search_half)
        w_end = min(n - sr * 10, nominal + search_half)
        if w_end <= w_start + frame_len:
            boundaries.append(nominal)
            continue
        window = audio[w_start:w_end]
        n_frames = len(window) // frame_len
        frames = window[: n_frames * frame_len].reshape(n_frames, frame_len)
        energies = np.mean(frames * frames, axis=1)
        min_frame_idx = int(np.argmin(energies))
        split_sample = w_start + min_frame_idx * frame_len
        boundaries.append(split_sample)
    boundaries.append(n)

    chunks: List[Tuple[float, np.ndarray]] = []
    for i in range(len(boundaries) - 1):
        s_idx, e_idx = boundaries[i], boundaries[i + 1]
        if e_idx > s_idx:
            chunks.append((s_idx / float(sr), audio[s_idx:e_idx]))
    return chunks


class TranscriberEngine:
    """
    Stateful, multi-worker transcription engine that loads the ASR model once and scales
    across many recordings and multi-core CPUs / NVIDIA CUDA / AMD ROCm & DirectML GPUs.
    """

    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        compute_type: str = "auto",
        cpu_threads: Optional[int] = None,
        num_workers: Optional[int] = None,
        batch_size: Optional[int] = None,
        chunk_length: int = 20,
        use_batched: bool = True,
        vad_filter: bool = True,
        min_silence_duration_ms: int = 400,
        vad_threshold: float = 0.25,
        enhance_speech: bool = False,
        mix_all_audio_streams: bool = False,
        initial_prompt: Optional[str] = DEFAULT_INITIAL_PROMPT,
        replacements: Sequence[Tuple[str, str]] = DEFAULT_TECHNICAL_REPLACEMENTS,
        log_fn: Optional[Callable[[str], None]] = None,
    ):
        self.model_size = model_size
        self.chunk_length = chunk_length
        self.use_batched = use_batched
        self.vad_filter = vad_filter
        self.min_silence_duration_ms = min_silence_duration_ms
        self.vad_threshold = vad_threshold
        self.enhance_speech = enhance_speech
        self.mix_all_audio_streams = mix_all_audio_streams
        self.initial_prompt = initial_prompt
        self.replacements = replacements
        self.compiled_replacements = compile_replacements(replacements)
        self.log_fn = log_fn or (lambda msg: print(msg, flush=True))

        self.hardware: HardwareProfile = detect_hardware_profile(
            preferred_device=device,
            preferred_compute_type=compute_type,
            user_threads=cpu_threads,
            user_workers=num_workers,
            user_batch_size=batch_size,
        )

        self.model: Optional[WhisperModel] = None
        self.batched_pipeline: Optional[BatchedInferencePipeline] = None
        self.pytorch_pipe: Any = None
        self.model_load_sec: float = 0.0
        self._files_processed: int = 0

    def _log(self, msg: str):
        self.log_fn(msg)

    def ensure_model_loaded(self) -> float:
        """Load the CTranslate2 or PyTorch AMD GPU model if not already loaded."""
        if self.model is not None or self.pytorch_pipe is not None:
            return self.model_load_sec

        t0 = time.perf_counter()
        hw = self.hardware
        self._log(f"[Hardware] {hw.summary_line()}")
        for note in hw.notes:
            self._log(f"[Hardware] {note}")

        if hw.recommended_backend == "pytorch_gpu":
            self._load_pytorch_amd_backend()
        else:
            ct2_device = "cuda" if hw.recommended_device in ("cuda", "rocm") else "cpu"
            self._log(
                f"[Model] Initializing WhisperModel('{self.model_size}', device='{ct2_device}', "
                f"compute_type='{hw.recommended_compute_type}', cpu_threads={hw.recommended_cpu_threads}, "
                f"num_workers={hw.recommended_num_workers})..."
            )
            self.model = WhisperModel(
                self.model_size,
                device=ct2_device,
                compute_type=hw.recommended_compute_type,
                cpu_threads=hw.recommended_cpu_threads,
                num_workers=hw.recommended_num_workers,
            )
            if self.use_batched:
                self.batched_pipeline = BatchedInferencePipeline(model=self.model)

        self.model_load_sec = time.perf_counter() - t0
        self._log(f"[Model] Ready in {self.model_load_sec:.2f}s")
        return self.model_load_sec

    def _load_pytorch_amd_backend(self):
        """Fallback GPU backend for AMD ROCm / DirectML when CTranslate2 is CPU-only."""
        try:
            import torch
            from transformers import pipeline as hf_pipeline

            if self.hardware.recommended_device == "directml":
                import torch_directml
                dev = torch_directml.device()
                dtype = torch.float32
            else:
                dev = "cuda:0"
                dtype = torch.float16

            hf_model_id = (
                self.model_size
                if "/" in self.model_size
                else f"openai/whisper-{self.model_size}"
            )
            self._log(f"[Model] Loading HuggingFace PyTorch Whisper '{hf_model_id}' on AMD GPU ({dev})...")
            self.pytorch_pipe = hf_pipeline(
                "automatic-speech-recognition",
                model=hf_model_id,
                torch_dtype=dtype,
                device=dev,
                chunk_length_s=self.chunk_length,
            )
        except Exception as e:
            self._log(
                f"[Model] PyTorch AMD GPU pipeline unavailable ({e}); falling back to CTranslate2 multi-worker CPU."
            )
            self.hardware.recommended_device = "cpu"
            self.hardware.recommended_backend = "ctranslate2"
            self.hardware.recommended_compute_type = "int8"
            self.model = WhisperModel(
                self.model_size,
                device="cpu",
                compute_type="int8",
                cpu_threads=self.hardware.recommended_cpu_threads,
                num_workers=self.hardware.recommended_num_workers,
            )
            if self.use_batched:
                self.batched_pipeline = BatchedInferencePipeline(model=self.model)

    def _decode_and_prepare(self, audio_path: str) -> Tuple[np.ndarray, MediaProbeInfo, float, float, int]:
        """Decode any audio/video container to 16 kHz mono float32 and optionally run fast block AGC."""
        t_dec0 = time.perf_counter()
        audio, probe_info, corrupt_count = load_and_resample_audio(
            audio_path,
            sampling_rate=16000,
            mix_all_audio_streams=self.mix_all_audio_streams,
        )
        decode_sec = time.perf_counter() - t_dec0

        enhance_sec = 0.0
        if self.enhance_speech:
            t_enh0 = time.perf_counter()
            audio = enhance_audio_fast(audio, sr=16000)
            enhance_sec = time.perf_counter() - t_enh0

        return audio, probe_info, decode_sec, enhance_sec, corrupt_count

    def _transcribe_chunk_ct2(
        self,
        chunk_audio: np.ndarray,
        offset_sec: float,
        on_segment: Optional[Callable[[SimpleSegment], None]] = None,
    ) -> Tuple[List[SimpleSegment], float]:
        """Run CTranslate2 inference on a single audio array slice."""
        hw = self.hardware
        if self.use_batched and self.batched_pipeline is not None:
            segments_iter, info = self.batched_pipeline.transcribe(
                chunk_audio,
                batch_size=hw.recommended_batch_size,
                beam_size=1,
                language="en",
                without_timestamps=False,
                chunk_length=self.chunk_length,
                condition_on_previous_text=False,
                no_speech_threshold=0.8 if self.enhance_speech else 0.6,
                log_prob_threshold=-1.5 if self.enhance_speech else -1.0,
                vad_filter=self.vad_filter,
                vad_parameters=dict(
                    threshold=self.vad_threshold if self.enhance_speech else 0.45,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                    max_speech_duration_s=self.chunk_length,
                ),
                initial_prompt=self.initial_prompt,
            )
        else:
            assert self.model is not None
            segments_iter, info = self.model.transcribe(
                chunk_audio,
                beam_size=1,
                language="en",
                vad_filter=self.vad_filter,
                vad_parameters=dict(
                    threshold=self.vad_threshold if self.enhance_speech else 0.45,
                    min_silence_duration_ms=self.min_silence_duration_ms,
                ),
                condition_on_previous_text=False,
                initial_prompt=self.initial_prompt,
            )

        out_segs: List[SimpleSegment] = []
        for seg in segments_iter:
            txt = seg.text.strip()
            if not txt:
                continue
            s_obj = SimpleSegment(
                start=offset_sec + float(seg.start),
                end=offset_sec + float(seg.end),
                text=txt,
            )
            out_segs.append(s_obj)
            if on_segment is not None:
                on_segment(s_obj)

        speech_dur = float(getattr(info, "duration_after_vad", None) or getattr(info, "duration", 0.0))
        return out_segs, speech_dur

    def transcribe_file(
        self,
        audio_path: str,
        output_dir: Optional[str] = None,
        output_txt_path: Optional[str] = None,
        raw_stream_path: Optional[str] = None,
        generate_docx: bool = True,
        generate_srt: bool = False,
        generate_json: bool = False,
        section_speaker_fn: Optional[Callable[[float, str], Tuple[str, str]]] = None,
        predecoded: Optional[Tuple[np.ndarray, MediaProbeInfo, float, float, int]] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> FileTranscriptionResult:
        """
        Transcribe a single audio or video file with full stage timing breakdown and multi-worker acceleration.
        """
        self.ensure_model_loaded()

        audio_abs = os.path.abspath(audio_path)
        base_dir = os.path.abspath(output_dir) if output_dir else os.path.dirname(audio_abs)
        os.makedirs(base_dir, exist_ok=True)
        stem_name = os.path.splitext(os.path.basename(audio_abs))[0]

        raw_path = raw_stream_path or os.path.join(base_dir, f"{stem_name}_raw_stream.txt")
        txt_path = output_txt_path or os.path.join(base_dir, f"{stem_name}-Transcript.txt")
        docx_path = os.path.join(base_dir, f"{stem_name}-Conversation-Report.docx") if generate_docx else None
        srt_path = os.path.join(base_dir, f"{stem_name}.srt") if generate_srt else None
        json_path = os.path.join(base_dir, f"{stem_name}-Telemetry.json") if generate_json else None

        # Stage 1 & 2: Decode + Optional Speech Enhancement (AGC)
        if predecoded is not None:
            audio, probe_info, decode_sec, enhance_sec, corrupt_count = predecoded
        else:
            self._log(
                f"[Decode] Reading {os.path.basename(audio_abs)} "
                f"(enhance_speech={self.enhance_speech})..."
            )
            audio, probe_info, decode_sec, enhance_sec, corrupt_count = self._decode_and_prepare(audio_abs)

        file_size_mb = probe_info.file_size_mb
        audio_duration_sec = len(audio) / 16000.0
        corrupt_msg = f" ({corrupt_count} corrupt packets skipped)" if corrupt_count > 0 else ""
        self._log(
            f"[Decode] {probe_info.summary_str()} -> {audio_duration_sec:.1f}s ({audio_duration_sec/60.0:.1f} min) "
            f"decoded in {decode_sec:.2f}s{corrupt_msg}"
            + (f" | AGC enhanced in {enhance_sec:.2f}s" if self.enhance_speech else "")
        )

        # Stage 3: VAD + ASR Inference (Multi-worker parallel chunks if num_workers > 1 on CPU)
        t_inf0 = time.perf_counter()
        all_raw_segments: List[SimpleSegment] = []
        speech_duration_sec = 0.0

        if self.pytorch_pipe is not None:
            res = self.pytorch_pipe(
                audio,
                batch_size=self.hardware.recommended_batch_size,
                return_timestamps=True,
            )
            for ch in res.get("chunks", []):
                ts = ch.get("timestamp") or (0.0, 0.0)
                t0_c = float(ts[0] or 0.0)
                t1_c = float(ts[1] or t0_c + 2.0)
                txt = (ch.get("text") or "").strip()
                if txt:
                    all_raw_segments.append(SimpleSegment(start=t0_c, end=t1_c, text=txt))
            speech_duration_sec = audio_duration_sec
        else:
            num_workers = self.hardware.recommended_num_workers
            if (
                num_workers > 1
                and self.hardware.recommended_device == "cpu"
                and audio_duration_sec >= 240.0
            ):
                chunks = _split_audio_at_silence_boundaries(audio, num_chunks=num_workers, sr=16000)
                self._log(
                    f"[ASR] Dispatching {len(chunks)} parallel audio chunks across "
                    f"{num_workers} CTranslate2 workers ({self.hardware.recommended_cpu_threads} threads/worker)..."
                )
                with ThreadPoolExecutor(max_workers=len(chunks)) as pool:
                    futures = [
                        pool.submit(self._transcribe_chunk_ct2, c_audio, c_offset, None)
                        for c_offset, c_audio in chunks
                    ]
                    for idx, fut in enumerate(futures, start=1):
                        c_segs, c_speech_dur = fut.result()
                        all_raw_segments.extend(c_segs)
                        speech_duration_sec += c_speech_dur
                        if progress_callback:
                            progress_callback(
                                idx / float(len(chunks)),
                                f"Completed worker chunk {idx}/{len(chunks)} ({len(c_segs)} segments)",
                            )
                all_raw_segments.sort(key=lambda s: s.start)
            else:
                seg_counter = [0]

                def _on_seg(s_obj: SimpleSegment):
                    seg_counter[0] += 1
                    if seg_counter[0] % 50 == 0 or s_obj.end >= audio_duration_sec - 15.0:
                        el = max(time.perf_counter() - t_inf0, 1e-4)
                        pct = min(100.0, 100.0 * s_obj.end / max(audio_duration_sec, 1.0))
                        msg = (
                            f"[{s_obj.start:07.1f}s - {s_obj.end:07.1f}s] "
                            f"({pct:4.1f}%, {seg_counter[0]} segs, {s_obj.end/el:.1f}x RT) {s_obj.text[:65]}"
                        )
                        self._log(f"      {msg}")
                        if progress_callback:
                            progress_callback(pct / 100.0, msg)

                all_raw_segments, speech_duration_sec = self._transcribe_chunk_ct2(
                    audio, offset_sec=0.0, on_segment=_on_seg
                )

        with open(raw_path, "w", encoding="utf-8") as f:
            raw_lines = [f"[{s.start:07.1f} - {s.end:07.1f}] {s.text}\n" for s in all_raw_segments]
            f.writelines(raw_lines)

        inference_sec = time.perf_counter() - t_inf0

        # Stage 4: Pre-compiled Regex Sanitization & Section/Speaker Grouping
        t_post0 = time.perf_counter()
        cleaned_segments: List[Dict] = []
        for s in all_raw_segments:
            txt = clean_text(s.text, compiled_replacements=self.compiled_replacements)
            if not txt:
                continue
            if (
                cleaned_segments
                and txt == cleaned_segments[-1]["text"]
                and len(txt.split()) <= 4
                and (s.start - cleaned_segments[-1]["end"]) < 2.0
            ):
                cleaned_segments[-1]["end"] = s.end
                continue
            cleaned_segments.append({"start": s.start, "end": s.end, "text": txt})

        grouped_blocks = group_transcript_segments(
            cleaned_segments, section_speaker_fn=section_speaker_fn
        )
        word_count = sum(len(s["text"].split()) for s in cleaned_segments)
        postprocess_sec = time.perf_counter() - t_post0

        # Stage 5: Deliverable Writing (.txt, optional .docx, .srt, .json)
        t_deliv0 = time.perf_counter()
        header = (
            "=" * 80
            + f"\nTRANSCRIPT: {os.path.basename(audio_abs)}\n"
            + f"MEDIA FORMAT: {probe_info.summary_str()}\n"
            + f"DURATION: {sec_to_hms(audio_duration_sec)} ({audio_duration_sec:.1f}s)\n"
            + f"HARDWARE: {self.hardware.summary_line()}\n"
            + f"SEGMENTS: {len(cleaned_segments)} | DIALOGUE BLOCKS: {len(grouped_blocks)} | WORDS: {word_count:,}\n"
            + "=" * 80
        )
        formatted_txt = format_transcript_sections(
            cleaned_segments, header_text=header, section_speaker_fn=section_speaker_fn
        )
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(formatted_txt)

        if srt_path:
            with open(srt_path, "w", encoding="utf-8") as f:
                f.write(format_srt(cleaned_segments))

        deliverables_sec = time.perf_counter() - t_deliv0
        self._files_processed += 1
        amortized_model_load = self.model_load_sec / float(self._files_processed)

        total_wall_sec = (
            decode_sec
            + enhance_sec
            + amortized_model_load
            + inference_sec
            + postprocess_sec
            + deliverables_sec
        )

        timings = StageTimings(
            file_size_mb=file_size_mb,
            container_info=probe_info.summary_str(),
            audio_duration_sec=audio_duration_sec,
            speech_duration_sec=speech_duration_sec,
            decode_sec=decode_sec,
            enhance_sec=enhance_sec,
            model_load_sec=amortized_model_load,
            inference_sec=inference_sec,
            postprocess_sec=postprocess_sec,
            deliverables_sec=deliverables_sec,
            total_wall_sec=total_wall_sec,
        )

        if docx_path:
            t_d0 = time.perf_counter()
            build_auto_docx_report(
                output_docx_path=docx_path,
                audio_path=audio_abs,
                segments=cleaned_segments,
                grouped_blocks=grouped_blocks,
                timings=timings.to_dict(),
                hardware_summary=self.hardware.summary_line(),
            )
            extra_docx_sec = time.perf_counter() - t_d0
            timings.deliverables_sec += extra_docx_sec
            timings.total_wall_sec += extra_docx_sec

        if json_path:
            export_json_result(
                output_json_path=json_path,
                metadata={
                    "audio_path": audio_abs,
                    "media_probe": probe_info.to_dict(),
                    "model_size": self.model_size,
                    "enhance_speech": self.enhance_speech,
                    "hardware": self.hardware.to_dict(),
                },
                timings=timings.to_dict(),
                segments=cleaned_segments,
            )

        self._log("\n" + timings.format_table(os.path.basename(audio_abs), len(cleaned_segments), word_count))
        return FileTranscriptionResult(
            audio_path=audio_abs,
            raw_stream_path=raw_path,
            transcript_txt_path=txt_path,
            docx_path=docx_path,
            srt_path=srt_path,
            json_path=json_path,
            segment_count=len(cleaned_segments),
            word_count=word_count,
            timings=timings,
            segments=cleaned_segments,
        )

    def transcribe_batch(
        self,
        inputs: Sequence[str],
        output_dir: Optional[str] = None,
        recursive: bool = False,
        skip_existing: bool = False,
        generate_docx: bool = True,
        generate_srt: bool = False,
        generate_json: bool = False,
        telemetry_json_path: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int, float, str], None]] = None,
    ) -> BatchTranscriptionSummary:
        """
        Transcribe an arbitrary batch of audio/video files or folders using a producer-consumer
        pre-decode thread pool that overlaps next-file I/O + 16 kHz decoding + AGC enhancement
        with active ASR inference.
        """
        t_batch0 = time.perf_counter()
        files = resolve_audio_inputs(inputs, recursive=recursive)
        if not files:
            raise FileNotFoundError(f"No supported audio or video files found in inputs: {inputs}")

        self.ensure_model_loaded()

        active_files: List[str] = []
        for fpath in files:
            bdir = os.path.abspath(output_dir) if output_dir else os.path.dirname(fpath)
            stem = os.path.splitext(os.path.basename(fpath))[0]
            expected_txt = os.path.join(bdir, f"{stem}-Transcript.txt")
            if skip_existing and os.path.exists(expected_txt):
                self._log(f"[Skip] Existing transcript found: {expected_txt}")
                continue
            active_files.append(fpath)

        total_files = len(active_files)
        results: List[FileTranscriptionResult] = []
        if total_files == 0:
            return BatchTranscriptionSummary(hardware=self.hardware, results=[], total_batch_wall_sec=0.0)

        self._log(f"[Batch] Processing {total_files} media recording(s) with pre-decode pipelining...")

        with ThreadPoolExecutor(max_workers=1) as predecode_pool:
            next_future: Optional[Future] = predecode_pool.submit(
                self._decode_and_prepare, active_files[0]
            )

            for idx, fpath in enumerate(active_files):
                self._log(f"\n[Batch {idx+1}/{total_files}] Starting {os.path.basename(fpath)}...")
                assert next_future is not None
                try:
                    predecoded_data = next_future.result()
                except Exception as e:
                    self._log(f"[Error] Skipping '{os.path.basename(fpath)}': {e}")
                    if idx + 1 < total_files:
                        next_future = predecode_pool.submit(
                            self._decode_and_prepare, active_files[idx + 1]
                        )
                    else:
                        next_future = None
                    continue

                if idx + 1 < total_files:
                    next_future = predecode_pool.submit(
                        self._decode_and_prepare, active_files[idx + 1]
                    )
                else:
                    next_future = None

                def _file_prog(frac: float, msg: str):
                    if progress_callback:
                        progress_callback(idx + 1, total_files, frac, msg)

                res = self.transcribe_file(
                    audio_path=fpath,
                    output_dir=output_dir,
                    generate_docx=generate_docx,
                    generate_srt=generate_srt,
                    generate_json=generate_json,
                    predecoded=predecoded_data,
                    progress_callback=_file_prog,
                )
                results.append(res)

        total_batch_wall = time.perf_counter() - t_batch0
        if results:
            for r in results:
                r.timings.model_load_sec = self.model_load_sec / float(len(results))

        summary = BatchTranscriptionSummary(
            hardware=self.hardware,
            results=results,
            total_batch_wall_sec=total_batch_wall,
        )
        self._log("\n" + summary.format_summary_table())

        if telemetry_json_path:
            os.makedirs(os.path.dirname(os.path.abspath(telemetry_json_path)), exist_ok=True)
            export_json_result(
                output_json_path=telemetry_json_path,
                metadata={
                    "file_count": len(results),
                    "hardware": self.hardware.to_dict(),
                },
                timings={
                    "total_batch_wall_sec": round(total_batch_wall, 3),
                    "total_audio_duration_sec": round(summary.total_audio_duration_sec, 3),
                    "overall_speedup_x_rt": round(summary.overall_speedup_x_rt, 2),
                    "overall_rtf": round(summary.overall_rtf, 5),
                    "files": [
                        {"audio_path": r.audio_path, **r.timings.to_dict()} for r in results
                    ],
                },
                segments=[],
            )
            self._log(f"[Telemetry] Saved batch benchmark JSON to: {telemetry_json_path}")

        return summary


def transcribe_audio_file(
    audio_path: str,
    output_stream_txt: str,
    model_size: str = "base.en",
    compute_type: str = "auto",
    cpu_threads: int = 4,
    batch_size: int = 16,
    chunk_length: int = 20,
    initial_prompt: Optional[str] = DEFAULT_INITIAL_PROMPT,
    use_batched: bool = True,
    vad_filter: bool = True,
    min_silence_duration_ms: int = 400,
    device: str = "auto",
    num_workers: Optional[int] = None,
    enhance_speech: bool = False,
    generate_docx: bool = True,
) -> str:
    """
    Backward-compatible single-file transcription helper.
    Runs `TranscriberEngine`, prints full per-stage timing telemetry, writes raw segments
    to `output_stream_txt`, generates the Word (.docx) executive summary report by default,
    and returns `output_stream_txt`.
    """
    engine = TranscriberEngine(
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        batch_size=batch_size,
        chunk_length=chunk_length,
        use_batched=use_batched,
        vad_filter=vad_filter,
        min_silence_duration_ms=min_silence_duration_ms,
        enhance_speech=enhance_speech,
        initial_prompt=initial_prompt,
    )
    engine.transcribe_file(
        audio_path=audio_path,
        raw_stream_path=output_stream_txt,
        generate_docx=generate_docx,
    )
    return output_stream_txt
