#!/usr/bin/env python3
"""
Multi-format audio and video container probing and resilient 16 kHz mono float32
audio extraction using PyAV (bundled FFmpeg libraries: libavformat, libavcodec, libswresample).

Supports:
  - Audio formats: .wav (16/24-bit PCM, 32-bit IEEE float BWF), .mp3, .m4a, .aac,
    .flac, .ogg, .opus, .wma, .aiff, .aif, .amr, .caf
  - Video containers (extracts audio stream automatically): .mp4, .mkv, .mov,
    .webm, .avi, .mpeg, .mpg, .m4v, .ts, .3gp, .wmv
"""

import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import av
import numpy as np

SUPPORTED_MEDIA_EXTENSIONS: Tuple[str, ...] = (
    # Audio containers & codecs
    ".wav",
    ".mp3",
    ".m4a",
    ".aac",
    ".flac",
    ".ogg",
    ".opus",
    ".wma",
    ".aiff",
    ".aif",
    ".amr",
    ".caf",
    # Video containers (audio track extracted automatically)
    ".mp4",
    ".mkv",
    ".mov",
    ".webm",
    ".avi",
    ".mpeg",
    ".mpg",
    ".m4v",
    ".ts",
    ".3gp",
    ".wmv",
)


@dataclass
class MediaProbeInfo:
    path: str
    container_format: str
    file_size_mb: float
    duration_sec: float
    audio_streams: int
    video_streams: int
    audio_codec: str
    sample_rate: int
    channels: int

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary_str(self) -> str:
        v_tag = f", {self.video_streams} video track(s)" if self.video_streams > 0 else ""
        return (
            f"{self.container_format.upper()} [{self.audio_codec}, {self.sample_rate} Hz, "
            f"{self.channels} ch{v_tag}, {self.file_size_mb:.1f} MB]"
        )


def _open_media_container(media_path: str):
    """
    Open a media container across all PyAV versions (PyAV 10 through 16+).
    PyAV >= 14 removed the `metadata_errors` keyword argument from `av.open()`.
    """
    try:
        return av.open(media_path, metadata_errors="ignore")
    except TypeError:
        return av.open(media_path)


def probe_media_file(media_path: str) -> MediaProbeInfo:
    """
    Inspect an audio or video file and return container, codec, stream counts,
    sample rate, channel count, and duration metadata.
    Raises FileNotFoundError if missing, or ValueError if the container has no audio streams.
    """
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"Media file not found: {media_path}")

    file_size_mb = os.path.getsize(media_path) / (1024.0 * 1024.0)
    try:
        with _open_media_container(media_path) as container:
            n_audio = len(container.streams.audio)
            n_video = len(container.streams.video)
            if n_audio == 0:
                raise ValueError(
                    f"Media file '{os.path.basename(media_path)}' contains {n_video} video stream(s) "
                    f"but 0 audio tracks. Cannot transcribe a silent video file."
                )
            a_stream = container.streams.audio[0]
            codec_name = getattr(a_stream.codec_context, "name", "unknown") or "unknown"
            sr = int(getattr(a_stream.codec_context, "sample_rate", 0) or getattr(a_stream, "rate", 0) or 16000)
            layout_obj = getattr(a_stream.codec_context, "layout", None)
            ch = int(
                getattr(a_stream.codec_context, "channels", 0)
                or getattr(layout_obj, "nb_channels", 0)
                or (len(getattr(layout_obj, "channels", [])) if layout_obj is not None else 0)
                or 1
            )
            dur = float(container.duration / 1e6) if container.duration else 0.0
            if dur <= 0.0 and a_stream.duration and a_stream.time_base:
                dur = float(a_stream.duration * a_stream.time_base)
            fmt_name = container.format.name.split(",")[0] if container.format and container.format.name else "unknown"
            return MediaProbeInfo(
                path=os.path.abspath(media_path),
                container_format=fmt_name,
                file_size_mb=file_size_mb,
                duration_sec=dur,
                audio_streams=n_audio,
                video_streams=n_video,
                audio_codec=codec_name,
                sample_rate=sr,
                channels=ch,
            )
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to open or probe media file '{media_path}': {e}") from e


def _decode_single_audio_stream(
    container: "av.container.InputContainer",
    stream_index: int = 0,
    sampling_rate: int = 16000,
) -> Tuple[np.ndarray, int]:
    """
    Resiliently decode a single audio stream from an open PyAV container into a
    1D float32 numpy array at `sampling_rate` Hz. Skips corrupted packets rather
    than aborting the entire file.
    Returns (audio_float32, skipped_corrupt_packets).
    """
    audio_stream = container.streams.audio[stream_index]
    audio_stream.thread_type = "AUTO"
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sampling_rate)

    frames_list: List[np.ndarray] = []
    corrupt_packets = 0
    ffmpeg_err_cls = getattr(av, "FFmpegError", Exception)

    try:
        for packet in container.demux(audio_stream):
            try:
                for frame in packet.decode():
                    frame.pts = None
                    for resampled in resampler.resample(frame):
                        arr = resampled.to_ndarray()
                        frames_list.append(arr.reshape(-1))
            except (ffmpeg_err_cls, Exception):
                corrupt_packets += 1
                continue
    except (ffmpeg_err_cls, Exception):
        # Handle truncated EOF packet gracefully if frames were already decoded
        corrupt_packets += 1

    # Flush resampler
    try:
        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray()
            frames_list.append(arr.reshape(-1))
    except Exception:
        pass

    if not frames_list:
        return np.zeros(0, dtype=np.float32), corrupt_packets

    audio_f32 = np.concatenate(frames_list, axis=0).astype(np.float32, copy=False)
    np.nan_to_num(audio_f32, copy=False, nan=0.0, posinf=1.0, neginf=-1.0)
    np.clip(audio_f32, -1.0, 1.0, out=audio_f32)
    return audio_f32, corrupt_packets


def load_and_resample_audio(
    media_path: str,
    sampling_rate: int = 16000,
    mix_all_audio_streams: bool = False,
) -> Tuple[np.ndarray, MediaProbeInfo, int]:
    """
    Decode any supported audio (.wav, .mp3, .m4a, .aac, .flac, .ogg, .opus, .wma)
    or video (.mp4, .mkv, .mov, .webm, .avi) file into a 16 kHz mono float32 array.

    If `mix_all_audio_streams=True` and the container has multiple audio tracks
    (e.g., separate microphone and desktop audio tracks in an .mp4/.mkv recording),
    decodes and mixes all audio tracks with equal weighting.

    Returns (audio_array_16k_f32, probe_info, total_corrupt_packets_skipped).
    """
    probe = probe_media_file(media_path)
    total_corrupt = 0

    if not mix_all_audio_streams or probe.audio_streams == 1:
        with _open_media_container(media_path) as container:
            audio, corrupt = _decode_single_audio_stream(
                container, stream_index=0, sampling_rate=sampling_rate
            )
            total_corrupt += corrupt
    else:
        track_arrays: List[np.ndarray] = []
        for s_idx in range(probe.audio_streams):
            with _open_media_container(media_path) as container:
                arr, corrupt = _decode_single_audio_stream(
                    container, stream_index=s_idx, sampling_rate=sampling_rate
                )
                total_corrupt += corrupt
                if len(arr) > 0:
                    track_arrays.append(arr)
        if not track_arrays:
            audio = np.zeros(0, dtype=np.float32)
        elif len(track_arrays) == 1:
            audio = track_arrays[0]
        else:
            max_len = max(len(a) for a in track_arrays)
            mixed = np.zeros(max_len, dtype=np.float32)
            for a in track_arrays:
                mixed[: len(a)] += a
            mixed /= float(len(track_arrays))
            np.clip(mixed, -1.0, 1.0, out=mixed)
            audio = mixed

    if len(audio) == 0:
        raise ValueError(
            f"Decoded 0 audio samples from '{os.path.basename(media_path)}' "
            f"(corrupt packets skipped: {total_corrupt}). File may be empty or corrupted."
        )

    probe.duration_sec = len(audio) / float(sampling_rate)
    return audio, probe, total_corrupt

