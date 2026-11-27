"""
audio_transcriber: A tool for making audio transcriptions and report summaries
off audio recordings from audio (.wav, .mp3, .mp4, etc.) recordings.
Supports scalable local multi-core CPU, NVIDIA CUDA, AMD ROCm, and Windows DirectML
execution, O(1)-memory far-field speech enhancement (AGC), pre-compiled terminology
post-processing, time-window speaker diarization, detailed stage timing telemetry,
python-docx conversation report generation, and Desktop/Web GUI.
"""

from .audio_io import (
    SUPPORTED_MEDIA_EXTENSIONS,
    MediaProbeInfo,
    load_and_resample_audio,
    probe_media_file,
)
from .hardware import GPUDeviceInfo, HardwareProfile, detect_hardware_profile
from .transcribe import (
    BatchTranscriptionSummary,
    FileTranscriptionResult,
    StageTimings,
    TranscriberEngine,
    enhance_audio_fast,
    resolve_audio_inputs,
    transcribe_audio_file,
)
from .postprocess import (
    DEFAULT_DAS_REPLACEMENTS,
    DEFAULT_TECHNICAL_REPLACEMENTS,
    clean_text,
    export_json_result,
    format_srt,
    format_transcript_sections,
    group_transcript_segments,
    load_custom_replacements,
    parse_raw_stream,
    sec_to_hms,
)
from .docx_builder import MeetingReportBuilder, build_auto_docx_report

__all__ = [
    "SUPPORTED_MEDIA_EXTENSIONS",
    "MediaProbeInfo",
    "load_and_resample_audio",
    "probe_media_file",
    "GPUDeviceInfo",
    "HardwareProfile",
    "detect_hardware_profile",
    "StageTimings",
    "FileTranscriptionResult",
    "BatchTranscriptionSummary",
    "TranscriberEngine",
    "enhance_audio_fast",
    "resolve_audio_inputs",
    "transcribe_audio_file",
    "clean_text",
    "format_transcript_sections",
    "group_transcript_segments",
    "load_custom_replacements",
    "parse_raw_stream",
    "format_srt",
    "export_json_result",
    "sec_to_hms",
    "DEFAULT_TECHNICAL_REPLACEMENTS",
    "DEFAULT_DAS_REPLACEMENTS",
    "MeetingReportBuilder",
    "build_auto_docx_report",
]
__version__ = "2.1.0"
