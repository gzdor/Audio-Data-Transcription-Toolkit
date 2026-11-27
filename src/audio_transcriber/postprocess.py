#!/usr/bin/env python3
"""
Stage 2: Pre-compiled domain terminology regex sanitization, time-interval
section/speaker attribution, contiguous utterance grouping, and multi-format
(.txt, .srt, .json) export.
"""

import json
import re
from typing import Any, Callable, Dict, List, Optional, Pattern, Sequence, Tuple

DEFAULT_TECHNICAL_REPLACEMENTS: List[Tuple[str, str]] = [
    (r"\bconsole\b(?=\s+multi\s*physics|\s+based|\s+FEA)", "COMSOL"),
    (r"\bconsole multi physics\b", "COMSOL Multiphysics"),
    (r"\bbrag ratings\b", "Bragg gratings"),
    (r"\bbrag\b(?=\s+gratings)", "Bragg"),
    (r"\bbat scatter\b", "backscatter"),
    (r"\bbat\s+scatter\b", "backscatter"),
    (r"\bbio of engines\b", "birefringence"),
    (r"\bbiophringence\b", "birefringence"),
    (r"\bblade path\b", "blade pass"),
    (r"\bpect tubing\b", "PEX tubing"),
    (r"\bperps\b", "Hertz"),
    (r"\bCIMF\b", "SigMF"),
    (r"\bPECMF\b", "SigMF"),
    (r"\bSIGAMF\b", "SigMF"),
    (r"\bRaytune\b", "Ray Tune"),
    (r"\bhyperparamviratuning\b", "hyperparameter tuning"),
    (r"\bsurface-blade\b", "surface-laid"),
    (r"\bsurface-late\b", "surface-laid"),
    (r"\bsurface plate\b", "surface-laid"),
    (r"\bmicro-erturbation\b", "micro-perturbation"),
    (r"\bPaul Urethane\b", "polyurethane"),
]

# Backward-compatible alias
DEFAULT_DAS_REPLACEMENTS = DEFAULT_TECHNICAL_REPLACEMENTS

# Pre-compile default regex list once at module load for O(1) setup across large batches
COMPILED_DEFAULT_REPLACEMENTS: List[Tuple[Pattern, str]] = [
    (re.compile(pat, flags=re.IGNORECASE), rep) for pat, rep in DEFAULT_TECHNICAL_REPLACEMENTS
]
REPETITION_LOOP_REGEX: Pattern = re.compile(
    r"\b(\w+(?:\s+\w+){0,3})(?:\s*,?\s*\1){3,}\b", flags=re.IGNORECASE
)
STREAM_LINE_REGEX: Pattern = re.compile(r"\[([0-9.]+)\s*-\s*([0-9.]+)\]\s*(.*)")


def load_custom_replacements(json_path: str) -> List[Tuple[str, str]]:
    """
    Load custom regex replacement rules from a JSON file (either a dict of
    {"pattern": "replacement"} or a list of ["pattern", "replacement"] pairs)
    and append them to DEFAULT_TECHNICAL_REPLACEMENTS.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    custom: List[Tuple[str, str]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            custom.append((str(k), str(v)))
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                custom.append((str(item[0]), str(item[1])))
    return list(DEFAULT_TECHNICAL_REPLACEMENTS) + custom


def compile_replacements(
    replacements: Sequence[Tuple[str, str]],
) -> List[Tuple[Pattern, str]]:
    """Compile a sequence of (pattern_str, replacement_str) tuples once."""
    if replacements is DEFAULT_TECHNICAL_REPLACEMENTS:
        return COMPILED_DEFAULT_REPLACEMENTS
    return [(re.compile(pat, flags=re.IGNORECASE), rep) for pat, rep in replacements]


def clean_text(
    text: str,
    replacements: Sequence[Tuple[str, str]] = DEFAULT_TECHNICAL_REPLACEMENTS,
    compiled_replacements: Optional[Sequence[Tuple[Pattern, str]]] = None,
) -> str:
    """Apply pre-compiled case-insensitive regex replacements and collapse repetition loops."""
    active_compiled = (
        compiled_replacements
        if compiled_replacements is not None
        else compile_replacements(replacements)
    )
    for regex_obj, rep in active_compiled:
        text = regex_obj.sub(rep, text)
    text = REPETITION_LOOP_REGEX.sub(r"\1", text)
    return text.strip()


def sec_to_hms(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def sec_to_srt_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SubRip (.srt) files."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms >= 1000:
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_raw_stream(
    raw_stream_path: str,
    replacements: Sequence[Tuple[str, str]] = DEFAULT_TECHNICAL_REPLACEMENTS,
) -> List[Dict]:
    """Parse `[start - end] text` lines from raw Whisper stream output."""
    compiled = compile_replacements(replacements)
    with open(raw_stream_path, "r", encoding="utf-8") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    parsed: List[Dict] = []
    for line in raw_lines:
        m = STREAM_LINE_REGEX.match(line)
        if not m:
            continue
        t_start = float(m.group(1))
        t_end = float(m.group(2))
        text = clean_text(m.group(3), compiled_replacements=compiled)
        if not text:
            continue
        if (
            parsed
            and text == parsed[-1]["text"]
            and len(text.split()) <= 4
            and (t_start - parsed[-1]["end"]) < 2.0
        ):
            parsed[-1]["end"] = t_end
            continue
        parsed.append({"start": t_start, "end": t_end, "text": text})
    return parsed


def group_transcript_segments(
    segments: List[Dict],
    section_speaker_fn: Optional[Callable[[float, str], Tuple[str, str]]] = None,
    max_gap_sec: float = 10.0,
    max_block_sec: float = 60.0,
    auto_topic_window_sec: float = 900.0,
) -> List[Dict]:
    """Group contiguous segments by section and speaker."""
    tagged = []
    for seg in segments:
        if section_speaker_fn is not None:
            section, speaker = section_speaker_fn(seg["start"], seg["text"])
        else:
            win_idx = int(seg["start"] // auto_topic_window_sec)
            w_start = win_idx * auto_topic_window_sec
            w_end = (win_idx + 1) * auto_topic_window_sec
            section = f"SECTION {win_idx + 1} [{sec_to_hms(w_start)} - {sec_to_hms(w_end)}] | Chronological Segment Window"
            speaker = seg.get("speaker", "Speaker")
        tagged.append({
            "start": seg["start"],
            "end": seg["end"],
            "section": section,
            "speaker": speaker,
            "text": seg["text"],
        })

    grouped: List[Dict] = []
    for seg in tagged:
        if not grouped:
            grouped.append({
                "section": seg["section"],
                "speaker": seg["speaker"],
                "start": seg["start"],
                "end": seg["end"],
                "texts": [seg["text"]],
            })
        else:
            last = grouped[-1]
            if (
                last["section"] == seg["section"]
                and last["speaker"] == seg["speaker"]
                and (seg["start"] - last["end"]) < max_gap_sec
                and (seg["end"] - last["start"]) < max_block_sec
            ):
                last["end"] = seg["end"]
                last["texts"].append(seg["text"])
            else:
                grouped.append({
                    "section": seg["section"],
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "texts": [seg["text"]],
                })
    return grouped


def format_transcript_sections(
    segments: List[Dict],
    header_text: str,
    section_speaker_fn: Optional[Callable[[float, str], Tuple[str, str]]] = None,
    max_gap_sec: float = 10.0,
    max_block_sec: float = 60.0,
) -> str:
    """
    Assign section headers and speaker labels to segments, merge contiguous utterances,
    and return a formatted full-text transcript string.
    """
    grouped = group_transcript_segments(
        segments,
        section_speaker_fn=section_speaker_fn,
        max_gap_sec=max_gap_sec,
        max_block_sec=max_block_sec,
    )

    lines = [header_text.rstrip() + "\n"]
    current_sec = None
    for g in grouped:
        if g["section"] != current_sec:
            current_sec = g["section"]
            lines.append("\n" + "=" * 80 + "\n")
            lines.append(f"{current_sec}\n")
            lines.append("=" * 80 + "\n\n")
        t_str = f"[{sec_to_hms(g['start'])} - {sec_to_hms(g['end'])}]"
        body = " ".join(g["texts"])
        lines.append(f"{t_str} {g['speaker']}:\n{body}\n\n")

    return "".join(lines)


def format_srt(segments: List[Dict]) -> str:
    """Format cleaned segments into standard SubRip (.srt) subtitle format."""
    blocks = []
    for idx, seg in enumerate(segments, start=1):
        t0 = sec_to_srt_timestamp(seg["start"])
        t1 = sec_to_srt_timestamp(max(seg["end"], seg["start"] + 0.5))
        blocks.append(f"{idx}\n{t0} --> {t1}\n{seg['text']}\n")
    return "\n".join(blocks)


def export_json_result(
    output_json_path: str,
    metadata: Dict[str, Any],
    timings: Dict[str, Any],
    segments: List[Dict],
) -> str:
    """Write structured JSON containing file metadata, stage timings, and segments."""
    payload = {
        "metadata": metadata,
        "timings": timings,
        "segment_count": len(segments),
        "segments": segments,
    }
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return output_json_path
