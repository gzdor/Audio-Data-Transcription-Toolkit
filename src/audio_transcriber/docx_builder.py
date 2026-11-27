#!/usr/bin/env python3
"""
Stage 3: Executive Summary Word Document (.docx) Builder.
Provides reusable OpenXML styling (Deep Navy / Slate Blue headers, alternating
table row shading, callout boxes, contact directories, time-indexed sections,
and automated report generation with hardware/timing telemetry).
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls
from .postprocess import sec_to_hms

COLOR_PRIMARY = RGBColor(0x1B, 0x36, 0x5D)    # Deep Navy
COLOR_SECONDARY = RGBColor(0x00, 0x66, 0x99)  # Slate Blue
COLOR_DARK = RGBColor(0x22, 0x22, 0x22)       # Charcoal Body


class MeetingReportBuilder:
    """Helper class for programmatically constructing executive summary .docx deliverables."""

    def __init__(self, margin_inches: float = 0.85):
        self.doc = docx.Document()
        for section in self.doc.sections:
            section.top_margin = Inches(margin_inches)
            section.bottom_margin = Inches(margin_inches)
            section.left_margin = Inches(margin_inches)
            section.right_margin = Inches(margin_inches)

    def add_title(self, text: str):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(3)
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(21)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        return p

    def add_subtitle(self, text: str):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(12)
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(11.5)
        run.font.italic = True
        run.font.color.rgb = COLOR_SECONDARY
        return p

    def add_h1(self, text: str):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(14.5)
        run.font.bold = True
        run.font.color.rgb = COLOR_PRIMARY
        return p

    def add_h2(self, text: str):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(9)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(text)
        run.font.name = "Calibri"
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = COLOR_SECONDARY
        return p

    def add_body(self, text: str, bold_prefix: Optional[str] = None):
        p = self.doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.12
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10.5)
            r_pre.font.bold = True
            r_pre.font.color.rgb = COLOR_DARK
        r_body = p.add_run(text)
        r_body.font.name = "Calibri"
        r_body.font.size = Pt(10.5)
        r_body.font.color.rgb = COLOR_DARK
        return p

    def add_bullet(self, text: str, bold_prefix: Optional[str] = None, level: int = 0):
        p = self.doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(2.5)
        p.paragraph_format.line_spacing = 1.12
        p.paragraph_format.left_indent = Inches(0.25 * (level + 1))
        if bold_prefix:
            r_pre = p.add_run(bold_prefix)
            r_pre.font.name = "Calibri"
            r_pre.font.size = Pt(10.5)
            r_pre.font.bold = True
            r_pre.font.color.rgb = COLOR_DARK
        r_body = p.add_run(text)
        r_body.font.name = "Calibri"
        r_body.font.size = Pt(10.5)
        r_body.font.color.rgb = COLOR_DARK
        return p

    def add_callout(self, text: str, title: str = "KEY TAKEAWAY"):
        tbl = self.doc.add_table(rows=1, cols=1)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = tbl.cell(0, 0)
        cell.width = Inches(6.8)
        tcPr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(r'<w:shd {} w:fill="F0F4F8"/>'.format(nsdecls("w")))
        tcPr.append(shd)
        borders = parse_xml(
            r'<w:tcBorders {}><w:top w:val="none"/><w:left w:val="single" w:sz="24" w:space="0" w:color="1B365D"/><w:bottom w:val="none"/><w:right w:val="none"/></w:tcBorders>'.format(
                nsdecls("w")
            )
        )
        tcPr.append(borders)
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(3)
        p.paragraph_format.space_after = Pt(3)
        r_title = p.add_run(f"[{title}] ")
        r_title.font.name = "Calibri"
        r_title.font.size = Pt(10)
        r_title.font.bold = True
        r_title.font.color.rgb = COLOR_PRIMARY
        r_txt = p.add_run(text)
        r_txt.font.name = "Calibri"
        r_txt.font.size = Pt(10)
        r_txt.font.color.rgb = COLOR_DARK
        p_after = self.doc.add_paragraph()
        p_after.paragraph_format.space_before = Pt(0)
        p_after.paragraph_format.space_after = Pt(4)

    def add_metadata_table(self, rows_data: Sequence[Tuple[str, str]]):
        tbl = self.doc.add_table(rows=len(rows_data), cols=2)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for idx, (label, val) in enumerate(rows_data):
            row = tbl.rows[idx]
            c0, c1 = row.cells[0], row.cells[1]
            c0.width = Inches(1.8)
            c1.width = Inches(5.0)
            p0 = c0.paragraphs[0]
            p0.paragraph_format.space_after = Pt(2)
            r0 = p0.add_run(label)
            r0.font.name = "Calibri"
            r0.font.bold = True
            r0.font.size = Pt(9.5)
            r0.font.color.rgb = COLOR_PRIMARY
            p1 = c1.paragraphs[0]
            p1.paragraph_format.space_after = Pt(2)
            r1 = p1.add_run(val)
            r1.font.name = "Calibri"
            r1.font.size = Pt(9.5)
            r1.font.color.rgb = COLOR_DARK
            for c in (c0, c1):
                tcPr = c._tc.get_or_add_tcPr()
                shd = parse_xml(
                    r'<w:shd {} w:fill="{}"/>'.format(
                        nsdecls("w"), "F4F6F9" if idx % 2 == 0 else "FFFFFF"
                    )
                )
                tcPr.append(shd)
        self.doc.add_paragraph().paragraph_format.space_after = Pt(6)

    def add_styled_table(
        self,
        headers: Sequence[str],
        rows_data: Sequence[Sequence[str]],
        col_widths_inches: Optional[Sequence[float]] = None,
        font_size_pt: float = 9.0,
    ):
        tbl = self.doc.add_table(rows=len(rows_data) + 1, cols=len(headers))
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        for idx, h in enumerate(headers):
            cell = tbl.rows[0].cells[idx]
            p = cell.paragraphs[0]
            r = p.add_run(h)
            r.font.name = "Calibri"
            r.font.bold = True
            r.font.size = Pt(font_size_pt)
            r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            tcPr = cell._tc.get_or_add_tcPr()
            shd = parse_xml(r'<w:shd {} w:fill="1B365D"/>'.format(nsdecls("w")))
            tcPr.append(shd)

        for row_idx, data in enumerate(rows_data, start=1):
            row = tbl.rows[row_idx]
            for col_idx, text in enumerate(data):
                cell = row.cells[col_idx]
                p = cell.paragraphs[0]
                p.paragraph_format.space_after = Pt(2)
                r = p.add_run(str(text))
                r.font.name = "Calibri"
                r.font.size = Pt(font_size_pt)
                if col_idx == 0:
                    r.font.bold = True
                r.font.color.rgb = COLOR_DARK
                tcPr = cell._tc.get_or_add_tcPr()
                shd = parse_xml(
                    r'<w:shd {} w:fill="{}"/>'.format(
                        nsdecls("w"), "F4F6F9" if row_idx % 2 == 1 else "FFFFFF"
                    )
                )
                tcPr.append(shd)

        if col_widths_inches:
            for row in tbl.rows:
                for i, w in enumerate(col_widths_inches):
                    row.cells[i].width = Inches(w)

        self.doc.add_paragraph().paragraph_format.space_after = Pt(4)
        return tbl

    def save(self, output_path: str):
        parent = os.path.dirname(os.path.abspath(output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.doc.save(output_path)


_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "have", "for", "not", "are", "was",
    "were", "but", "what", "all", "when", "we're", "there", "can", "you", "your",
    "they", "their", "about", "like", "just", "going", "know", "think", "really",
    "would", "could", "should", "into", "some", "more", "than", "then", "them",
    "also", "very", "because", "where", "which", "been", "being", "will", "here",
    "yeah", "okay", "right", "well", "thing", "things", "those", "these", "much",
    "over", "only", "other", "such", "even", "back", "after", "before", "through",
}


def _extract_top_keywords(texts: List[str], top_k: int = 6) -> List[str]:
    import re
    counts: Dict[str, int] = {}
    for t in texts:
        for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", t):
            lw = w.lower()
            if lw not in _STOPWORDS and len(lw) >= 4:
                counts[lw] = counts.get(lw, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:top_k]]


def _extract_key_takeaways(segments: List[Dict], max_items: int = 5) -> List[Dict]:
    """
    Select up to `max_items` high-information representative statements spread across
    the chronological span of the recording using TF-weighted keyword density scoring.
    """
    import re
    if not segments:
        return []

    word_freq: Dict[str, int] = {}
    for seg in segments:
        for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", seg.get("text", "")):
            lw = w.lower()
            if lw not in _STOPWORDS:
                word_freq[lw] = word_freq.get(lw, 0) + 1

    if not word_freq:
        return segments[:max_items]

    scored = []
    for idx, seg in enumerate(segments):
        txt = seg.get("text", "").strip()
        words = txt.split()
        if len(words) < 7:
            continue
        tokens = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", txt) if w.lower() not in _STOPWORDS]
        if not tokens:
            continue
        score = sum(word_freq.get(t, 0) for t in set(tokens)) / (len(words) ** 0.35)
        scored.append((score, idx, seg))

    if not scored:
        return segments[: min(max_items, len(segments))]

    # Partition recording into chronological buckets so takeaways cover the whole recording
    n_buckets = min(max_items, len(scored))
    bucket_size = max(1, len(segments) // n_buckets)
    selected = []
    used_indices = set()
    for b in range(n_buckets):
        b_start = b * bucket_size
        b_end = (b + 1) * bucket_size if b + 1 < n_buckets else len(segments)
        candidates = [item for item in scored if b_start <= item[1] < b_end]
        if candidates:
            best = max(candidates, key=lambda x: x[0])
            selected.append(best[2])
            used_indices.add(best[1])

    if len(selected) < max_items:
        for score, idx, seg in sorted(scored, key=lambda x: -x[0]):
            if idx not in used_indices:
                selected.append(seg)
                used_indices.add(idx)
                if len(selected) >= max_items:
                    break

    return sorted(selected, key=lambda s: float(s.get("start", 0.0)))


def build_auto_docx_report(
    output_docx_path: str,
    audio_path: str,
    segments: List[Dict],
    grouped_blocks: List[Dict],
    timings: Dict[str, Any],
    hardware_summary: str,
) -> str:
    """
    Automatically generate a formatted Word (.docx) Executive Summary & Time-Indexed
    Conversation Report for any transcribed recording, including key takeaways,
    section topic overview, pipeline timing telemetry, and chronological dialogue.
    """
    builder = MeetingReportBuilder()
    base_name = os.path.basename(audio_path)
    dur_sec = float(timings.get("audio_duration_sec", 0.0))
    speech_sec = float(timings.get("speech_duration_sec", 0.0)) or dur_sec
    wall_sec = float(timings.get("total_wall_sec", 0.01))
    speedup = float(timings.get("speedup_x_rt", 0.0))
    rtf = float(timings.get("rtf", 0.0))
    total_words = sum(len(s.get("text", "").split()) for s in segments)
    wpm = (total_words / (speech_sec / 60.0)) if speech_sec > 0 else 0.0
    top_terms = _extract_top_keywords([s.get("text", "") for s in segments], top_k=8)

    builder.add_title(f"Executive Summary & Audio Transcription Report: {base_name}")
    builder.add_subtitle(
        f"Automated Executive Takeaways, Topic Breakdown, Time-Indexed Transcript & Telemetry | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    builder.add_metadata_table([
        ("Source Media File", f"{audio_path} ({timings.get('file_size_mb', 0.0):.2f} MB | {timings.get('container_info', 'Audio/Video')})"),
        ("Recording Duration", f"{sec_to_hms(dur_sec)} ({dur_sec:.1f}s / {dur_sec/60.0:.1f} min) | Active Speech: {sec_to_hms(speech_sec)}"),
        ("Speech & Pace", f"{len(segments)} segments | {len(grouped_blocks)} dialogue blocks | {total_words:,} words ({wpm:.0f} WPM)"),
        ("Primary Topics / Terms", ", ".join(top_terms) if top_terms else "General conversation / audio recording"),
        ("Hardware & Speed", f"{hardware_summary} | Wall Time: {wall_sec:.2f}s ({speedup:.1f}x Real-Time, RTF: {rtf:.4f})"),
    ])

    # Section 1: Executive Summary & Key Takeaways
    builder.add_h1("1. Executive Summary & Key Takeaways")
    section_groups: Dict[str, List[Dict]] = {}
    for blk in grouped_blocks:
        section_groups.setdefault(blk["section"], []).append(blk)

    builder.add_body(
        f"This report synthesizes the audio recording '{base_name}' spanning {sec_to_hms(dur_sec)} "
        f"({dur_sec/60.0:.1f} minutes) with {total_words:,} transcribed words across {len(grouped_blocks)} "
        f"grouped dialogue blocks and {len(section_groups)} chronological section(s). "
        + (
            f"Primary recurring themes and terminology identified across the recording include: {', '.join(top_terms)}."
            if top_terms
            else ""
        )
    )

    takeaways = _extract_key_takeaways(segments, max_items=5)
    if takeaways:
        lead = takeaways[0]
        builder.add_callout(
            f"[{sec_to_hms(lead['start'])} – {sec_to_hms(lead['end'])}] \"{lead['text']}\"",
            "KEY HIGHLIGHT FROM RECORDING",
        )
        builder.add_h2("Representative Highlights Across Recording")
        for tk_seg in takeaways:
            ts_prefix = f"[{sec_to_hms(tk_seg['start'])} – {sec_to_hms(tk_seg['end'])}] "
            builder.add_bullet(tk_seg["text"], bold_prefix=ts_prefix)

    if section_groups:
        builder.add_h2("Chronological Section & Topic Overview")
        sec_rows = []
        for sec_name, blks in section_groups.items():
            s_t0 = min(b["start"] for b in blks)
            s_t1 = max(b["end"] for b in blks)
            s_texts = [" ".join(b["texts"]) for b in blks]
            s_words = sum(len(t.split()) for t in s_texts)
            s_terms = ", ".join(_extract_top_keywords(s_texts, top_k=5)) or "General discussion"
            sec_rows.append((sec_name, f"{sec_to_hms(s_t0)} – {sec_to_hms(s_t1)}", f"{s_words:,} words", s_terms))
        builder.add_styled_table(
            ["Chronological Section", "Time Window", "Volume", "Dominant Keywords / Themes"],
            sec_rows,
            [2.2, 1.4, 1.0, 2.2],
            font_size_pt=9.0,
        )

    # Section 2: Pipeline Execution Timing & Telemetry
    builder.add_h1("2. Pipeline Execution Timing & Telemetry")
    timing_rows = [
        ("1. Media Decode & 16 kHz Resampling", f"{timings.get('decode_sec', 0.0):.2f}s", f"{timings.get('decode_pct', 0.0):.1f}%"),
        ("2. High-Pass + RMS AGC Speech Enhancement", f"{timings.get('enhance_sec', 0.0):.2f}s", f"{timings.get('enhance_pct', 0.0):.1f}%"),
        ("3. Model Initialization (Amortized)", f"{timings.get('model_load_sec', 0.0):.2f}s", f"{timings.get('model_load_pct', 0.0):.1f}%"),
        ("4. VAD + ASR Inference", f"{timings.get('inference_sec', 0.0):.2f}s", f"{timings.get('inference_pct', 0.0):.1f}%"),
        ("5. Regex Post-Processing & Grouping", f"{timings.get('postprocess_sec', 0.0):.2f}s", f"{timings.get('postprocess_pct', 0.0):.1f}%"),
        ("Total End-to-End Pipeline Wall Time", f"{wall_sec:.2f}s ({speedup:.1f}x RT)", "100.0%"),
    ]
    builder.add_styled_table(
        ["Pipeline Stage", "Elapsed Time", "% of Total Wall Time"],
        timing_rows,
        [3.4, 1.8, 1.6],
        font_size_pt=9.0,
    )

    # Section 3: Full Time-Indexed Transcript
    builder.add_h1("3. Chronological Section Breakdown & Full Time-Indexed Dialogue")
    current_sec = None
    for block in grouped_blocks:
        if block["section"] != current_sec:
            current_sec = block["section"]
            builder.add_h2(current_sec)
        t_str = f"[{sec_to_hms(block['start'])} – {sec_to_hms(block['end'])}] {block['speaker']}: "
        body_txt = " ".join(block["texts"])
        builder.add_bullet(body_txt, bold_prefix=t_str)

    builder.save(output_docx_path)
    return output_docx_path
