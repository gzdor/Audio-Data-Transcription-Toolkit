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


def build_auto_docx_report(
    output_docx_path: str,
    audio_path: str,
    segments: List[Dict],
    grouped_blocks: List[Dict],
    timings: Dict[str, Any],
    hardware_summary: str,
) -> str:
    """
    Automatically generate a formatted Word (.docx) summary & time-indexed transcript
    report for any transcribed recording, including pipeline timing telemetry.
    """
    builder = MeetingReportBuilder()
    base_name = os.path.basename(audio_path)
    dur_sec = float(timings.get("audio_duration_sec", 0.0))
    wall_sec = float(timings.get("total_wall_sec", 0.01))
    speedup = float(timings.get("speedup_x_rt", 0.0))
    rtf = float(timings.get("rtf", 0.0))
    total_words = sum(len(s.get("text", "").split()) for s in segments)

    builder.add_title(f"Audio Transcription & Conversation Summary Report: {base_name}")
    builder.add_subtitle(
        f"Automated ASR Synthesis, Chronological Section Breakdown & Execution Telemetry | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    )

    builder.add_metadata_table([
        ("Source Audio File", f"{audio_path} ({timings.get('file_size_mb', 0.0):.2f} MB)"),
        ("Audio Duration", f"{sec_to_hms(dur_sec)} ({dur_sec:.1f}s / {dur_sec/60.0:.1f} min)"),
        ("Speech Activity", f"{len(segments)} segments | {len(grouped_blocks)} grouped dialogue blocks | {total_words:,} words"),
        ("Hardware Profile", hardware_summary),
        ("End-to-End Speed", f"Total Wall Time: {wall_sec:.2f}s | Speedup: {speedup:.1f}x Real-Time | RTF: {rtf:.4f}"),
    ])

    builder.add_h1("1. Pipeline Execution Timing & Telemetry")
    timing_rows = [
        ("1. Audio Decode & 16 kHz Resampling", f"{timings.get('decode_sec', 0.0):.2f}s", f"{timings.get('decode_pct', 0.0):.1f}%"),
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

    builder.add_h1("2. Chronological Section Breakdown & Dialogue")
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
