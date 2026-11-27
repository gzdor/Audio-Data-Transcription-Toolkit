#!/usr/bin/env python3
"""
Graphical User Interface (GUI) for the Audio Data Transcription Toolkit.
Provides:
1. Native Desktop GUI (`tkinter` / `ttk`) for Windows, macOS, and Linux/WSLg.
2. Zero-dependency Local Web GUI (`http.server` on http://127.0.0.1:8765) for headless/SSH
   sessions or browser-based workflow execution (`audio-transcribe-gui --web`).
"""

import argparse
import json
import os
import queue
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from .audio_io import SUPPORTED_MEDIA_EXTENSIONS
from .hardware import detect_hardware_profile
from .transcribe import (
    DEFAULT_INITIAL_PROMPT,
    TranscriberEngine,
    resolve_audio_inputs,
)


# ==============================================================================
# 1. Native Desktop GUI (tkinter / ttk)
# ==============================================================================
def launch_desktop_gui(initial_files: Optional[List[str]] = None):
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    hw = detect_hardware_profile()

    root = tk.Tk()
    root.title("Audio Data Transcription Toolkit")
    root.geometry("1060x800")
    root.minsize(920, 680)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    NAVY = "#1B365D"

    style.configure("Header.TFrame", background=NAVY)
    style.configure("HeaderTitle.TLabel", background=NAVY, foreground="white", font=("Calibri", 16, "bold"))
    style.configure("HeaderSub.TLabel", background=NAVY, foreground="#D0E2F2", font=("Calibri", 10))
    style.configure("Run.TButton", font=("Calibri", 11, "bold"), padding=6)

    # Top Banner
    header_frame = ttk.Frame(root, style="Header.TFrame", padding=(14, 10))
    header_frame.pack(fill=tk.X)
    ttk.Label(
        header_frame,
        text="Audio Data Transcription & Conversation Report Toolkit",
        style="HeaderTitle.TLabel",
    ).pack(anchor=tk.W)
    hw_label_var = tk.StringVar(value=f"Detected Hardware: {hw.summary_line()}")
    ttk.Label(header_frame, textvariable=hw_label_var, style="HeaderSub.TLabel").pack(anchor=tk.W, pady=(2, 0))

    main_pane = ttk.PanedWindow(root, orient=tk.VERTICAL)
    main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    top_container = ttk.Frame(main_pane)
    bottom_container = ttk.Frame(main_pane)
    main_pane.add(top_container, weight=3)
    main_pane.add(bottom_container, weight=4)

    # --- Section A: Input Media Queue ---
    ext_display = ", ".join(SUPPORTED_MEDIA_EXTENSIONS[:12]) + ", ..."
    queue_group = ttk.LabelFrame(
        top_container,
        text=f"1. Media Recordings Queue — Supports Audio & Video ({ext_display})",
        padding=8,
    )
    queue_group.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

    q_list_frame = ttk.Frame(queue_group)
    q_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    file_listbox = tk.Listbox(
        q_list_frame,
        selectmode=tk.EXTENDED,
        height=6,
        font=("Consolas", 9),
        bg="white",
    )
    q_scroll = ttk.Scrollbar(q_list_frame, orient=tk.VERTICAL, command=file_listbox.yview)
    file_listbox.configure(yscrollcommand=q_scroll.set)
    file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    q_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    if initial_files:
        for f in resolve_audio_inputs(initial_files, recursive=True):
            file_listbox.insert(tk.END, f)

    btn_col = ttk.Frame(queue_group, padding=(8, 0, 0, 0))
    btn_col.pack(side=tk.RIGHT, fill=tk.Y)

    ext_glob = " ".join(f"*{ext}" for ext in SUPPORTED_MEDIA_EXTENSIONS)

    def _add_files():
        paths = filedialog.askopenfilenames(
            title="Select Audio or Video Recordings",
            filetypes=[
                ("Audio & Video Media", ext_glob),
                ("Audio Files", "*.wav *.mp3 *.m4a *.aac *.flac *.ogg *.opus *.wma *.aiff"),
                ("Video Files", "*.mp4 *.mkv *.mov *.webm *.avi *.mpeg *.mpg *.m4v *.wmv"),
                ("All Files", "*.*"),
            ],
        )
        existing = set(file_listbox.get(0, tk.END))
        for p in paths:
            ap = os.path.abspath(p)
            if ap not in existing:
                file_listbox.insert(tk.END, ap)
                existing.add(ap)

    def _add_folder():
        folder = filedialog.askdirectory(title="Select Folder Containing Audio or Video Recordings")
        if not folder:
            return
        found = resolve_audio_inputs([folder], recursive=True)
        if not found:
            messagebox.showinfo("No Media Files", f"No supported audio or video files found in:\n{folder}")
            return
        existing = set(file_listbox.get(0, tk.END))
        added = 0
        for ap in found:
            if ap not in existing:
                file_listbox.insert(tk.END, ap)
                existing.add(ap)
                added += 1
        messagebox.showinfo("Folder Added", f"Added {added} media recording(s) from:\n{folder}")

    def _remove_selected():
        for idx in reversed(file_listbox.curselection()):
            file_listbox.delete(idx)

    def _clear_queue():
        file_listbox.delete(0, tk.END)

    ttk.Button(btn_col, text="Add Media Files...", command=_add_files, width=18).pack(pady=2)
    ttk.Button(btn_col, text="Add Folder...", command=_add_folder, width=18).pack(pady=2)
    ttk.Button(btn_col, text="Remove Selected", command=_remove_selected, width=18).pack(pady=2)
    ttk.Button(btn_col, text="Clear Queue", command=_clear_queue, width=18).pack(pady=2)

    # --- Section B: Hardware, Model & Deliverable Options ---
    cfg_group = ttk.LabelFrame(top_container, text="2. Hardware Acceleration, ASR Model & Output Options", padding=8)
    cfg_group.pack(fill=tk.X, pady=(0, 4))

    row1 = ttk.Frame(cfg_group)
    row1.pack(fill=tk.X, pady=2)

    ttk.Label(row1, text="Model:").pack(side=tk.LEFT)
    model_var = tk.StringVar(value="base.en")
    ttk.Combobox(
        row1,
        textvariable=model_var,
        values=["tiny.en", "base.en", "small.en", "medium.en", "large-v3-turbo"],
        width=13,
        state="readonly",
    ).pack(side=tk.LEFT, padx=(4, 12))

    ttk.Label(row1, text="Device:").pack(side=tk.LEFT)
    device_var = tk.StringVar(value="auto")
    ttk.Combobox(
        row1,
        textvariable=device_var,
        values=["auto", "cpu", "cuda", "rocm", "directml"],
        width=10,
        state="readonly",
    ).pack(side=tk.LEFT, padx=(4, 12))

    ttk.Label(row1, text="Compute Type:").pack(side=tk.LEFT)
    compute_var = tk.StringVar(value="auto")
    ttk.Combobox(
        row1,
        textvariable=compute_var,
        values=["auto", "int8", "float16", "int8_float16", "float32"],
        width=12,
        state="readonly",
    ).pack(side=tk.LEFT, padx=(4, 12))

    ttk.Label(row1, text="Workers:").pack(side=tk.LEFT)
    workers_var = tk.IntVar(value=hw.recommended_num_workers)
    ttk.Spinbox(row1, from_=1, to=32, textvariable=workers_var, width=5).pack(side=tk.LEFT, padx=(4, 12))

    ttk.Label(row1, text="Threads/Worker:").pack(side=tk.LEFT)
    threads_var = tk.IntVar(value=hw.recommended_cpu_threads)
    ttk.Spinbox(row1, from_=1, to=64, textvariable=threads_var, width=5).pack(side=tk.LEFT, padx=(4, 12))

    ttk.Label(row1, text="Batch Size:").pack(side=tk.LEFT)
    batch_var = tk.IntVar(value=hw.recommended_batch_size)
    ttk.Spinbox(row1, from_=1, to=64, textvariable=batch_var, width=5).pack(side=tk.LEFT, padx=(4, 0))

    # Row 2: Checkboxes
    row2 = ttk.Frame(cfg_group)
    row2.pack(fill=tk.X, pady=4)

    enhance_var = tk.BooleanVar(value=True)
    docx_var = tk.BooleanVar(value=True)
    srt_var = tk.BooleanVar(value=False)
    json_var = tk.BooleanVar(value=True)
    mix_var = tk.BooleanVar(value=False)
    skip_var = tk.BooleanVar(value=False)

    ttk.Checkbutton(
        row2,
        text="Far-Field Speech Enhancement (High-Pass + RMS AGC)",
        variable=enhance_var,
    ).pack(side=tk.LEFT, padx=(0, 12))
    ttk.Checkbutton(
        row2,
        text="Generate Word (.docx) Conversation Report",
        variable=docx_var,
    ).pack(side=tk.LEFT, padx=(0, 12))
    ttk.Checkbutton(
        row2,
        text="Export .srt Subtitles",
        variable=srt_var,
    ).pack(side=tk.LEFT, padx=(0, 12))
    ttk.Checkbutton(
        row2,
        text="Export Telemetry (.json)",
        variable=json_var,
    ).pack(side=tk.LEFT, padx=(0, 12))
    ttk.Checkbutton(
        row2,
        text="Mix Multi-Track Video Audio",
        variable=mix_var,
    ).pack(side=tk.LEFT, padx=(0, 12))
    ttk.Checkbutton(
        row2,
        text="Skip Existing",
        variable=skip_var,
    ).pack(side=tk.LEFT)

    # Row 3: Output Dir & Initial Prompt
    row3 = ttk.Frame(cfg_group)
    row3.pack(fill=tk.X, pady=2)
    ttk.Label(row3, text="Output Folder (blank = next to each input file):").pack(side=tk.LEFT)
    outdir_var = tk.StringVar(value="")
    ttk.Entry(row3, textvariable=outdir_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)

    def _browse_outdir():
        d = filedialog.askdirectory(title="Select Output Folder")
        if d:
            outdir_var.set(os.path.abspath(d))

    ttk.Button(row3, text="Browse...", command=_browse_outdir).pack(side=tk.LEFT)

    row4 = ttk.Frame(cfg_group)
    row4.pack(fill=tk.X, pady=2)
    ttk.Label(row4, text="Domain Vocabulary Prompt:").pack(side=tk.LEFT)
    prompt_var = tk.StringVar(value=DEFAULT_INITIAL_PROMPT)
    ttk.Entry(row4, textvariable=prompt_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))

    # --- Section C: Run Controls, Progress & Live Telemetry Console ---
    ctrl_bar = ttk.Frame(bottom_container)
    ctrl_bar.pack(fill=tk.X, pady=(0, 4))

    run_btn = ttk.Button(ctrl_bar, text="▶ Run Batch Transcription", style="Run.TButton")
    run_btn.pack(side=tk.LEFT, padx=(0, 10))

    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = ttk.Progressbar(ctrl_bar, variable=progress_var, maximum=100.0)
    progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    status_var = tk.StringVar(value="Ready.")
    ttk.Label(ctrl_bar, textvariable=status_var, font=("Calibri", 10, "bold")).pack(side=tk.RIGHT)

    log_group = ttk.LabelFrame(
        bottom_container,
        text="3. Live Transcription Stream & Stage Timing Telemetry",
        padding=6,
    )
    log_group.pack(fill=tk.BOTH, expand=True)

    log_text = tk.Text(
        log_group,
        font=("Consolas", 9),
        bg="#101820",
        fg="#E6EDF3",
        insertbackground="white",
        wrap=tk.NONE,
    )
    y_scroll = ttk.Scrollbar(log_group, orient=tk.VERTICAL, command=log_text.yview)
    x_scroll = ttk.Scrollbar(log_group, orient=tk.HORIZONTAL, command=log_text.xview)
    log_text.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)
    y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
    log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    msg_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
    is_running = [False]

    def _append_log(line: str):
        log_text.insert(tk.END, line + "\n")
        log_text.see(tk.END)

    _append_log(f"[System] {hw.summary_line()}")
    for note in hw.notes:
        _append_log(f"[System] {note}")
    _append_log(
        "[System] Supported formats: WAV (16/24-bit PCM & 32-bit float), MP3, MP4, M4A, AAC, FLAC, OGG, OPUS, WMA, MKV, MOV, WEBM, AVI."
    )
    _append_log("[System] Add one or more media files or a folder and click 'Run Batch Transcription'.\n")

    def _poll_queue():
        while not msg_queue.empty():
            item = msg_queue.get_nowait()
            kind = item.get("kind")
            if kind == "log":
                _append_log(item["text"])
            elif kind == "progress":
                progress_var.set(item["pct"])
                status_var.set(item["status"])
            elif kind == "done":
                is_running[0] = False
                run_btn.configure(state=tk.NORMAL)
                progress_var.set(100.0)
                status_var.set(item["status"])
                if item.get("error"):
                    messagebox.showerror("Transcription Error", item["error"])
        root.after(80, _poll_queue)

    def _start_run():
        if is_running[0]:
            return
        files = list(file_listbox.get(0, tk.END))
        if not files:
            messagebox.showwarning("Empty Queue", "Please add at least one audio/video file or folder to the queue.")
            return

        is_running[0] = True
        run_btn.configure(state=tk.DISABLED)
        progress_var.set(0.0)
        status_var.set("Initializing model...")
        _append_log("=" * 84)

        cfg = {
            "files": files,
            "model_size": model_var.get(),
            "device": device_var.get(),
            "compute_type": compute_var.get(),
            "num_workers": int(workers_var.get()),
            "cpu_threads": int(threads_var.get()),
            "batch_size": int(batch_var.get()),
            "enhance_speech": bool(enhance_var.get()),
            "mix_all_audio_streams": bool(mix_var.get()),
            "generate_docx": bool(docx_var.get()),
            "generate_srt": bool(srt_var.get()),
            "generate_json": bool(json_var.get()),
            "skip_existing": bool(skip_var.get()),
            "output_dir": outdir_var.get().strip() or None,
            "initial_prompt": prompt_var.get().strip() or DEFAULT_INITIAL_PROMPT,
        }

        def _worker():
            try:
                engine = TranscriberEngine(
                    model_size=cfg["model_size"],
                    device=cfg["device"],
                    compute_type=cfg["compute_type"],
                    cpu_threads=cfg["cpu_threads"],
                    num_workers=cfg["num_workers"],
                    batch_size=cfg["batch_size"],
                    enhance_speech=cfg["enhance_speech"],
                    mix_all_audio_streams=cfg["mix_all_audio_streams"],
                    initial_prompt=cfg["initial_prompt"],
                    log_fn=lambda m: msg_queue.put({"kind": "log", "text": m}),
                )

                def _prog_cb(file_idx: int, total_files: int, file_frac: float, msg: str):
                    overall_pct = ((file_idx - 1 + file_frac) / max(1, total_files)) * 100.0
                    msg_queue.put({
                        "kind": "progress",
                        "pct": overall_pct,
                        "status": f"File {file_idx}/{total_files} ({overall_pct:.1f}%)",
                    })

                summary = engine.transcribe_batch(
                    inputs=cfg["files"],
                    output_dir=cfg["output_dir"],
                    skip_existing=cfg["skip_existing"],
                    generate_docx=cfg["generate_docx"],
                    generate_srt=cfg["generate_srt"],
                    generate_json=cfg["generate_json"],
                    progress_callback=_prog_cb,
                )
                if len(summary.results) == 0:
                    msg_queue.put({
                        "kind": "done",
                        "status": "Failed (0 files completed — see console log)",
                        "error": "No media files were transcribed. Check the console log for details.",
                    })
                else:
                    msg_queue.put({
                        "kind": "done",
                        "status": f"Completed {len(summary.results)} file(s) at {summary.overall_speedup_x_rt:.1f}x RT!",
                    })
            except Exception as e:
                msg_queue.put({"kind": "log", "text": f"[ERROR] {e}"})
                msg_queue.put({"kind": "done", "status": "Error", "error": str(e)})

        threading.Thread(target=_worker, daemon=True).start()

    run_btn.configure(command=_start_run)
    root.after(80, _poll_queue)
    root.mainloop()


# ==============================================================================
# 2. Zero-Dependency Local Web GUI (http.server on http://127.0.0.1:8765)
# ==============================================================================
WEB_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Audio Data Transcription Toolkit</title>
<style>
  body { font-family: 'Segoe UI', Calibri, sans-serif; margin: 0; background: #F4F6F9; color: #222; }
  header { background: #1B365D; color: white; padding: 16px 24px; }
  header h1 { margin: 0 0 4px 0; font-size: 20px; }
  header div { color: #D0E2F2; font-size: 13px; font-family: Consolas, monospace; }
  .container { max-width: 1100px; margin: 16px auto; padding: 0 16px; }
  .card { background: white; border-radius: 6px; padding: 16px; margin-bottom: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
  .card h2 { margin-top: 0; font-size: 15px; color: #006699; border-bottom: 1px solid #E1E7EF; padding-bottom: 6px; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; margin-bottom: 10px; }
  label { font-size: 13px; font-weight: 600; }
  input[type="text"], input[type="number"], select, textarea {
    padding: 6px 8px; border: 1px solid #CBD5E1; border-radius: 4px; font-size: 13px;
  }
  textarea { width: 100%; height: 70px; font-family: Consolas, monospace; box-sizing: border-box; }
  button {
    background: #1B365D; color: white; border: none; padding: 9px 18px; border-radius: 4px;
    font-size: 14px; font-weight: bold; cursor: pointer;
  }
  button:hover { background: #006699; }
  button:disabled { background: #94A3B8; cursor: not-allowed; }
  .progress-wrap { flex: 1; background: #E2E8F0; height: 18px; border-radius: 9px; overflow: hidden; }
  .progress-bar { background: #006699; height: 100%; width: 0%; transition: width 0.2s; }
  pre#console {
    background: #101820; color: #E6EDF3; padding: 12px; border-radius: 5px;
    height: 360px; overflow-y: auto; font-family: Consolas, monospace; font-size: 12px; margin: 0;
  }
</style>
</head>
<body>
<header>
  <h1>Audio Data Transcription & Conversation Report Toolkit</h1>
  <div id="hw-banner">Loading hardware profile...</div>
</header>
<div class="container">
  <div class="card">
    <h2>1. Input Audio or Video Files, Folders, or Glob Patterns (One Path per Line)</h2>
    <textarea id="inputs" placeholder="D:\\Recordings\\field_session_01.wav&#10;C:\\Recordings\\architecture_meeting.mp4&#10;C:\\Recordings\\voice_memo.mp3"></textarea>
    <div class="row" style="margin-top:8px;">
      <label>Output Folder (optional):</label>
      <input type="text" id="output_dir" style="flex:1;" placeholder="Leave blank to save next to each input file">
    </div>
  </div>
  <div class="card">
    <h2>2. Hardware Acceleration, ASR Model & Deliverables</h2>
    <div class="row">
      <label>Model:
        <select id="model_size">
          <option value="tiny.en">tiny.en</option>
          <option value="base.en" selected>base.en</option>
          <option value="small.en">small.en</option>
          <option value="medium.en">medium.en</option>
          <option value="large-v3-turbo">large-v3-turbo</option>
        </select>
      </label>
      <label>Device:
        <select id="device">
          <option value="auto" selected>auto</option>
          <option value="cpu">cpu</option>
          <option value="cuda">cuda (NVIDIA)</option>
          <option value="rocm">rocm (AMD)</option>
          <option value="directml">directml (AMD/Win)</option>
        </select>
      </label>
      <label>Compute Type:
        <select id="compute_type">
          <option value="auto" selected>auto</option>
          <option value="int8">int8</option>
          <option value="float16">float16</option>
          <option value="int8_float16">int8_float16</option>
          <option value="float32">float32</option>
        </select>
      </label>
      <label>Workers: <input type="number" id="num_workers" value="2" min="1" max="32" style="width:55px;"></label>
      <label>Threads/Worker: <input type="number" id="cpu_threads" value="2" min="1" max="64" style="width:55px;"></label>
      <label>Batch Size: <input type="number" id="batch_size" value="16" min="1" max="64" style="width:55px;"></label>
    </div>
    <div class="row">
      <label><input type="checkbox" id="enhance_speech" checked> Far-Field Speech Enhancement (High-Pass + RMS AGC)</label>
      <label><input type="checkbox" id="generate_docx" checked> Generate Word (.docx) Conversation Report</label>
      <label><input type="checkbox" id="generate_srt"> Export .srt Subtitles</label>
      <label><input type="checkbox" id="generate_json" checked> Export Telemetry (.json)</label>
      <label><input type="checkbox" id="mix_streams"> Mix Multi-Track Video Audio</label>
      <label><input type="checkbox" id="skip_existing"> Skip Existing</label>
    </div>
    <div class="row">
      <label>Domain Prompt:</label>
      <input type="text" id="initial_prompt" style="flex:1;">
    </div>
  </div>
  <div class="card">
    <div class="row">
      <button id="run-btn" onclick="startJob()">▶ Run Batch Transcription</button>
      <div class="progress-wrap"><div class="progress-bar" id="pbar"></div></div>
      <span id="status-lbl" style="font-weight:bold; font-size:13px;">Ready</span>
    </div>
    <pre id="console"></pre>
  </div>
</div>
<script>
let pollTimer = null;
async function loadState() {
  const r = await fetch('/api/state');
  const d = await r.json();
  document.getElementById('hw-banner').textContent = d.hardware_summary;
  document.getElementById('num_workers').value = d.recommended_num_workers;
  document.getElementById('cpu_threads').value = d.recommended_cpu_threads;
  document.getElementById('batch_size').value = d.recommended_batch_size;
  document.getElementById('initial_prompt').value = d.default_prompt;
  if (d.initial_files && d.initial_files.length) {
    document.getElementById('inputs').value = d.initial_files.join('\\n');
  }
}
async function startJob() {
  const lines = document.getElementById('inputs').value.split('\\n').map(s => s.trim()).filter(Boolean);
  if (!lines.length) { alert('Please enter at least one audio/video file or directory path.'); return; }
  document.getElementById('run-btn').disabled = true;
  const payload = {
    inputs: lines,
    output_dir: document.getElementById('output_dir').value.trim() || null,
    model_size: document.getElementById('model_size').value,
    device: document.getElementById('device').value,
    compute_type: document.getElementById('compute_type').value,
    num_workers: parseInt(document.getElementById('num_workers').value, 10),
    cpu_threads: parseInt(document.getElementById('cpu_threads').value, 10),
    batch_size: parseInt(document.getElementById('batch_size').value, 10),
    enhance_speech: document.getElementById('enhance_speech').checked,
    mix_all_audio_streams: document.getElementById('mix_streams').checked,
    generate_docx: document.getElementById('generate_docx').checked,
    generate_srt: document.getElementById('generate_srt').checked,
    generate_json: document.getElementById('generate_json').checked,
    skip_existing: document.getElementById('skip_existing').checked,
    initial_prompt: document.getElementById('initial_prompt').value
  };
  await fetch('/api/run', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload) });
  if (!pollTimer) pollTimer = setInterval(pollLogs, 400);
}
async function pollLogs() {
  const r = await fetch('/api/status');
  const d = await r.json();
  const c = document.getElementById('console');
  c.textContent = d.logs.join('\\n');
  c.scrollTop = c.scrollHeight;
  document.getElementById('pbar').style.width = d.pct + '%';
  document.getElementById('status-lbl').textContent = d.status;
  document.getElementById('run-btn').disabled = d.running;
}
loadState();
pollTimer = setInterval(pollLogs, 1000);
</script>
</body>
</html>
"""


def launch_web_gui(port: int = 8765, initial_files: Optional[List[str]] = None, open_browser: bool = True):
    hw = detect_hardware_profile()
    state: Dict[str, Any] = {
        "running": False,
        "pct": 0.0,
        "status": "Ready",
        "logs": [f"[System] {hw.summary_line()}"] + [f"[System] {n}" for n in hw.notes],
    }
    lock = threading.Lock()

    def _log(msg: str):
        with lock:
            state["logs"].append(msg)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                body = WEB_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/state":
                payload = json.dumps({
                    "hardware_summary": hw.summary_line(),
                    "recommended_num_workers": hw.recommended_num_workers,
                    "recommended_cpu_threads": hw.recommended_cpu_threads,
                    "recommended_batch_size": hw.recommended_batch_size,
                    "default_prompt": DEFAULT_INITIAL_PROMPT,
                    "initial_files": initial_files or [],
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/api/status":
                with lock:
                    payload = json.dumps(state).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)
            else:
                self.send_error(404)

        def do_POST(self):
            if self.path == "/api/run":
                length = int(self.headers.get("Content-Length", "0"))
                cfg = json.loads(self.rfile.read(length).decode("utf-8"))
                with lock:
                    if state["running"]:
                        self.send_response(409)
                        self.end_headers()
                        return
                    state["running"] = True
                    state["pct"] = 0.0
                    state["status"] = "Starting..."
                    state["logs"].append("=" * 84)

                def _worker():
                    try:
                        engine = TranscriberEngine(
                            model_size=cfg.get("model_size", "base.en"),
                            device=cfg.get("device", "auto"),
                            compute_type=cfg.get("compute_type", "auto"),
                            cpu_threads=cfg.get("cpu_threads"),
                            num_workers=cfg.get("num_workers"),
                            batch_size=cfg.get("batch_size"),
                            enhance_speech=bool(cfg.get("enhance_speech", True)),
                            mix_all_audio_streams=bool(cfg.get("mix_all_audio_streams", False)),
                            initial_prompt=cfg.get("initial_prompt") or DEFAULT_INITIAL_PROMPT,
                            log_fn=_log,
                        )

                        def _prog(f_idx: int, tot: int, frac: float, msg: str):
                            pct = ((f_idx - 1 + frac) / max(1, tot)) * 100.0
                            with lock:
                                state["pct"] = round(pct, 1)
                                state["status"] = f"File {f_idx}/{tot} ({pct:.1f}%)"

                        summary = engine.transcribe_batch(
                            inputs=cfg["inputs"],
                            output_dir=cfg.get("output_dir"),
                            recursive=True,
                            skip_existing=bool(cfg.get("skip_existing", False)),
                            generate_docx=bool(cfg.get("generate_docx", True)),
                            generate_srt=bool(cfg.get("generate_srt", False)),
                            generate_json=bool(cfg.get("generate_json", True)),
                            progress_callback=_prog,
                        )
                        with lock:
                            state["pct"] = 100.0
                            state["status"] = f"Done ({summary.overall_speedup_x_rt:.1f}x RT)"
                            state["running"] = False
                    except Exception as e:
                        _log(f"[ERROR] {e}")
                        with lock:
                            state["status"] = f"Error: {e}"
                            state["running"] = False

                threading.Thread(target=_worker, daemon=True).start()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"ok":true}')
            else:
                self.send_error(404)

    url = f"http://127.0.0.1:{port}"
    print(f"Audio Data Transcription Web GUI listening at: {url}", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down Audio Data Transcription Web GUI.")


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Audio Data Transcription & Report GUI (Desktop tkinter or Local Web UI)."
    )
    parser.add_argument("inputs", nargs="*", help="Optional initial media files or folders to populate in the queue")
    parser.add_argument("--web", action="store_true", help="Launch local browser Web GUI instead of desktop window")
    parser.add_argument("--port", type=int, default=8765, help="Port for --web GUI (default: 8765)")
    args = parser.parse_args()

    if args.web:
        launch_web_gui(port=args.port, initial_files=args.inputs)
        return

    try:
        import tkinter as tk
        test_root = tk.Tk()
        test_root.destroy()
        launch_desktop_gui(initial_files=args.inputs)
    except Exception as e:
        print(
            f"[GUI] Desktop display unavailable ({e}); launching built-in Local Web GUI on http://127.0.0.1:{args.port}...",
            flush=True,
        )
        launch_web_gui(port=args.port, initial_files=args.inputs, open_browser=False)


if __name__ == "__main__":
    main()
