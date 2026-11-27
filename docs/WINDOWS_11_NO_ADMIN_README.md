# Run on Windows 11 Without Git, Conda, or Administrator Rights

This guide is for anyone on a standard Windows 11 PC who does not have Git, does not have Conda, and **does not have an administrator password**. You will install Python from the Microsoft Store (per-user), download a `.zip` of this repository from GitHub, and launch the graphical interface by double-clicking `run-gui.bat`.

All audio and video transcription runs **100% locally on your computer**—no recordings or transcripts are ever uploaded to a cloud service.

---

## 1. Install Python from the Microsoft Store (No Admin Needed)

1. Open the **Microsoft Store** (press **Start** and type `Store`).
2. Search for **Python 3.12** (or **Python 3.13**).
3. Select the official package published by the **Python Software Foundation**.
4. Click **Get** / **Install** and wait until installation completes.
5. Verify that **App execution aliases** for `python.exe` and `python3.exe` are enabled (*Settings → Apps → Advanced app settings → App execution aliases*).

---

## 2. Download and Extract the Toolkit ZIP

1. In your browser, open the GitHub repository page:  
   `https://github.com/gzdor/Audio-Data-Transcription-Toolkit`
2. Click the green **Code** button and select **Download ZIP**.
3. Open your **Downloads** folder, right-click `Audio-Data-Transcription-Toolkit-main.zip`, and click **Extract All... → Extract**.
4. Open the extracted folder until you see `run-gui.bat` and `gui.py`.

---

## 3. Double-Click `run-gui.bat`

1. Double-click **`run-gui.bat`**.
2. A console window opens first with the startup splash screen, verifies your Python runtime, and automatically installs the required packages (`faster-whisper`, `python-docx`, `av`, `numpy`, `lxml`) on the first run.
3. The **Audio Data Transcription Toolkit** window opens automatically. Leave the background console window open while using the app.

---

## 4. Transcribe Audio or Video Files & Generate Conversation Reports

1. Click **Add Media Files...** (to select one or more `.wav`, `.mp3`, `.mp4`, `.m4a`, `.aac`, `.flac`, `.ogg`, `.opus`, `.wma`, `.mkv`, `.mov`, `.webm`, or `.avi` files) or click **Add Folder...** to queue an entire folder of recordings.
2. Leave **Device** set to `auto` (automatically selects NVIDIA CUDA, AMD ROCm/DirectML, or multi-core CPU).
3. Keep **Far-Field Speech Enhancement (High-Pass + RMS AGC)** checked if your recording includes speakers sitting across a conference table or room.
4. Keep **Generate Word (.docx) Conversation Report** checked to produce a formatted Word summary alongside the plain-text `.txt` transcript.
5. Click **▶ Run Batch Transcription**.
6. Watch live segment progress and stage timing telemetry in the bottom console. Your `<filename>-Transcript.txt`, `<filename>-Conversation-Report.docx`, and `<filename>-Telemetry.json` files are saved directly next to your input recordings (or in your chosen Output Folder).

