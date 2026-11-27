#!/usr/bin/env python3
"""
Root entry-point launcher for the Audio Data Transcription Toolkit GUI.
Allows running `python gui.py` directly after unzipping the repository without
requiring `pip install -e .`.
"""

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from audio_transcriber.gui import main

if __name__ == "__main__":
    main()

