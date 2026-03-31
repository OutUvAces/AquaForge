"""
Legacy entry point — prefer project root ``app.py`` or ``run_web.bat``.

  py -3 -m streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vessel_detection.web_ui import main

if __name__ == "__main__":
    main()
