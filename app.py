"""
AquaForge — web UI (Streamlit).

**Easiest (no terminal):** double-click ``Open Web App (no terminal).vbs``,
then use the bookmark file ``AquaForge.url`` or http://127.0.0.1:8501

**Terminal:** ``py -3 -m streamlit run app.py`` or ``run_web.bat``

**Daily flow:** open a scene → review spots → save labels → **Advanced → Train first AquaForge model** once if weights are missing, then **Retrain AquaForge** for longer training when you want a new model.

**Main:** large close-up, **On image** toggles, **Back / Next** and save buttons. **Left:** scene + refresh; **Advanced** has retrain, downloads, exports, label fixer.

The app only answers at that URL while this computer is running the server.
For a public URL with no local server, host the app (e.g. Streamlit Community Cloud).
"""

import importlib
import sys


def _streamlit_refresh_review_card_export() -> None:
    """
    Streamlit re-executes this file on every interaction, but Python keeps ``sys.modules``
    cached — edits to ``aquaforge/review_card_export.py`` would otherwise be ignored
    until the server process restarts. Reload that module and re-bind the function ``web_ui``
    imported at load time so **Generate preview ZIP** always uses the current code.
    """
    name = "aquaforge.review_card_export"
    if name not in sys.modules:
        return
    importlib.reload(sys.modules[name])
    rce = sys.modules[name]
    wu = sys.modules.get("aquaforge.web_ui")
    if wu is not None and hasattr(wu, "export_review_cards_zip"):
        wu.export_review_cards_zip = rce.export_review_cards_zip


_streamlit_refresh_review_card_export()

from aquaforge.web_ui import main

main()
