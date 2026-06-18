"""Put the app's Python strategy dir on sys.path so the pytest suite can import
the qm_* strategy modules and qm_common. The `stonks` package itself is resolved
from the venv (editable install)."""

import os
import sys

_APP_PYTHON = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "app", "python")
)
if _APP_PYTHON not in sys.path:
    sys.path.insert(0, _APP_PYTHON)
