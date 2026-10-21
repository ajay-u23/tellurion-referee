import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "src", ROOT / "scripts"):
    sys.path.insert(0, str(p))
