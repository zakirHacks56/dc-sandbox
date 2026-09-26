import sys
from pathlib import Path

FIXER = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FIXER))