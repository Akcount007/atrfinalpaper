import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import yaml


def load_cfg(path=None):
    with open(path or ROOT / "configs" / "protocol.yaml") as f:
        return yaml.safe_load(f)
