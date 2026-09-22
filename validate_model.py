from __future__ import annotations

import argparse
import json
from pathlib import Path

from road_damage_detector import validate_model_classes

parser = argparse.ArgumentParser(description="Validate the classes embedded in RoadGuard weights.")
parser.add_argument("--model", default="roadguard_best.pt")
args = parser.parse_args()
print(json.dumps(validate_model_classes(Path(args.model)), indent=2))
