import json
from pathlib import Path

PROGRESSION_FILE = Path("progression.json")

def load_unlocked_stage() -> int:
    try:
        if PROGRESSION_FILE.exists():
            with open(PROGRESSION_FILE, "r") as f:
                data = json.load(f)
                return int(data.get("highest_unlocked_stage", 1))
    except Exception:
        pass
    return 1

def save_unlocked_stage(stage: int) -> None:
    try:
        data = {"highest_unlocked_stage": stage}
        with open(PROGRESSION_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception:
        pass
