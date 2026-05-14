import argparse
import json
from collections import defaultdict
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def infer_from_path(path):
    parts = Path(path).parts

    user = "unknown_user"
    mouse = "unknown_mouse"
    grip = "unknown_grip"

    for p in parts:
        low = p.lower()
        if low.startswith("user_"):
            user = p
        if p in ["G102", "X2H", "XliteV3ES"]:
            mouse = p
        if low in ["palm", "claw", "fingertip"]:
            grip = low

    return user, mouse, grip

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_root", default="data/outputs")
    args = parser.parse_args()

    total = defaultdict(int)
    stable = defaultdict(int)

    files = list(Path(args.outputs_root).rglob("*_trial_features_v2.json"))

    for file in files:
        user, mouse, grip = infer_from_path(file)
        data = load_json(file)

        for trial in data.get("trials", []):
            features = trial.get("features", {})
            key = (user, mouse, grip)
            total[key] += 1
            if features.get("pose_stable", 0.0) >= 1.0:
                stable[key] += 1

    print("user,mouse,grip,total,stable,stable_ratio")
    for key in sorted(total.keys()):
        t = total[key]
        s = stable[key]
        ratio = s / t if t else 0
        print(f"{key[0]},{key[1]},{key[2]},{t},{s},{ratio:.3f}")

if __name__ == "__main__":
    main()