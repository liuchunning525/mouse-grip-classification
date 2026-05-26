import argparse
import csv
import json
import os
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_bad_frame(frame, angle_jump_thresh=None, prev_angle=None):
    status = frame.get("status", {})
    marker_ok = bool(status.get("marker_ok"))
    affine_ok = bool(status.get("affine_ok"))
    hand_ok = bool(status.get("hand_ok"))

    reasons = []

    if not marker_ok:
        reasons.append("marker_bad")
    if not affine_ok:
        reasons.append("affine_bad")
    if not hand_ok:
        reasons.append("hand_bad")

    detected_count = frame.get("mouse", {}).get("detected_count", None)
    if detected_count is not None:
        try:
            if int(detected_count) != 3:
                reasons.append(f"marker_count_{detected_count}")
        except Exception:
            pass

    angle = frame.get("mouse", {}).get("angle_deg", None)
    if angle_jump_thresh is not None and prev_angle is not None and angle is not None:
        try:
            jump = abs(float(angle) - float(prev_angle))
            if jump > angle_jump_thresh:
                reasons.append(f"angle_jump_{jump:.1f}")
        except Exception:
            pass

    return len(reasons) > 0, reasons, angle

def analyze_file(path, angle_jump_thresh=20.0):
    data = load_json(path)
    frames = data.get("frames", [])
    total = len(frames)

    bad = 0
    reason_counts = {}
    prev_angle = None

    for frame in frames:
        is_bad, reasons, angle = is_bad_frame(frame, angle_jump_thresh, prev_angle)
        if is_bad:
            bad += 1
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        if angle is not None:
            prev_angle = angle

    good = total - bad
    return {
        "file": str(path),
        "total_frames": total,
        "good_frames": good,
        "bad_frames": bad,
        "good_ratio": round(good / total, 4) if total else 0,
        "bad_ratio": round(bad / total, 4) if total else 0,
        "reason_counts": reason_counts,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/outputs")
    parser.add_argument("--output", default="data/outputs/frame_quality_summary.csv")
    parser.add_argument("--angle_jump_thresh", type=float, default=20.0)
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.rglob("*_frame_features.json"))

    rows = []
    for file in files:
        rows.append(analyze_file(file, args.angle_jump_thresh))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "total_frames", "good_frames", "bad_frames", "good_ratio", "bad_ratio", "reason_counts"])
        for r in rows:
            writer.writerow([
                r["file"],
                r["total_frames"],
                r["good_frames"],
                r["bad_frames"],
                r["good_ratio"],
                r["bad_ratio"],
                json.dumps(r["reason_counts"], ensure_ascii=False),
            ])

    total_frames = sum(r["total_frames"] for r in rows)
    total_bad = sum(r["bad_frames"] for r in rows)
    total_good = total_frames - total_bad

    print("=== Frame Quality Summary ===")
    print("Files:", len(rows))
    print("Total frames:", total_frames)
    print("Good frames:", total_good)
    print("Bad frames:", total_bad)
    print("Good ratio:", round(total_good / total_frames, 4) if total_frames else 0)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
