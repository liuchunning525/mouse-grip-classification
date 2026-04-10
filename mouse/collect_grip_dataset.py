import os
import glob
import json
import argparse
import subprocess
import sys
from typing import Dict, List, Optional


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def run_cmd(args: List[str]):
    print("\n[RUN]", " ".join(f'"{a}"' if " " in a else a for a in args))
    subprocess.run(args, check=True)


def find_generated_hand_json(output_root: str, stem: str) -> Optional[str]:
    candidates = [
        os.path.join(output_root, f"{stem}_hand_landmarks.json"),
        os.path.join(output_root, f"{stem}_hand", f"{stem}_hand_landmarks.json"),
        os.path.join(output_root, f"{stem}_hand.json", f"{stem}_hand_landmarks.json"),
        os.path.join(output_root, f"{stem}_hand.json"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    pattern = os.path.join(output_root, f"{stem}_hand*", "*_hand_landmarks.json")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def process_one_image(
    image_path: str,
    ref_image: str,
    ref_pose: str,
    work_area_json: str,
    output_dir: str,
    hand_script: str,
    refine_script: str,
) -> Dict:
    stem = os.path.splitext(os.path.basename(image_path))[0]

    # 1) Hand landmarks. This script may treat --output as a directory.
    hand_output_target = os.path.join(output_dir, f"{stem}_hand")
    run_cmd([
        sys.executable,
        hand_script,
        "--image",
        image_path,
        "--output",
        hand_output_target,
    ])

    hand_json = find_generated_hand_json(output_dir, stem)
    if hand_json is None:
        raise FileNotFoundError(f"No hand landmark json found for {stem} under {output_dir}")

    # 2) Mouse pose relative to reference.
    current_pose_json = os.path.join(output_dir, f"{stem}_current_pose.json")
    run_cmd([
        sys.executable,
        refine_script,
        "--ref_image",
        ref_image,
        "--cur_image",
        image_path,
        "--ref_pose",
        ref_pose,
        "--work_area_json",
        work_area_json,
        "--hand_json",
        hand_json,
        "--output_dir",
        output_dir,
        "--output_name",
        os.path.basename(current_pose_json),
    ])

    record = {
        "image": image_path,
        "hand_json": hand_json,
        "current_pose_json": current_pose_json,
    }
    return record


def infer_label(stem: str) -> str:
    lower = stem.lower()
    for label in ["palm", "claw", "fingertip"]:
        if label in lower:
            return label
    return "unknown"


def collect_from_images(
    images_dir: str,
    ref_image: str,
    ref_pose: str,
    work_area_json: str,
    output_dir: str,
    labels_dir: str,
    hand_script: str,
    refine_script: str,
    pattern: str,
):
    ensure_dir(output_dir)
    ensure_dir(labels_dir)

    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_paths.extend(glob.glob(os.path.join(images_dir, ext)))
    image_paths = sorted(image_paths)
    if pattern:
        image_paths = [p for p in image_paths if pattern.lower() in os.path.basename(p).lower()]

    # Exclude the reference image if it lives in the same folder.
    ref_abs = os.path.abspath(ref_image)
    image_paths = [p for p in image_paths if os.path.abspath(p) != ref_abs]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    summary = []
    for image_path in image_paths:
        stem = os.path.splitext(os.path.basename(image_path))[0]
        try:
            rec = process_one_image(
                image_path=image_path,
                ref_image=ref_image,
                ref_pose=ref_pose,
                work_area_json=work_area_json,
                output_dir=output_dir,
                hand_script=hand_script,
                refine_script=refine_script,
            )
            label_data = {
                "sample_id": stem,
                "label": infer_label(stem),
                **rec,
            }
            label_path = os.path.join(labels_dir, f"{stem}_label.json")
            with open(label_path, "w", encoding="utf-8") as f:
                json.dump(label_data, f, indent=2, ensure_ascii=False)
            summary.append(label_data)
            print(f"✅ 完成: {stem}")
        except Exception as e:
            print(f"❌ 失败: {stem} -> {e}")

    summary_path = os.path.join(labels_dir, "collection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n处理完成: 成功 {len(summary)} 张")
    print(f"标签汇总: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Process all sample images into hand + pose jsons for the current project layout.")
    parser.add_argument("--images_dir", default="images", help="Folder containing sample images")
    parser.add_argument("--ref_image", default=os.path.join("images", "mouse_ref.jpg"), help="Reference mouse-only image")
    parser.add_argument("--ref_pose", default=os.path.join("outputs", "mouse_pose_ref.json"), help="Reference mouse pose json")
    parser.add_argument("--work_area_json", default=os.path.join("outputs", "mouse_ref_work_area.json"), help="Reference work area json")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    parser.add_argument("--labels_dir", default=os.path.join("outputs", "labels"), help="Where to write per-sample label json files")
    parser.add_argument("--hand_script", default="extract_hand_landmarks.py", help="Path to hand landmark script")
    parser.add_argument("--refine_script", default="refine_mouse_pose_with_hand.py", help="Path to mouse pose refinement script")
    parser.add_argument("--pattern", default="", help="Only process images whose filename contains this substring")
    args = parser.parse_args()

    for required in [args.images_dir, args.ref_image, args.ref_pose, args.work_area_json, args.hand_script, args.refine_script]:
        if not os.path.exists(required):
            raise FileNotFoundError(f"Required path not found: {required}")

    collect_from_images(
        images_dir=args.images_dir,
        ref_image=args.ref_image,
        ref_pose=args.ref_pose,
        work_area_json=args.work_area_json,
        output_dir=args.output_dir,
        labels_dir=args.labels_dir,
        hand_script=args.hand_script,
        refine_script=args.refine_script,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
