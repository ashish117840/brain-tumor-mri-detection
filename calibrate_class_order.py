from __future__ import annotations

from collections import Counter
from pathlib import Path
import random

import cv2
import numpy as np

from model_loader import CLASS_NAMES, load_trained_model, preprocess_bgr_image


def _iter_images(folder: Path):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def main():
    root = Path(__file__).resolve().parent
    data_dir = root / "sample MRI Images"

    folders = ["glioma", "meningioma", "notumor", "pituitary"]

    print("Data dir:", data_dir)
    print("Current CLASS_NAMES:", CLASS_NAMES)

    model = load_trained_model()

    per_folder: dict[str, Counter[int]] = {}
    n_per_class = 12

    for name in folders:
        folder = data_dir / name
        paths = list(_iter_images(folder))
        if not paths:
            print(f"⚠️ No images found in: {folder}")
            continue

        sample = random.sample(paths, k=min(n_per_class, len(paths)))
        counts: Counter[int] = Counter()

        for p in sample:
            img = cv2.imread(str(p))
            if img is None:
                continue
            x = preprocess_bgr_image(img)
            probs = model.predict(x, verbose=0)[0]
            idx = int(np.argmax(probs))
            counts[idx] += 1

        per_folder[name] = counts

    print("\n=== Predicted index counts (argmax) ===")
    for name, counts in per_folder.items():
        best = counts.most_common(1)[0] if counts else (None, 0)
        print(f"{name:10} -> {dict(counts)}  (most common: {best})")

    # Suggest mapping if possible.
    folder_to_index = {
        name: (counts.most_common(1)[0][0] if counts else None)
        for name, counts in per_folder.items()
    }

    if any(v is None for v in folder_to_index.values()):
        print("\n❌ Not enough data to suggest CLASS_NAMES (missing folder predictions).")
        return

    used = list(folder_to_index.values())
    if len(set(used)) != len(used):
        print("\n⚠️ Conflicting mapping detected (two folders map to same index).")
        print("folder_to_index:", folder_to_index)
        print("Try increasing n_per_class or check model quality.")
        return

    num_classes = max(int(v) for v in used) + 1
    suggested = [""] * num_classes
    for folder, idx in folder_to_index.items():
        suggested[int(idx)] = folder

    if any(s == "" for s in suggested):
        print("\n⚠️ Some class indices were not observed in samples.")

    print("\n✅ Suggested CLASS_NAMES (by model output index):")
    print("CLASS_NAMES = [")
    for s in suggested:
        print(f"    {s!r},")
    print("]")


if __name__ == "__main__":
    main()
