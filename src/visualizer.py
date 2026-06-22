from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .utils import ensure_dir


def _points_from_bbox(bbox: Any) -> np.ndarray | None:
    try:
        points = np.array(bbox, dtype=np.int32)
    except Exception:
        return None
    if points.ndim != 2 or points.shape[0] < 4 or points.shape[1] != 2:
        return None
    return points


def draw_ocr_boxes(image_path: str, ocr_result: dict, output_path: str) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    for item in ocr_result.get("items", []):
        points = _points_from_bbox(item.get("bbox"))
        if points is None:
            continue
        score = item.get("score", 0)
        text = str(item.get("text", ""))
        cv2.polylines(image, [points], isClosed=True, color=(0, 180, 0), thickness=2)
        x = int(points[:, 0].min())
        y = max(18, int(points[:, 1].min()) - 6)
        label = f"{float(score):.3f} {text[:24]}"
        try:
            cv2.putText(
                image,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        except Exception:
            cv2.putText(
                image,
                f"{float(score):.3f}",
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    target = Path(output_path)
    ensure_dir(target.parent)
    cv2.imwrite(str(target), image)
