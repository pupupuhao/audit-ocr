from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .utils import ensure_dir, make_json_safe, page_file_name, write_json

_RAPID_OCR_ENGINE: Any | None = None
_RAPID_OCR_ENGINE_KEY: tuple[str | None, str | None, str | None] | None = None


def _get_rapid_ocr_engine(
    det_model_path: str | None = None,
    rec_model_path: str | None = None,
    rec_keys_path: str | None = None,
) -> Any:
    global _RAPID_OCR_ENGINE, _RAPID_OCR_ENGINE_KEY
    engine_key = (det_model_path, rec_model_path, rec_keys_path)
    if _RAPID_OCR_ENGINE is None or _RAPID_OCR_ENGINE_KEY != engine_key:
        from rapidocr_onnxruntime import RapidOCR

        kwargs = {}
        if det_model_path:
            kwargs["det_model_path"] = det_model_path
        if rec_model_path:
            kwargs["rec_model_path"] = rec_model_path
        if rec_keys_path:
            kwargs["rec_keys_path"] = rec_keys_path
        _RAPID_OCR_ENGINE = RapidOCR(**kwargs)
        _RAPID_OCR_ENGINE_KEY = engine_key
    return _RAPID_OCR_ENGINE


def _normalize_rapid_result(raw_result: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not raw_result:
        return items

    for row in raw_result:
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        bbox, text, score = row[0], row[1], row[2]
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            score_value = 0.0
        items.append(
            {
                "text": str(text),
                "score": score_value,
                "bbox": bbox,
            }
        )
    return items


def run_rapid_text_ocr(
    image_path: str,
    output_dir: str,
    page_no: int,
    det_model_path: str | None = None,
    rec_model_path: str | None = None,
    rec_keys_path: str | None = None,
) -> dict[str, Any]:
    output = ensure_dir(output_dir)
    engine = _get_rapid_ocr_engine(
        det_model_path=det_model_path,
        rec_model_path=rec_model_path,
        rec_keys_path=rec_keys_path,
    )
    started_at = time.perf_counter()
    raw_result, rapid_elapse = engine(image_path)
    wall_seconds = round(time.perf_counter() - started_at, 4)

    result = {
        "image_path": str(Path(image_path)),
        "page_no": page_no,
        "engine": "rapidocr_onnxruntime",
        "det_model_path": det_model_path,
        "rec_model_path": rec_model_path,
        "rec_keys_path": rec_keys_path,
        "elapsed_seconds": wall_seconds,
        "rapid_elapse": rapid_elapse,
        "items": _normalize_rapid_result(raw_result),
        "raw_result": make_json_safe(raw_result),
    }

    json_path = output / page_file_name(page_no, "_rapid_ocr.json")
    txt_path = output / page_file_name(page_no, "_rapid_ocr.txt")
    write_json(json_path, result)

    lines = [f"{item['score']:.4f}\t{item['text']}" for item in result["items"]]
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return result
