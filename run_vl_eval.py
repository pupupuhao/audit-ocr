from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path
from typing import Any

from src.pdf_converter import pdf_to_images
from src.utils import ensure_dir, pdf_output_name, write_json
from src.vl_ocr import run_vl_ocr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct PaddleOCR-VL evaluation for selected PDF pages.")
    parser.add_argument("--input", default="input", help="Input PDF directory. Default: input")
    parser.add_argument("--output", default="output_vl", help="Output directory. Default: output_vl")
    parser.add_argument("--file", default=None, help="Process one PDF file name or path.")
    parser.add_argument("--dpi", type=int, default=160, help="PDF render DPI. Default: 160")
    parser.add_argument("--max-pages", type=int, default=None, help="Process only the first N selected pages.")
    parser.add_argument("--start-page", type=int, default=None, help="First 1-based page number to process.")
    parser.add_argument("--end-page", type=int, default=None, help="Last 1-based page number to process.")
    return parser.parse_args()


def _resolve_pdf_files(input_dir: Path, file_arg: str | None) -> list[Path]:
    if file_arg:
        file_path = Path(file_arg)
        if not file_path.is_absolute():
            file_path = input_dir / file_path
        return [file_path] if file_path.exists() else []
    if not input_dir.exists():
        return []
    return sorted(path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".pdf")


def _page_no_from_image_path(image_path: str) -> int:
    try:
        return int(Path(image_path).stem.split("_")[-1])
    except (TypeError, ValueError):
        return 0


def _record_error(errors: list[dict[str, Any]], page_errors: list[dict[str, Any]], stage: str, exc: Exception) -> None:
    error = {"stage": stage, "error": str(exc), "traceback": traceback.format_exc()}
    errors.append(error)
    page_errors.append(error)


def process_pdf_vl(
    pdf_path: Path,
    output_root: Path,
    dpi: int,
    max_pages: int | None,
    start_page: int | None = None,
    end_page: int | None = None,
) -> dict[str, Any]:
    pdf_name = pdf_output_name(pdf_path)
    print(f"\n==> VL processing {pdf_path.name}")
    started_at = time.perf_counter()
    errors: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []

    pages_root = output_root / "pages"
    vl_dir = output_root / "vl" / pdf_name
    reports_dir = ensure_dir(output_root / "reports")

    try:
        image_paths = pdf_to_images(
            str(pdf_path),
            str(pages_root),
            dpi=dpi,
            start_page=start_page,
            end_page=end_page,
        )
    except Exception as exc:
        errors.append({"stage": "pdf_to_images", "error": str(exc), "traceback": traceback.format_exc()})
        summary = _build_summary(pdf_path, pages, errors, dpi, started_at)
        write_json(reports_dir / f"{pdf_name}_vl_summary.json", summary)
        return summary

    if max_pages is not None:
        image_paths = image_paths[:max_pages]

    for ordinal, image_path in enumerate(image_paths, start=1):
        page_no = _page_no_from_image_path(image_path) or ordinal
        print(f"  - Page {page_no} ({ordinal}/{len(image_paths)}): {image_path}")
        page_started_at = time.perf_counter()
        page_errors: list[dict[str, Any]] = []
        page_result: dict[str, Any] = {"page_no": page_no, "image_path": image_path, "errors": page_errors}

        try:
            vl_result = run_vl_ocr(image_path, str(vl_dir), page_no)
            page_result["vl_summary"] = vl_result.get("summary", {})
        except Exception as exc:
            _record_error(errors, page_errors, "vl_ocr", exc)
            print(f"    ERROR: {exc}")
            page_result["vl_summary"] = {}

        page_result["elapsed_seconds"] = round(time.perf_counter() - page_started_at, 3)
        pages.append(page_result)

    summary = _build_summary(pdf_path, pages, errors, dpi, started_at)
    write_json(reports_dir / f"{pdf_name}_vl_summary.json", summary)
    return summary


def _build_summary(
    pdf_path: Path,
    pages: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    dpi: int,
    started_at: float,
) -> dict[str, Any]:
    html_pages = [
        page["page_no"]
        for page in pages
        if int(page.get("vl_summary", {}).get("html_files", 0)) > 0
    ]
    markdown_pages = [
        page["page_no"]
        for page in pages
        if page.get("vl_summary", {}).get("has_markdown")
    ]
    return {
        "file_name": pdf_path.name,
        "mode": "direct_vl",
        "dpi": dpi,
        "total_pages": len(pages),
        "processed_pages": len([page for page in pages if not page.get("errors")]),
        "markdown_pages": markdown_pages,
        "markdown_page_count": len(markdown_pages),
        "html_pages": html_pages,
        "html_page_count": len(html_pages),
        "pages": pages,
        "errors": errors,
        "elapsed_seconds": round(time.perf_counter() - started_at, 3),
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_root = Path(args.output)
    reports_dir = ensure_dir(output_root / "reports")

    pdf_files = _resolve_pdf_files(input_dir, args.file)
    if args.file and not pdf_files:
        file_path = Path(args.file)
        if not file_path.is_absolute():
            file_path = input_dir / file_path
        print(f"PDF file not found: {file_path}")
        write_json(reports_dir / "all_files_vl_summary.json", {"file_count": 0, "files": []})
        return
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        write_json(reports_dir / "all_files_vl_summary.json", {"file_count": 0, "files": []})
        return

    summaries = [
        process_pdf_vl(
            pdf_path,
            output_root,
            args.dpi,
            args.max_pages,
            start_page=args.start_page,
            end_page=args.end_page,
        )
        for pdf_path in pdf_files
    ]
    write_json(
        reports_dir / "all_files_vl_summary.json",
        {
            "file_count": len(summaries),
            "total_pages": sum(summary.get("total_pages", 0) for summary in summaries),
            "markdown_page_count": sum(summary.get("markdown_page_count", 0) for summary in summaries),
            "html_page_count": sum(summary.get("html_page_count", 0) for summary in summaries),
            "files": summaries,
        },
    )
    print(f"\nDone. VL reports written to {reports_dir}")


if __name__ == "__main__":
    main()
