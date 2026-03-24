"""
Restitch and render a Markdown report using the latest chapter JSON.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger

# Ensure project modules can be imported
sys.path.insert(0, str(Path(__file__).parent))

from ReportEngine.core import ChapterStorage, DocumentComposer
from ReportEngine.ir import IRValidator
from ReportEngine.renderers import MarkdownRenderer
from ReportEngine.utils.config import settings


def find_latest_run_dir(chapter_root: Path):
    """
    Locate the latest run output directory under the chapter root.

    Scan all subdirectories under `chapter_root`, filter candidates that
    contain `manifest.json`, and pick the newest by modification time.
    If the directory does not exist or no valid manifest is found, log
    an error and return None.

    Args:
        chapter_root: Root directory for chapter outputs (usually settings.CHAPTER_OUTPUT_DIR)

    Returns:
        Path | None: Latest run directory path, or None if not found.
    """
    if not chapter_root.exists():
        logger.error(f"Chapter directory does not exist: {chapter_root}")
        return None

    run_dirs = []
    for candidate in chapter_root.iterdir():
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "manifest.json"
        if manifest_path.exists():
            run_dirs.append((candidate, manifest_path.stat().st_mtime))

    if not run_dirs:
        logger.error("No chapter directories with manifest.json were found")
        return None

    latest_dir = sorted(run_dirs, key=lambda item: item[1], reverse=True)[0][0]
    logger.info(f"Latest run directory found: {latest_dir.name}")
    return latest_dir


def load_manifest(run_dir: Path):
    """
    Read manifest.json from a single run directory.

    On success, return reportId and metadata dictionary. If reading or
    parsing fails, log an error and return (None, None) so the caller
    can terminate early.

    Args:
        run_dir: Chapter output directory containing manifest.json

    Returns:
        tuple[str | None, dict | None]: (report_id, metadata)
    """
    manifest_path = run_dir / "manifest.json"
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        report_id = manifest.get("reportId") or run_dir.name
        metadata = manifest.get("metadata") or {}
        logger.info(f"Report ID: {report_id}")
        if manifest.get("createdAt"):
            logger.info(f"Created at: {manifest['createdAt']}")
        return report_id, metadata
    except Exception as exc:
        logger.error(f"Failed to read manifest: {exc}")
        return None, None


def load_chapters(run_dir: Path):
    """
    Read all chapter JSON files from the specified run directory.

    Reuses ChapterStorage.load_chapters and sorts automatically by order.
    Logs chapter count after loading for completeness checks.

    Args:
        run_dir: Chapter directory for one report run

    Returns:
        list[dict]: List of chapter JSON objects (empty if directory is empty)
    """
    storage = ChapterStorage(settings.CHAPTER_OUTPUT_DIR)
    chapters = storage.load_chapters(run_dir)
    logger.info(f"Loaded chapter count: {len(chapters)}")
    return chapters


def validate_chapters(chapters):
    """
    Perform a quick structural validation of chapters using IRValidator.

    Only logs failed chapters and their first three errors. It does not
    stop the workflow; the goal is to detect potential structure issues
    before restitching.

    Args:
        chapters: List of chapter JSON objects
    """
    validator = IRValidator()
    invalid = []
    for chapter in chapters:
        ok, errors = validator.validate_chapter(chapter)
        if not ok:
            invalid.append((chapter.get("chapterId") or "unknown", errors))

    if invalid:
        logger.warning(f"{len(invalid)} chapters failed structural validation; stitching will continue:")
        for chapter_id, errors in invalid:
            preview = "; ".join(errors[:3])
            logger.warning(f"  - {chapter_id}: {preview}")
    else:
        logger.info("Chapter structure validation passed")


def stitch_document(report_id, metadata, chapters):
    """
    Stitch chapters and metadata into a complete Document IR.

    Uses DocumentComposer to unify chapter order and global metadata,
    then logs the final chapter and chart counts.

    Args:
        report_id: Report ID (from manifest or directory name)
        metadata: Global metadata from manifest
        chapters: Loaded chapter list

    Returns:
        dict: Complete Document IR object
    """
    composer = DocumentComposer()
    document_ir = composer.build_document(report_id, metadata, chapters)
    logger.info(
        f"Stitching complete: {len(document_ir.get('chapters', []))} chapters, "
        f"{count_charts(document_ir)} charts"
    )
    return document_ir


def count_charts(document_ir):
    """
    Count the number of Chart.js charts in the entire Document IR.

    Traverses chapter blocks and recursively finds widget entries whose
    type starts with `chart.js`, for a quick chart-scale estimate.

    Args:
        document_ir: Complete Document IR

    Returns:
        int: Total chart count
    """
    chart_count = 0
    for chapter in document_ir.get("chapters", []):
        blocks = chapter.get("blocks", [])
        chart_count += _count_chart_blocks(blocks)
    return chart_count


def _count_chart_blocks(blocks):
    """
    Recursively count Chart.js components in a block list.

    Supports nested blocks/list/table structures so charts at all levels
    are counted.

    Args:
        blocks: Block list at any nesting level

    Returns:
        int: Counted number of chart.js charts
    """
    count = 0
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "widget" and str(block.get("widgetType", "")).startswith("chart.js"):
            count += 1
        nested = block.get("blocks")
        if isinstance(nested, list):
            count += _count_chart_blocks(nested)
        if block.get("type") == "list":
            for item in block.get("items", []):
                if isinstance(item, list):
                    count += _count_chart_blocks(item)
        if block.get("type") == "table":
            for row in block.get("rows", []):
                for cell in row.get("cells", []):
                    if isinstance(cell, dict):
                        cell_blocks = cell.get("blocks", [])
                        if isinstance(cell_blocks, list):
                            count += _count_chart_blocks(cell_blocks)
    return count


def save_document_ir(document_ir, base_name, timestamp):
    """
    Save the restitched full Document IR to disk.

    Writes to `settings.DOCUMENT_IR_OUTPUT_DIR` using the filename pattern
    `report_ir_{slug}_{timestamp}_regen.json`, ensures the directory exists,
    and returns the saved path.

    Args:
        document_ir: Fully stitched IR
        base_name: Safe filename fragment from topic/title
        timestamp: Timestamp string for distinguishing repeated regenerations

    Returns:
        Path: Saved IR file path
    """
    output_dir = Path(settings.DOCUMENT_IR_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    ir_filename = f"report_ir_{base_name}_{timestamp}_regen.json"
    ir_path = output_dir / ir_filename
    ir_path.write_text(json.dumps(document_ir, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"IR saved: {ir_path}")
    return ir_path


def render_markdown(document_ir, base_name, timestamp, ir_path=None):
    """
    Render Document IR to Markdown with MarkdownRenderer and save it.

    Writes output to `final_reports/md` and logs generated file size for
    quick output verification.

    Args:
        document_ir: Fully stitched IR
        base_name: Filename fragment (from report topic/title)
        timestamp: Timestamp string
        ir_path: Optional IR file path; when provided, fixes are auto-saved

    Returns:
        Path: Generated Markdown file path
    """
    renderer = MarkdownRenderer()
    # Pass ir_file_path so fixes are auto-saved
    markdown_content = renderer.render(document_ir, ir_file_path=str(ir_path) if ir_path else None)

    output_dir = Path(settings.OUTPUT_DIR) / "md"
    output_dir.mkdir(parents=True, exist_ok=True)
    md_filename = f"report_md_{base_name}_{timestamp}.md"
    md_path = output_dir / md_filename
    md_path.write_text(markdown_content, encoding="utf-8")

    file_size_kb = md_path.stat().st_size / 1024
    logger.info(f"Markdown generated successfully: {md_path} ({file_size_kb:.1f} KB)")
    return md_path


def build_slug(text):
    """
    Convert a topic/title to a filesystem-safe fragment.

    Keeps only letters/digits/spaces/underscores/hyphens, normalizes spaces
    to underscores, and limits length to 60 characters to avoid long filenames.

    Args:
        text: Original topic or title

    Returns:
        str: Sanitized safe string
    """
    text = str(text or "report")
    sanitized = "".join(c for c in text if c.isalnum() or c in (" ", "-", "_")).strip()
    sanitized = sanitized.replace(" ", "_")
    return sanitized[:60] or "report"


def main():
    """
    Main entry: load latest chapters, stitch IR, and render Markdown.

    Flow:
        1) Find the latest chapter run directory and read manifest;
        2) Load chapters and run structural validation (warnings only);
        3) Stitch full IR and save an IR copy;
        4) Render Markdown and print output paths.

    Returns:
        int: 0 means success, non-zero means failure.
    """
    logger.info("🚀 Restitching and rendering Markdown with the latest LLM chapters")

    chapter_root = Path(settings.CHAPTER_OUTPUT_DIR)
    latest_run = find_latest_run_dir(chapter_root)
    if not latest_run:
        return 1

    report_id, metadata = load_manifest(latest_run)
    if not report_id or metadata is None:
        return 1

    chapters = load_chapters(latest_run)
    if not chapters:
        logger.error("No chapter JSON found; stitching cannot proceed")
        return 1

    validate_chapters(chapters)

    document_ir = stitch_document(report_id, metadata, chapters)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = build_slug(
        metadata.get("query") or metadata.get("title") or metadata.get("reportId") or report_id
    )

    ir_path = save_document_ir(document_ir, base_name, timestamp)
    # Pass ir_path so fixed charts are auto-saved to the IR file
    md_path = render_markdown(document_ir, base_name, timestamp, ir_path=ir_path)

    logger.info("")
    logger.info("🎉 Markdown stitching and rendering completed")
    logger.info(f"IR file: {ir_path.resolve()}")
    logger.info(f"Markdown file: {md_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
