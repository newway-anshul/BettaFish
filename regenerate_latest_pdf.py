"""
Regenerate the latest report PDF using the new SVG vector chart feature.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add project path
sys.path.insert(0, str(Path(__file__).parent))

from ReportEngine.renderers import PDFRenderer

def find_latest_report():
    """
    Find the latest report IR JSON in `final_reports/ir`.

    Select the first file by descending modification time. If the directory
    or files are missing, log an error and return None.

    Returns:
        Path | None: Latest IR file path, or None if not found.
    """
    ir_dir = Path("final_reports/ir")

    if not ir_dir.exists():
        logger.error(f"Report directory does not exist: {ir_dir}")
        return None

    # Get all JSON files and sort by modification time
    json_files = sorted(ir_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)

    if not json_files:
        logger.error("No report files found")
        return None

    latest_file = json_files[0]
    logger.info(f"Latest report found: {latest_file.name}")

    return latest_file

def load_document_ir(file_path):
    """
    Read Document IR JSON from the given path and count chapters/charts.

    Return None on parse failure. On success, print chapter and chart counts
    to help confirm input report scale.

    Args:
        file_path: IR file path

    Returns:
        dict | None: Parsed Document IR; None on failure.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        logger.info(f"Report loaded successfully: {file_path.name}")

        # Count charts
        chart_count = 0
        chapters = document_ir.get('chapters', [])

        def count_charts(blocks):
            """Recursively count Chart.js charts in a block list."""
            count = 0
            for block in blocks:
                if isinstance(block, dict):
                    if block.get('type') == 'widget' and block.get('widgetType', '').startswith('chart.js'):
                        count += 1
                    # Recursively handle nested blocks
                    nested = block.get('blocks')
                    if isinstance(nested, list):
                        count += count_charts(nested)
            return count

        for chapter in chapters:
            blocks = chapter.get('blocks', [])
            chart_count += count_charts(blocks)

        logger.info(f"Report contains {len(chapters)} chapters and {chart_count} charts")

        return document_ir

    except Exception as e:
        logger.error(f"Failed to load report: {e}")
        return None

def generate_pdf_with_vector_charts(document_ir, output_path, ir_file_path=None):
    """
    Render Document IR to a PDF with SVG vector charts using PDFRenderer.

    Enable layout optimization, print file size and success info after generation;
    return None on exception.

    Args:
        document_ir: Complete Document IR
        output_path: Target PDF path
        ir_file_path: Optional IR file path; when provided, fixes are auto-saved

    Returns:
        Path | None: Generated PDF path on success, None on failure.
    """
    try:
        logger.info("=" * 60)
        logger.info("Starting PDF generation (with vector charts)")
        logger.info("=" * 60)

        # Create PDF renderer
        renderer = PDFRenderer()

        # Render PDF; pass ir_file_path to auto-save post-fix changes
        result_path = renderer.render_to_pdf(
            document_ir,
            output_path,
            optimize_layout=True,
            ir_file_path=str(ir_file_path) if ir_file_path else None
        )

        logger.info("=" * 60)
        logger.info(f"✓ PDF generated successfully: {result_path}")
        logger.info("=" * 60)

        # Display file size
        file_size = result_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        logger.info(f"File size: {size_mb:.2f} MB")

        return result_path

    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        return None

def main():
    """
    Main entry point: regenerate the latest report vector PDF.

    Steps:
        1) Find latest IR file;
        2) Read and summarize report structure;
        3) Build output filename and ensure directory exists;
        4) Call renderer to generate PDF and print output details.

    Returns:
        int: 0 for success, non-zero for failure.
    """
    logger.info("🚀 Regenerating latest report PDF using SVG vector charts")
    logger.info("")

    # 1. Find latest report
    latest_report = find_latest_report()
    if not latest_report:
        logger.error("No report file found")
        return 1

    # 2. Load report data
    document_ir = load_document_ir(latest_report)
    if not document_ir:
        logger.error("Failed to load report")
        return 1

    # 3. Build output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = latest_report.stem.replace("report_ir_", "")
    output_filename = f"report_vector_{report_name}_{timestamp}.pdf"
    output_path = Path("final_reports/pdf") / output_filename

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output path: {output_path}")
    logger.info("")

    # 4. Generate PDF; pass IR path to auto-save post-fix changes
    result = generate_pdf_with_vector_charts(document_ir, output_path, ir_file_path=latest_report)

    if result:
        logger.info("")
        logger.info("🎉 PDF generation completed!")
        logger.info("")
        logger.info("Feature notes:")
        logger.info("  ✓ Charts are rendered in SVG vector format")
        logger.info("  ✓ Supports unlimited scaling without quality loss")
        logger.info("  ✓ Preserves full chart visual fidelity")
        logger.info("  ✓ Line, bar, pie and other charts use vector curves")
        logger.info("")
        logger.info(f"PDF file location: {result.absolute()}")
        return 0
    else:
        logger.error("❌ PDF generation failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
