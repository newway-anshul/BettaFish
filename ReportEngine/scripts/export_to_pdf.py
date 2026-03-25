#!/usr/bin/env python3
"""
PDF export tool - generate PDF directly with Python (no garbled text)

Usage:
    python ReportEngine/scripts/export_to_pdf.py <report IR JSON file> [output PDF path]

Examples:
    python ReportEngine/scripts/export_to_pdf.py final_reports/ir/report_ir_xxx.json output.pdf
    python ReportEngine/scripts/export_to_pdf.py final_reports/ir/report_ir_xxx.json
"""

import sys
import json
from pathlib import Path
from loguru import logger

from ReportEngine.renderers import PDFRenderer


def export_to_pdf(ir_json_path: str, output_pdf_path: str = None):
    """
    Generate a PDF from an IR JSON file.

    Args:
        ir_json_path: Path to the Document IR JSON file.
        output_pdf_path: Output PDF path (optional, defaults to same name with .pdf).
    """
    ir_path = Path(ir_json_path)

    if not ir_path.exists():
        logger.error(f"File does not exist: {ir_path}")
        return False

    # Read IR data
    logger.info(f"Reading report: {ir_path}")
    with open(ir_path, 'r', encoding='utf-8') as f:
        document_ir = json.load(f)

    # Resolve output path
    if output_pdf_path is None:
        output_pdf_path = ir_path.parent / f"{ir_path.stem}.pdf"
    else:
        output_pdf_path = Path(output_pdf_path)

    # Generate PDF
    logger.info(f"Starting PDF generation...")
    renderer = PDFRenderer()

    try:
        renderer.render_to_pdf(document_ir, output_pdf_path)
        logger.success(f"✓ PDF generated: {output_pdf_path}")
        return True
    except Exception as e:
        logger.error(f"✗ PDF generation failed: {e}")
        logger.exception("Detailed error information:")
        return False


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    ir_json_path = sys.argv[1]
    output_pdf_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Check environment variable
    import os
    if 'DYLD_LIBRARY_PATH' not in os.environ:
        logger.warning("DYLD_LIBRARY_PATH is not set; attempting automatic setup...")
        os.environ['DYLD_LIBRARY_PATH'] = '/opt/homebrew/lib'

    success = export_to_pdf(ir_json_path, output_pdf_path)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
