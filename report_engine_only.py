#!/usr/bin/env python
"""
Report Engine Command-Line Version

This is a command-line report generation tool that does not require a frontend.
Main workflow:
1. Check PDF dependencies
2. Get the latest log and Markdown files
3. Directly call Report Engine to generate reports (skip file increment review)
4. Automatically save HTML, PDF (if dependencies are available), and Markdown to final_reports/ (Markdown is generated after PDF)

Usage:
    python report_engine_only.py [options]

Options:
    --query QUERY     Specify report topic (optional; extracted from filename by default)
    --skip-pdf        Skip PDF generation (even if dependencies are available)
    --skip-markdown   Skip Markdown generation
    --verbose         Show verbose logs
    --help            Show help information
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from loguru import logger

# Global configuration
VERBOSE = False

# Configure logging
def setup_logger(verbose: bool = False):
    """Set up logging configuration."""
    global VERBOSE
    VERBOSE = verbose

    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="DEBUG" if verbose else "INFO"
    )


def check_dependencies() -> tuple[bool, Optional[str]]:
    """
    Check system dependencies required for PDF generation.

    Returns:
        tuple: (is_available: bool, message: str)
            - is_available: Whether PDF functionality is available
            - message: Dependency check result message
    """
    logger.info("=" * 70)
    logger.info("Step 1/4: Check system dependencies")
    logger.info("=" * 70)

    try:
        from ReportEngine.utils.dependency_check import check_pango_available
        is_available, message = check_pango_available()

        if is_available:
            logger.success("✓ PDF dependency check passed; both HTML and PDF will be generated")
        else:
            logger.warning("⚠ Missing PDF dependencies; only HTML will be generated")
            logger.info("\n" + message)

        return is_available, message
    except Exception as e:
        logger.error(f"Dependency check failed: {e}")
        return False, str(e)


def get_latest_engine_reports() -> Dict[str, str]:
    """
    Get the latest report files from the three engine directories.

    Returns:
        Dict[str, str]: Mapping of engine name to file path
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 2/4: Get the latest analysis engine reports")
    logger.info("=" * 70)

    # Define directories for the three engines
    directories = {
        'insight': 'insight_engine_streamlit_reports',
        'media': 'media_engine_streamlit_reports',
        'query': 'query_engine_streamlit_reports'
    }

    latest_files = {}

    for engine, directory in directories.items():
        if not os.path.exists(directory):
            logger.warning(f"⚠ {engine.capitalize()} Engine directory does not exist: {directory}")
            continue

        # Get all .md files
        md_files = [f for f in os.listdir(directory) if f.endswith('.md')]

        if not md_files:
            logger.warning(f"⚠ No .md files found in {engine.capitalize()} Engine directory")
            continue

        # Get the most recent file
        latest_file = max(
            md_files,
            key=lambda x: os.path.getmtime(os.path.join(directory, x))
        )
        latest_path = os.path.join(directory, latest_file)
        latest_files[engine] = latest_path

        logger.info(f"✓ Found latest report for {engine.capitalize()} Engine")

    if not latest_files:
        logger.error("❌ No engine report files found. Please run analysis engines first.")
        sys.exit(1)

    logger.info(f"\nFound latest reports from {len(latest_files)} engines")

    return latest_files


def confirm_file_selection(latest_files: Dict[str, str]) -> bool:
    """
    Ask the user to confirm whether the selected files are correct.

    Args:
        latest_files: Mapping of engine name to file path

    Returns:
        bool: Returns True if confirmed by the user; otherwise False
    """
    logger.info("\n" + "=" * 70)
    logger.info("Please confirm the selected files below:")
    logger.info("=" * 70)

    for engine, file_path in latest_files.items():
        filename = os.path.basename(file_path)
        # Get file modification time
        mtime = os.path.getmtime(file_path)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')

        logger.info(f"  {engine.capitalize()} Engine:")
        logger.info(f"    Filename: {filename}")
        logger.info(f"    Path: {file_path}")
        logger.info(f"    Modified: {mtime_str}")
        logger.info("")

    logger.info("=" * 70)

    # Prompt user for confirmation
    try:
        response = input("Use the files above to generate the report? [Y/n]: ").strip().lower()

        # Default is yes, so empty input or y/yes means confirm
        if response == '' or response == 'y' or response == 'yes':
            logger.success("✓ User confirmed. Continuing report generation")
            return True
        else:
            logger.warning("✗ Operation cancelled by user")
            return False
    except (KeyboardInterrupt, EOFError):
        logger.warning("\n✗ Operation cancelled by user")
        return False


def load_engine_reports(latest_files: Dict[str, str]) -> list[str]:
    """
    Load content from engine reports.

    Args:
        latest_files: Mapping of engine name to file path

    Returns:
        list[str]: List of report contents
    """
    reports = []

    for engine, file_path in latest_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                reports.append(content)
                logger.debug(f"Loaded {engine} report, length: {len(content)} characters")
        except Exception as e:
            logger.error(f"Failed to load {engine} report: {e}")

    return reports


def extract_query_from_reports(latest_files: Dict[str, str]) -> str:
    """
    Extract query topic from report filenames.

    Args:
        latest_files: Mapping of engine name to file path

    Returns:
        str: Extracted query topic
    """
    # Try extracting topic from filename
    for engine, file_path in latest_files.items():
        filename = os.path.basename(file_path)
        # Assume filename format: report_topic_timestamp.md
        if '_' in filename:
            parts = filename.replace('.md', '').split('_')
            if len(parts) >= 2:
                # Extract middle section as topic
                topic = '_'.join(parts[1:-1]) if len(parts) > 2 else parts[1]
                if topic:
                    return topic

    # Return default if extraction fails
    return "Comprehensive Analysis Report"


def generate_report(reports: list[str], query: str, pdf_available: bool) -> Dict[str, Any]:
    """
    Generate report by calling Report Engine.

    Args:
        reports: List of report contents
        query: Report topic
        pdf_available: Whether PDF functionality is available

    Returns:
        Dict[str, Any]: Dictionary containing generation results
    """
    logger.info("\n" + "=" * 70)
    logger.info("Step 3/4: Generate consolidated report")
    logger.info("=" * 70)
    logger.info(f"Report topic: {query}")
    logger.info(f"Input report count: {len(reports)}")

    try:
        from ReportEngine.agent import ReportAgent

        # Initialize Report Agent
        logger.info("Initializing Report Engine...")
        agent = ReportAgent()

        # Define streaming event handler
        def stream_handler(event_type: str, payload: Dict[str, Any]):
            """Handle streaming events from Report Engine."""
            if event_type == 'stage':
                stage = payload.get('stage', '')
                if stage == 'agent_start':
                    logger.info(f"Starting report generation: {payload.get('report_id', '')}")
                elif stage == 'template_selected':
                    logger.info(f"✓ Template selected: {payload.get('template', '')}")
                elif stage == 'template_sliced':
                    logger.info(f"✓ Template parsed, total sections: {payload.get('section_count', 0)}")
                elif stage == 'layout_designed':
                    logger.info(f"✓ Document layout design completed")
                    logger.info(f"  Title: {payload.get('title', '')}")
                elif stage == 'word_plan_ready':
                    logger.info(f"✓ Length planning completed, target sections: {payload.get('chapter_targets', 0)}")
                elif stage == 'chapters_compiled':
                    logger.info(f"✓ Section generation completed, total sections: {payload.get('chapter_count', 0)}")
                elif stage == 'html_rendered':
                    logger.info(f"✓ HTML rendering completed")
                elif stage == 'report_saved':
                    logger.info(f"✓ Report saved")
            elif event_type == 'chapter_status':
                chapter_id = payload.get('chapterId', '')
                title = payload.get('title', '')
                status = payload.get('status', '')
                if status == 'generating':
                    logger.info(f"  Generating section: {title}")
                elif status == 'completed':
                    attempt = payload.get('attempt', 1)
                    warning = payload.get('warning', '')
                    if warning:
                        logger.warning(f"  ✓ Section completed: {title} (Attempt {attempt}, {payload.get('warningMessage', '')})")
                    else:
                        logger.success(f"  ✓ Section completed: {title}")
            elif event_type == 'error':
                logger.error(f"Error: {payload.get('message', '')}")

        # Generate report
        logger.info("Starting report generation. This may take a few minutes...")
        result = agent.generate_report(
            query=query,
            reports=reports,
            forum_logs="",  # Forum logs are not used
            custom_template="",  # Use automatic template selection
            save_report=True,  # Automatically save report
            stream_handler=stream_handler
        )

        logger.success("✓ Report generated successfully!")
        return result

    except Exception as e:
        logger.exception(f"❌ Report generation failed: {e}")
        sys.exit(1)


def save_pdf(document_ir_path: str, query: str) -> Optional[str]:
    """
    Generate and save PDF from an IR file.

    Args:
        document_ir_path: Document IR file path
        query: Report topic

    Returns:
        Optional[str]: PDF file path, or None if failed
    """
    logger.info("\nGenerating PDF file...")

    try:
        # Read IR data
        with open(document_ir_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        # Create PDF renderer
        from ReportEngine.renderers import PDFRenderer
        renderer = PDFRenderer()

        # Prepare output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(
            c for c in query if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        query_safe = query_safe.replace(" ", "_")[:30] or "report"

        pdf_dir = Path("final_reports") / "pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)

        pdf_filename = f"final_report_{query_safe}_{timestamp}.pdf"
        pdf_path = pdf_dir / pdf_filename

        # Generate PDF directly via render_to_pdf; pass IR path for post-fix save
        logger.info(f"Starting PDF rendering: {pdf_path}")
        result_path = renderer.render_to_pdf(
            document_ir,
            pdf_path,
            optimize_layout=True,
            ir_file_path=document_ir_path
        )

        # Show file size
        file_size = result_path.stat().st_size
        size_mb = file_size / (1024 * 1024)
        logger.success(f"✓ PDF saved: {pdf_path}")
        logger.info(f"  File size: {size_mb:.2f} MB")

        return str(result_path)

    except Exception as e:
        logger.exception(f"❌ PDF generation failed: {e}")
        return None


def save_markdown(document_ir_path: str, query: str) -> Optional[str]:
    """
    Generate and save Markdown from an IR file.

    Args:
        document_ir_path: Document IR file path
        query: Report topic

    Returns:
        Optional[str]: Markdown file path, or None if failed
    """
    logger.info("\nGenerating Markdown file...")

    try:
        with open(document_ir_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        from ReportEngine.renderers import MarkdownRenderer
        renderer = MarkdownRenderer()
        # Pass IR file path for post-fix save
        markdown_content = renderer.render(document_ir, ir_file_path=document_ir_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_safe = "".join(
            c for c in query if c.isalnum() or c in (" ", "-", "_")
        ).rstrip()
        query_safe = query_safe.replace(" ", "_")[:30] or "report"

        md_dir = Path("final_reports") / "md"
        md_dir.mkdir(parents=True, exist_ok=True)

        md_filename = f"final_report_{query_safe}_{timestamp}.md"
        md_path = md_dir / md_filename

        md_path.write_text(markdown_content, encoding='utf-8')

        file_size_kb = md_path.stat().st_size / 1024
        logger.success(f"✓ Markdown saved: {md_path}")
        logger.info(f"  File size: {file_size_kb:.1f} KB")

        return str(md_path)

    except Exception as e:
        logger.exception(f"❌ Markdown generation failed: {e}")
        return None


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
                description="Report Engine CLI - Report generation tool without frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python report_engine_only.py
    python report_engine_only.py --query "Civil engineering industry analysis"
  python report_engine_only.py --skip-pdf --verbose

Notes:
    The program automatically gets the latest report files from three engine directories,
    skips file increment review, generates a consolidated report directly,
    and generates Markdown after PDF by default.
        """
    )

    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Specify report topic (auto-extracted from filename by default)'
    )

    parser.add_argument(
        '--skip-pdf',
        action='store_true',
        help='Skip PDF generation (even if supported by the system)'
    )

    parser.add_argument(
        '--skip-markdown',
        action='store_true',
        help='Skip Markdown generation'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show verbose logs'
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command-line arguments
    args = parse_arguments()

    # Configure logging
    setup_logger(verbose=args.verbose)

    logger.info("\n")
    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║" + " " * 20 + "Report Engine CLI Version" + " " * 25 + "║")
    logger.info("╚" + "═" * 68 + "╝")
    logger.info("\n")

    # Step 1: Check dependencies
    pdf_available, _ = check_dependencies()
    markdown_enabled = not args.skip_markdown

    # Disable PDF generation if user explicitly requested --skip-pdf
    if args.skip_pdf:
        logger.info("User specified --skip-pdf; PDF generation will be skipped")
        pdf_available = False

    if not markdown_enabled:
        logger.info("User specified --skip-markdown; Markdown generation will be skipped")

    # Step 2: Get latest files
    latest_files = get_latest_engine_reports()

    # Confirm file selection
    if not confirm_file_selection(latest_files):
        logger.info("\nProgram exited")
        sys.exit(0)

    # Load report contents
    reports = load_engine_reports(latest_files)

    if not reports:
        logger.error("❌ Failed to load any report content")
        sys.exit(1)

    # Extract query topic or use the specified one
    query = args.query if args.query else extract_query_from_reports(latest_files)
    logger.info(f"Using report topic: {query}")

    # Step 3: Generate report
    result = generate_report(reports, query, pdf_available)

    # Step 4: Save files
    logger.info("\n" + "=" * 70)
    logger.info("Step 4/4: Save generated files")
    logger.info("=" * 70)

    # HTML is already saved automatically in generate_report
    html_path = result.get('report_filepath', '')
    ir_path = result.get('ir_filepath', '')
    pdf_path = None
    markdown_path = None

    if html_path:
        logger.success(f"✓ HTML saved: {result.get('report_relative_path', html_path)}")

    # Generate and save PDF if dependencies are available
    if pdf_available:
        if ir_path and os.path.exists(ir_path):
            pdf_path = save_pdf(ir_path, query)
        else:
            logger.warning("⚠ IR file not found; cannot generate PDF")
    else:
        logger.info("⚠ Skipping PDF generation (missing system dependencies or explicitly skipped)")

    # Generate and save Markdown (after PDF)
    if markdown_enabled:
        if ir_path and os.path.exists(ir_path):
            markdown_path = save_markdown(ir_path, query)
        else:
            logger.warning("⚠ IR file not found; cannot generate Markdown")
    else:
        logger.info("⚠ Skipping Markdown generation (user specified)")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.success("✓ Report generation completed!")
    logger.info("=" * 70)
    logger.info(f"Report ID: {result.get('report_id', 'N/A')}")
    logger.info(f"HTML file: {result.get('report_relative_path', 'N/A')}")
    if pdf_available:
        if pdf_path:
            logger.info(f"PDF file: {os.path.relpath(pdf_path, os.getcwd())}")
        else:
            logger.info("PDF file: Generation failed, please check logs")
    else:
        logger.info("PDF file: Skipped")
    if markdown_enabled:
        if markdown_path:
            logger.info(f"Markdown file: {os.path.relpath(markdown_path, os.getcwd())}")
        else:
            logger.info("Markdown file: Generation failed, please check logs")
    else:
        logger.info("Markdown file: Skipped")
    logger.info("=" * 70)
    logger.info("\nProgram finished")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\nProgram interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"\nProgram exited with an exception: {e}")
        sys.exit(1)
