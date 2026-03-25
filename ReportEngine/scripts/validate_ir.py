#!/usr/bin/env python3
"""
IR document validation tool.

Command-line utility for:
- Scanning all charts and tables in target JSON files
- Reporting structural issues and missing data
- Optionally auto-fixing common issues
- Supporting batch processing

Usage:
    python -m ReportEngine.scripts.validate_ir chapter-030-section-3-0.json
    python -m ReportEngine.scripts.validate_ir *.json --fix
    python -m ReportEngine.scripts.validate_ir ./output/ --recursive --fix --verbose
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to sys.path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from loguru import logger

from ReportEngine.utils.chart_validator import (
    ChartValidator,
    ChartRepairer,
    ValidationResult,
)
from ReportEngine.utils.table_validator import (
    TableValidator,
    TableRepairer,
    TableValidationResult,
)


@dataclass
class BlockIssue:
    """Issue details for a single block."""
    block_type: str
    block_id: str
    path: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_fixable: bool = False


@dataclass
class DocumentReport:
    """Validation report for a document."""
    file_path: str
    total_blocks: int = 0
    chart_count: int = 0
    table_count: int = 0
    wordcloud_count: int = 0
    issues: List[BlockIssue] = field(default_factory=list)
    fixed_count: int = 0

    @property
    def has_issues(self) -> bool:
        return len(self.issues) > 0

    @property
    def error_count(self) -> int:
        return sum(len(issue.errors) for issue in self.issues)

    @property
    def warning_count(self) -> int:
        return sum(len(issue.warnings) for issue in self.issues)


class IRValidator:
    """IR document validator."""

    def __init__(
        self,
        chart_validator: Optional[ChartValidator] = None,
        table_validator: Optional[TableValidator] = None,
        chart_repairer: Optional[ChartRepairer] = None,
        table_repairer: Optional[TableRepairer] = None,
    ):
        self.chart_validator = chart_validator or ChartValidator()
        self.table_validator = table_validator or TableValidator()
        self.chart_repairer = chart_repairer or ChartRepairer(self.chart_validator)
        self.table_repairer = table_repairer or TableRepairer(self.table_validator)

    def validate_document(
        self,
        document: Dict[str, Any],
        file_path: str = "<unknown>",
    ) -> DocumentReport:
        """
        Validate a whole document.

        Args:
            document: IR document data
            file_path: File path (for reporting)

        Returns:
            DocumentReport: Validation report
        """
        report = DocumentReport(file_path=file_path)

        # Iterate over all chapters
        chapters = document.get("chapters", [])
        for chapter_idx, chapter in enumerate(chapters):
            if not isinstance(chapter, dict):
                continue

            chapter_id = chapter.get("chapterId", f"chapter-{chapter_idx}")
            blocks = chapter.get("blocks", [])

            self._validate_blocks(
                blocks,
                f"chapters[{chapter_idx}].blocks",
                chapter_id,
                report,
            )

        return report

    def _validate_blocks(
        self,
        blocks: List[Any],
        path: str,
        chapter_id: str,
        report: DocumentReport,
    ):
        """Recursively validate a list of blocks."""
        if not isinstance(blocks, list):
            return

        for idx, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue

            report.total_blocks += 1
            block_path = f"{path}[{idx}]"
            block_type = block.get("type", "")
            block_id = block.get("widgetId") or block.get("id") or f"block-{idx}"

            # Validate by block type
            if block_type == "widget":
                widget_type = (block.get("widgetType") or "").lower()
                if "chart.js" in widget_type:
                    report.chart_count += 1
                    self._validate_chart(block, block_path, block_id, report)
                elif "wordcloud" in widget_type:
                    report.wordcloud_count += 1
                    self._validate_wordcloud(block, block_path, block_id, report)

            elif block_type == "table":
                report.table_count += 1
                self._validate_table(block, block_path, block_id, report)

            # Recursively process nested blocks
            nested_blocks = block.get("blocks")
            if isinstance(nested_blocks, list):
                self._validate_blocks(nested_blocks, f"{block_path}.blocks", chapter_id, report)

            # Process blocks inside table rows
            if block_type == "table":
                rows = block.get("rows", [])
                for row_idx, row in enumerate(rows):
                    if isinstance(row, dict):
                        cells = row.get("cells", [])
                        for cell_idx, cell in enumerate(cells):
                            if isinstance(cell, dict):
                                cell_blocks = cell.get("blocks", [])
                                self._validate_blocks(
                                    cell_blocks,
                                    f"{block_path}.rows[{row_idx}].cells[{cell_idx}].blocks",
                                    chapter_id,
                                    report,
                                )

            # Process blocks inside list items
            if block_type == "list":
                items = block.get("items", [])
                for item_idx, item in enumerate(items):
                    if isinstance(item, list):
                        self._validate_blocks(
                            item,
                            f"{block_path}.items[{item_idx}]",
                            chapter_id,
                            report,
                        )

    def _validate_chart(
        self,
        block: Dict[str, Any],
        path: str,
        block_id: str,
        report: DocumentReport,
    ):
        """Validate a chart block."""
        result = self.chart_validator.validate(block)

        if not result.is_valid or result.warnings:
            issue = BlockIssue(
                block_type="chart",
                block_id=block_id,
                path=path,
                errors=result.errors,
                warnings=result.warnings,
                is_fixable=result.has_critical_errors(),
            )
            report.issues.append(issue)

    def _validate_table(
        self,
        block: Dict[str, Any],
        path: str,
        block_id: str,
        report: DocumentReport,
    ):
        """Validate a table block."""
        result = self.table_validator.validate(block)

        if not result.is_valid or result.warnings or result.nested_cells_detected:
            issue = BlockIssue(
                block_type="table",
                block_id=block_id,
                path=path,
                errors=result.errors,
                warnings=result.warnings,
                is_fixable=result.nested_cells_detected or result.has_critical_errors(),
            )

            # Add nested-cells warning
            if result.nested_cells_detected:
                issue.warnings.insert(0, "Nested cells structure detected (common LLM error)")

            # Add empty-cell information
            if result.empty_cells_count > 0:
                issue.warnings.append(
                    f"Empty cells: {result.empty_cells_count}/{result.total_cells_count}"
                )

            report.issues.append(issue)

    def _validate_wordcloud(
        self,
        block: Dict[str, Any],
        path: str,
        block_id: str,
        report: DocumentReport,
    ):
        """Validate a word cloud block."""
        errors: List[str] = []
        warnings: List[str] = []

        # Check data structure
        data = block.get("data")
        props = block.get("props", {})

        words_found = False
        words_count = 0

        # Check all possible word-cloud data paths
        data_paths = [
            ("data.words", data.get("words") if isinstance(data, dict) else None),
            ("data.items", data.get("items") if isinstance(data, dict) else None),
            ("data", data if isinstance(data, list) else None),
            ("props.words", props.get("words") if isinstance(props, dict) else None),
            ("props.items", props.get("items") if isinstance(props, dict) else None),
            ("props.data", props.get("data") if isinstance(props, dict) else None),
        ]

        for path_name, value in data_paths:
            if isinstance(value, list) and len(value) > 0:
                words_found = True
                words_count = len(value)

                # Validate word-cloud item format
                for idx, item in enumerate(value[:5]):  # Only check the first 5 items
                    if isinstance(item, dict):
                        word = item.get("word") or item.get("text") or item.get("label")
                        weight = item.get("weight") or item.get("value")
                        if not word:
                            warnings.append(f"{path_name}[{idx}] Missing word/text/label field")
                        if weight is None:
                            warnings.append(f"{path_name}[{idx}] Missing weight/value field")
                    elif not isinstance(item, (str, list, tuple)):
                        warnings.append(f"{path_name}[{idx}] Invalid format")

                break

        if not words_found:
            errors.append("Word cloud data missing: no valid data found in paths like data.words, data.items, or props.words")
        elif words_count == 0:
            warnings.append("Word cloud data is empty")

        if errors or warnings:
            issue = BlockIssue(
                block_type="wordcloud",
                block_id=block_id,
                path=path,
                errors=errors,
                warnings=warnings,
                is_fixable=False,  # Missing word cloud data is usually not auto-fixable
            )
            report.issues.append(issue)

    def repair_document(
        self,
        document: Dict[str, Any],
        report: DocumentReport,
    ) -> Tuple[Dict[str, Any], int]:
        """
        Repair issues in a document.

        Args:
            document: IR document data
            report: Validation report

        Returns:
            Tuple[Dict[str, Any], int]: (repaired document, number of fixes)
        """
        fixed_count = 0

        # Iterate over all chapters
        chapters = document.get("chapters", [])
        for chapter in chapters:
            if not isinstance(chapter, dict):
                continue

            blocks = chapter.get("blocks", [])
            chapter["blocks"], chapter_fixed = self._repair_blocks(blocks)
            fixed_count += chapter_fixed

        return document, fixed_count

    def _repair_blocks(
        self,
        blocks: List[Any],
    ) -> Tuple[List[Any], int]:
        """Recursively repair a list of blocks."""
        if not isinstance(blocks, list):
            return blocks, 0

        fixed_count = 0
        repaired_blocks: List[Any] = []

        for block in blocks:
            if not isinstance(block, dict):
                repaired_blocks.append(block)
                continue

            block_type = block.get("type", "")

            # Repair table
            if block_type == "table":
                result = self.table_repairer.repair(block)
                if result.has_changes():
                    block = result.repaired_block
                    fixed_count += 1
                    logger.info(f"Table repaired: {result.changes}")

            # Repair chart
            elif block_type == "widget":
                widget_type = (block.get("widgetType") or "").lower()
                if "chart.js" in widget_type:
                    result = self.chart_repairer.repair(block)
                    if result.has_changes():
                        block = result.repaired_block
                        fixed_count += 1
                        logger.info(f"Chart repaired: {result.changes}")

            # Recursively process nested blocks
            nested_blocks = block.get("blocks")
            if isinstance(nested_blocks, list):
                block["blocks"], nested_fixed = self._repair_blocks(nested_blocks)
                fixed_count += nested_fixed

            # Process blocks inside table rows
            if block_type == "table":
                rows = block.get("rows", [])
                for row in rows:
                    if isinstance(row, dict):
                        cells = row.get("cells", [])
                        for cell in cells:
                            if isinstance(cell, dict):
                                cell_blocks = cell.get("blocks", [])
                                cell["blocks"], cell_fixed = self._repair_blocks(cell_blocks)
                                fixed_count += cell_fixed

            # Process blocks inside list items
            if block_type == "list":
                items = block.get("items", [])
                for i, item in enumerate(items):
                    if isinstance(item, list):
                        items[i], item_fixed = self._repair_blocks(item)
                        fixed_count += item_fixed

            repaired_blocks.append(block)

        return repaired_blocks, fixed_count


def print_report(report: DocumentReport, verbose: bool = False):
    """Print validation report."""
    print(f"\n{'=' * 60}")
    print(f"File: {report.file_path}")
    print(f"{'=' * 60}")

    print(f"\n📊 Statistics:")
    print(f"  - Total blocks: {report.total_blocks}")
    print(f"  - Charts: {report.chart_count}")
    print(f"  - Tables: {report.table_count}")
    print(f"  - Word clouds: {report.wordcloud_count}")

    if report.has_issues:
        print(f"\n⚠️  Found {len(report.issues)} issues:")
        print(f"  - Errors: {report.error_count}")
        print(f"  - Warnings: {report.warning_count}")

        if verbose:
            for issue in report.issues:
                print(f"\n  [{issue.block_type}] {issue.block_id}")
                print(f"    Path: {issue.path}")
                if issue.errors:
                    for error in issue.errors:
                        print(f"    ❌ {error}")
                if issue.warnings:
                    for warning in issue.warnings:
                        print(f"    ⚠️  {warning}")
                if issue.is_fixable:
                    print(f"    🔧 Auto-fixable")
    else:
        print(f"\n✅ No issues found")

    if report.fixed_count > 0:
        print(f"\n🔧 Fixed {report.fixed_count} issues")


def validate_file(
    file_path: Path,
    validator: IRValidator,
    fix: bool = False,
    verbose: bool = False,
) -> DocumentReport:
    """Validate a single file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            document = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {file_path}: {e}")
        report = DocumentReport(file_path=str(file_path))
        report.issues.append(BlockIssue(
            block_type="document",
            block_id="root",
            path="",
            errors=[f"JSON parse error: {e}"],
        ))
        return report
    except Exception as e:
        logger.error(f"File read error: {file_path}: {e}")
        report = DocumentReport(file_path=str(file_path))
        report.issues.append(BlockIssue(
            block_type="document",
            block_id="root",
            path="",
            errors=[f"File read error: {e}"],
        ))
        return report

    # Validate document
    report = validator.validate_document(document, str(file_path))

    # Repair issues
    if fix and report.has_issues:
        fixable_issues = [i for i in report.issues if i.is_fixable]
        if fixable_issues:
            logger.info(f"Attempting to fix {len(fixable_issues)} issues...")
            document, fixed_count = validator.repair_document(document, report)
            report.fixed_count = fixed_count

            if fixed_count > 0:
                # Save repaired file
                backup_path = file_path.with_suffix(f".bak{file_path.suffix}")
                try:
                    # Create backup
                    import shutil
                    shutil.copy(file_path, backup_path)
                    logger.info(f"Backup created: {backup_path}")

                    # Save repaired file
                    with open(file_path, "w", encoding="utf-8") as f:
                        json.dump(document, f, ensure_ascii=False, indent=2)
                    logger.info(f"Repaired file saved: {file_path}")
                except Exception as e:
                    logger.error(f"Failed to save file: {e}")

    return report


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="IR document validation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s chapter-030-section-3-0.json
  %(prog)s *.json --fix
  %(prog)s ./output/ --recursive --fix --verbose
        """,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="JSON files or directories to validate",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively",
    )
    parser.add_argument(
        "-f", "--fix",
        action="store_true",
        help="Auto-fix common issues",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed output",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )

    args = parser.parse_args()

    # Configure logging
    logger.remove()
    if args.verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.add(sys.stderr, level="INFO")

    # Collect files
    files: List[Path] = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            if path.suffix.lower() == ".json":
                files.append(path)
        elif path.is_dir():
            if args.recursive:
                files.extend(path.rglob("*.json"))
            else:
                files.extend(path.glob("*.json"))
        else:
            # Might be a glob pattern
            import glob
            matched = glob.glob(path_str)
            for m in matched:
                mp = Path(m)
                if mp.is_file() and mp.suffix.lower() == ".json":
                    files.append(mp)

    if not files:
        print("No JSON files found")
        sys.exit(1)

    print(f"Found {len(files)} files")

    # Create validator
    validator = IRValidator()

    # Validate files
    total_issues = 0
    total_fixed = 0
    reports: List[DocumentReport] = []

    for file_path in files:
        report = validate_file(file_path, validator, args.fix, args.verbose)
        reports.append(report)
        total_issues += len(report.issues)
        total_fixed += report.fixed_count

        if args.verbose or report.has_issues:
            print_report(report, args.verbose)

    # Print summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"  - Files: {len(files)}")
    print(f"  - Total issues: {total_issues}")
    if args.fix:
        print(f"  - Fixed: {total_fixed}")

    # Return proper exit code
    if total_issues > 0 and total_fixed < total_issues:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
