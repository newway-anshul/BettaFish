#!/usr/bin/env python
"""
PDF export script
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project path to sys.path
sys.path.insert(0, '/Users/mayiding/Desktop/GitMy/BettaFish')

def export_pdf(ir_file_path):
    """Export PDF."""
    try:
        # Read IR file
        print(f"Reading report file: {ir_file_path}")
        with open(ir_file_path, 'r', encoding='utf-8') as f:
            document_ir = json.load(f)

        # Import PDF renderer
        from ReportEngine.renderers.pdf_renderer import PDFRenderer

        # Create PDF renderer
        print("Initializing PDF renderer...")
        renderer = PDFRenderer()

        # Generate PDF
        print("Generating PDF...")
        pdf_bytes = renderer.render_to_bytes(document_ir, optimize_layout=True)

        # Determine output file name
        topic = document_ir.get('metadata', {}).get('topic', 'report')
        output_dir = Path('/Users/mayiding/Desktop/GitMy/BettaFish/final_reports/pdf')
        output_dir.mkdir(parents=True, exist_ok=True)

        pdf_filename = f"report_{topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        output_path = output_dir / pdf_filename

        # Save PDF file
        print(f"Saving PDF to: {output_path}")
        with open(output_path, 'wb') as f:
            f.write(pdf_bytes)

        print("✅ PDF exported successfully!")
        print(f"File location: {output_path}")
        print(f"File size: {len(pdf_bytes) / 1024 / 1024:.2f} MB")

        return str(output_path)

    except Exception as e:
        print(f"❌ PDF export failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Use the latest report file
    latest_report = "/Users/mayiding/Desktop/GitMy/BettaFish/final_reports/ir/report_ir_AI_market_development_trend_20251119_235407.json"

    if os.path.exists(latest_report):
        print("="*50)
        print("Starting PDF export")
        print("="*50)
        result = export_pdf(latest_report)
        if result:
            print(f"\n📄 PDF file generated: {result}")
    else:
        print(f"❌ Report file does not exist: {latest_report}")