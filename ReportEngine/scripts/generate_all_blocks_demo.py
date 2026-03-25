#!/usr/bin/env python3
"""
Generate a demo IR that covers all allowed block types for HTML / PDF / Markdown rendering tests.

After execution, the script writes a timestamped IR file to `final_reports/ir`,
and outputs rendered files to `final_reports/html`, `final_reports/pdf`, and `final_reports/md`.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

# Allow direct script execution
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ReportEngine.core import DocumentComposer
from ReportEngine.ir import IRValidator
from ReportEngine.ir.schema import ENGINE_AGENT_TITLES
from ReportEngine.renderers import HTMLRenderer, MarkdownRenderer, PDFRenderer
from ReportEngine.utils.config import settings


def build_inline_marks_demo() -> dict:
    """Build a paragraph block that covers all inline marks."""
    return {
        "type": "paragraph",
        "inlines": [
            {"text": "This paragraph demonstrates all inline marks:"},
            {"text": "Bold", "marks": [{"type": "bold"}]},
            {"text": " / Italic", "marks": [{"type": "italic"}]},
            {"text": " / Underline", "marks": [{"type": "underline"}]},
            {"text": " / Strike", "marks": [{"type": "strike"}]},
            {"text": " / Code", "marks": [{"type": "code"}]},
            {
                "text": " / Link",
                "marks": [
                    {
                        "type": "link",
                        "href": "https://example.com/demo",
                        "title": "Example link",
                    }
                ],
            },
            {"text": " / Color", "marks": [{"type": "color", "value": "#c0392b"}]},
            {
                "text": " / Font",
                "marks": [
                    {
                        "type": "font",
                        "family": "Georgia, serif",
                        "size": "15px",
                        "weight": "600",
                    }
                ],
            },
            {"text": " / Highlight", "marks": [{"type": "highlight"}]},
            {"text": " / Subscript", "marks": [{"type": "subscript"}]},
            {"text": " / Superscript", "marks": [{"type": "superscript"}]},
            {"text": " / Inline math", "marks": [{"type": "math", "value": "E=mc^2"}]},
            {"text": "."},
        ],
    }


def build_widget_block() -> dict:
    """Build a valid Chart.js widget block."""
    return {
        "type": "widget",
        "widgetId": "demo-volume-trend",
        "widgetType": "chart.js/line",
        "props": {
            "type": "line",
            "options": {
                "responsive": True,
                "plugins": {"legend": {"position": "bottom"}},
                "scales": {"y": {"title": {"display": True, "text": "Mentions"}}},
            },
        },
        "data": {
            "labels": ["T0", "T0+6h", "T0+12h", "T0+18h", "T0+24h"],
            "datasets": [
                {
                    "label": "Mainstream media",
                    "data": [12, 18, 23, 30, 26],
                    "borderColor": "#2980b9",
                    "backgroundColor": "rgba(41,128,185,0.18)",
                    "tension": 0.25,
                    "fill": False,
                },
                {
                    "label": "Social platforms",
                    "data": [8, 10, 15, 28, 40],
                    "borderColor": "#c0392b",
                    "backgroundColor": "rgba(192,57,43,0.2)",
                    "tension": 0.35,
                    "fill": False,
                },
            ],
        },
    }


def build_chapters() -> list[dict]:
    """Build chapter list that covers all block types."""
    inline_demo = build_inline_marks_demo()

    bullet_list = {
        "type": "list",
        "listType": "bullet",
        "items": [
            [
                {
                    "type": "paragraph",
                    "inlines": [{"text": "Social media heat doubled within 48 hours"}],
                }
            ],
            [
                {
                    "type": "paragraph",
                    "inlines": [{"text": "Mainstream media coverage was concentrated in the morning"}],
                },
                {
                    "type": "list",
                    "listType": "ordered",
                    "items": [
                        [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "07:00-09:00: First wave of reports"}],
                            }
                        ],
                        [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "10:00-12:00: Commentary spread"}],
                            }
                        ],
                    ],
                },
            ],
            [
                {
                    "type": "paragraph",
                    "inlines": [{"text": "Local government accounts started responding and synced offline releases"}],
                }
            ],
        ],
    }

    task_list = {
        "type": "list",
        "listType": "task",
        "items": [
            [
                {
                    "type": "paragraph",
                    "inlines": [{"text": "Track whether authoritative fact-check materials are published"}],
                }
            ],
            [
                {
                    "type": "paragraph",
                    "inlines": [{"text": "Monitor emerging related keywords and long-tail questions"}],
                }
            ],
            [
                {
                    "type": "paragraph",
                    "inlines": [{"text": "Prepare FAQ for unified customer support responses"}],
                }
            ],
        ],
    }

    table_block = {
        "type": "table",
        "caption": "Core sources and dissemination paths",
        "zebra": True,
        "colgroup": [{"width": "22%"}, {"width": "38%"}, {"width": "40%"}],
        "rows": [
            {
                "cells": [
                    {
                        "align": "center",
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Timepoint", "marks": [{"type": "bold"}]}],
                            }
                        ],
                    },
                    {
                        "align": "center",
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Event", "marks": [{"type": "bold"}]}],
                            }
                        ],
                    },
                    {
                        "align": "center",
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Main channels", "marks": [{"type": "bold"}]}],
                            }
                        ],
                    },
                ]
            },
            {
                "cells": [
                    {"blocks": [{"type": "paragraph", "inlines": [{"text": "T0"}]}]},
                    {
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Offline conflict video was first uploaded"}],
                            }
                        ]
                    },
                    {
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Short-video platforms / private chat forwarding"}],
                            }
                        ]
                    },
                ]
            },
            {
                "cells": [
                    {"blocks": [{"type": "paragraph", "inlines": [{"text": "T0+6h"}]}]},
                    {
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Reached trending lists, with secondary edits appearing"}],
                            }
                        ]
                    },
                    {
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Weibo / Moments"}],
                            }
                        ]
                    },
                ]
            },
            {
                "cells": [
                    {"blocks": [{"type": "paragraph", "inlines": [{"text": "T0+18h"}]}]},
                    {
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Official response issued with factual clarification"}],
                            }
                        ]
                    },
                    {
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Government accounts / news apps"}],
                            }
                        ]
                    },
                ]
            },
            {
                "cells": [
                    {"blocks": [{"type": "paragraph", "inlines": [{"text": "T0+24h"}]}]},
                    {
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Expert commentary shifted focus toward responsibility attribution"}],
                            }
                        ]
                    },
                    {
                        "blocks": [
                            {
                                "type": "paragraph",
                                "inlines": [{"text": "Live video channels / industry communities"}],
                            }
                        ]
                    },
                ]
            },
        ],
    }

    blockquote_block = {
        "type": "blockquote",
        "variant": "accent",
        "blocks": [
            {
                "type": "paragraph",
                "inlines": [{"text": "\"The public cares most about truth and the boundaries of responsibility.\""}],
            },
            {
                "type": "paragraph",
                "inlines": [{"text": "- Simulated quote to validate blockquote styling"}],
            },
        ],
    }

    engine_quote_block = {
        "type": "engineQuote",
        "engine": "insight",
        "title": ENGINE_AGENT_TITLES["insight"],
        "blocks": [
            {
                "type": "paragraph",
                "inlines": [
                    {
                        "text": "The model suggests maintaining response cadence within 24 hours to avoid an information vacuum.",
                        "marks": [{"type": "bold"}],
                    }
                ],
            },
            {
                "type": "paragraph",
                "inlines": [
                    {"text": "It also recommends preparing a short FAQ for consistent cross-channel messaging."}
                ],
            },
        ],
    }

    swot_block = {
        "type": "swotTable",
        "title": "SWOT Snapshot of the Public Discourse",
        "summary": "Covers current sentiment distribution, potential risks, and opportunities.",
        "strengths": [
            {"title": "Fast official response", "detail": "First clarification video went live within 3 hours"},
            {"title": "Local media coordination", "impact": "High", "score": 8},
        ],
        "weaknesses": [
            {"title": "Large stock of early rumors", "detail": "Related reposts still account for 30%"},
            "External experts have not aligned on messaging yet",
        ],
        "opportunities": [
            {
                "title": "Community co-creation discussions",
                "detail": "Spontaneous \"fact-check volunteer\" topics emerged with positive sentiment",
            },
            {"title": "Public-welfare collaboration window", "impact": "Medium"},
        ],
        "threats": [
            {"title": "Cross-platform edited clips continue to spread", "impact": "High", "score": 9},
            {"title": "Some self-media accounts inflame emotions", "evidence": "Regional labeling tendencies observed"},
        ],
    }

    pest_block = {
        "type": "pestTable",
        "title": "Macro Environment Pulse Scan (PEST)",
        "summary": "Simulates external constraints and opportunities across four dimensions to validate pestTable rendering.",
        "political": [
            {
                "title": "Local regulation consultation",
                "detail": "Short-video posting requires real-name traceability, opening a platform compliance communication window",
                "trend": "Positive",
                "impact": 7,
            },
            {
                "title": "Regulatory focus on emotional manipulation",
                "detail": "Accounts exaggerating conflicts are under prioritized inspection, lowering discourse tolerance thresholds",
                "trend": "Watch",
                "impact": 6,
            },
        ],
        "economic": [
            {
                "title": "Revenue fluctuations for nearby merchants",
                "detail": "Foot traffic dropped 12% short-term, but livestream commerce orders increased",
                "trend": "Neutral",
                "impact": 5,
            },
            {
                "title": "Cautious brand sponsorship",
                "detail": "Sponsorship delays due to reputation risk observation pressure official announcement cadence",
                "trend": "Uncertain",
                "impact": 4,
            },
        ],
        "social": [
            {
                "title": "Sentiment divergence among core groups",
                "detail": "Local residents focus on safety, while non-local visitors focus on experience and refunds",
                "trend": "Negative",
                "impact": 8,
            },
            {
                "title": "University communities self-verify information",
                "detail": "Campus media and student unions organized image-verification educational posts, stabilizing sentiment",
                "trend": "Positive",
                "impact": 6,
            },
        ],
        "technological": [
            {
                "title": "AI-generated content mixed in",
                "detail": "Partial frames are enlarged and re-shared, requiring watermark traceability tools for verification",
                "trend": "Negative",
                "impact": 7,
            },
            {
                "title": "Multimodal retrieval launched",
                "detail": "Platforms pilot a video anti-fraud model that auto-flags editing traces",
                "trend": "Positive",
                "impact": 5,
            },
        ],
    }

    callout_block = {
        "type": "callout",
        "tone": "warning",
        "title": "Layout boundary note",
        "blocks": [
            {
                "type": "paragraph",
                "inlines": [
                    {"text": "Keep only lightweight content inside callouts; overflow will automatically move to outer layout."}
                ],
            },
            {
                "type": "list",
                "listType": "bullet",
                "items": [
                    [
                        {
                            "type": "paragraph",
                            "inlines": [{"text": "Supports nested lists / tables / math formulas"}],
                        }
                    ],
                    [
                        {
                            "type": "paragraph",
                            "inlines": [{"text": "Use this area for reminders or operational steps"}],
                        }
                    ],
                ],
            },
        ],
    }

    code_block = {
        "type": "code",
        "lang": "json",
        "caption": "Demo code block",
        "content": '{\n  "event": "Hot-topic example",\n  "topic": "Public event",\n  "status": "monitoring"\n}',
    }

    math_block = {
        "type": "math",
        "latex": r"E = mc^2",
        "displayMode": True,
    }

    figure_block = {
        "type": "figure",
        "img": {
            "src": "https://dummyimage.com/600x320/eeeeee/333333&text=Placeholder",
            "alt": "Placeholder illustration",
            "width": 600,
            "height": 320,
        },
        "caption": "External image links are replaced with a friendly notice to verify figure placeholder behavior.",
        "responsive": True,
    }

    widget_block = build_widget_block()
    stacked_bar_chart_block = {
        "type": "widget",
        "widgetId": "demo-stacked-sentiment",
        "widgetType": "chart.js/bar",
        "props": {
            "type": "bar",
            "options": {
                "responsive": True,
                "plugins": {"legend": {"position": "bottom"}},
                "scales": {
                    "x": {"stacked": True},
                    "y": {"stacked": True, "title": {"display": True, "text": "Information volume"}},
                },
            },
        },
        "data": {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri"],
            "datasets": [
                {"label": "Positive", "data": [18, 22, 24, 19, 16], "backgroundColor": "#27ae60"},
                {"label": "Neutral", "data": [22, 20, 18, 21, 23], "backgroundColor": "#f39c12"},
                {"label": "Negative", "data": [12, 14, 10, 9, 11], "backgroundColor": "#c0392b"},
            ],
        },
    }
    horizontal_bar_chart_block = {
        "type": "widget",
        "widgetId": "demo-horizontal-voice",
        "widgetType": "chart.js/bar",
        "props": {
            # Switch to a horizontal bar chart via indexAxis
            "type": "bar",
            "options": {
                "indexAxis": "y",
                "plugins": {"legend": {"position": "right"}},
                "scales": {"x": {"title": {"display": True, "text": "Mentions (10k)"}}},
            },
        },
        "data": {
            "labels": ["Weibo", "Short video", "Community forums", "News apps"],
            "datasets": [
                {
                    "label": "Volume comparison",
                    "data": [42, 58, 27, 36],
                    "backgroundColor": ["#2ecc71", "#3498db", "#9b59b6", "#f39c12"],
                }
            ],
        },
    }
    pie_chart_block = {
        "type": "widget",
        "widgetId": "demo-stance-pie",
        "widgetType": "chart.js/pie",
        "props": {
            "type": "pie",
            "options": {"plugins": {"legend": {"position": "bottom"}}},
        },
        "data": {
            "labels": ["Support", "Neutral", "Skeptical"],
            "datasets": [
                {
                    "label": "Stance distribution",
                    "data": [36, 28, 21],
                    "backgroundColor": ["#27ae60", "#f1c40f", "#c0392b"],
                }
            ],
        },
    }
    doughnut_chart_block = {
        "type": "widget",
        "widgetId": "demo-sentiment-share",
        "widgetType": "chart.js/doughnut",
        "props": {
            "type": "doughnut",
            "options": {"plugins": {"legend": {"position": "right"}, "tooltip": {"enabled": True}}},
        },
        "data": {
            "labels": ["Policy", "Economy", "Society", "Technology"],
            "datasets": [
                {
                    "label": "Attention share",
                    "data": [24, 30, 28, 18],
                    "backgroundColor": ["#8e44ad", "#16a085", "#e67e22", "#2980b9"],
                    "hoverOffset": 6,
                }
            ],
        },
    }
    radar_chart_block = {
        "type": "widget",
        "widgetId": "demo-response-radar",
        "widgetType": "chart.js/radar",
        "props": {
            "type": "radar",
            "options": {
                "plugins": {"legend": {"position": "top"}},
                "scales": {"r": {"beginAtZero": True, "max": 100}},
            },
        },
        "data": {
            "labels": ["Transparency", "Response speed", "Consistency", "Engagement", "Information volume"],
            "datasets": [
                {
                    "label": "Official channels",
                    "data": [78, 88, 82, 66, 91],
                    "backgroundColor": "rgba(46,204,113,0.15)",
                    "borderColor": "#2ecc71",
                    "pointBackgroundColor": "#27ae60",
                },
                {
                    "label": "Public discussion",
                    "data": [64, 72, 58, 74, 63],
                    "backgroundColor": "rgba(52,152,219,0.12)",
                    "borderColor": "#3498db",
                    "pointBackgroundColor": "#2980b9",
                },
            ],
        },
    }
    polar_area_chart_block = {
        "type": "widget",
        "widgetId": "demo-channel-polar",
        "widgetType": "chart.js/polarArea",
        "props": {"type": "polarArea"},
        "data": {
            "labels": ["Short video", "Weibo", "Community forums", "News apps", "Offline feedback"],
            "datasets": [
                {
                    "label": "Channel penetration",
                    "data": [62, 54, 38, 45, 28],
                    "backgroundColor": [
                        "rgba(231,76,60,0.65)",
                        "rgba(142,68,173,0.6)",
                        "rgba(52,152,219,0.55)",
                        "rgba(46,204,113,0.55)",
                        "rgba(241,196,15,0.6)",
                    ],
                }
            ],
        },
    }
    scatter_chart_block = {
        "type": "widget",
        "widgetId": "demo-correlation-scatter",
        "widgetType": "chart.js/scatter",
        "props": {
            "type": "scatter",
            "options": {
                "plugins": {"legend": {"position": "bottom"}},
                "scales": {
                    "x": {"title": {"display": True, "text": "Sentiment polarity"}, "min": -1, "max": 1},
                    "y": {"title": {"display": True, "text": "Engagement"}, "beginAtZero": True},
                },
            },
        },
        "data": {
            "datasets": [
                {
                    "label": "Post scatter",
                    "data": [
                        {"x": -0.65, "y": 120},
                        {"x": -0.25, "y": 190},
                        {"x": 0.05, "y": 260},
                        {"x": 0.42, "y": 340},
                        {"x": 0.78, "y": 410},
                    ],
                    "backgroundColor": "rgba(52,152,219,0.7)",
                }
            ],
        },
    }
    bubble_chart_block = {
        "type": "widget",
        "widgetId": "demo-impact-bubble",
        "widgetType": "chart.js/bubble",
        "props": {
            "type": "bubble",
            "options": {
                "plugins": {"legend": {"position": "bottom"}},
                "scales": {
                    "x": {"title": {"display": True, "text": "Exposure (10k)"}, "beginAtZero": True},
                    "y": {"title": {"display": True, "text": "Sentiment intensity"}, "min": -100, "max": 100},
                },
            },
        },
        "data": {
            "datasets": [
                {
                    "label": "Channel distribution",
                    "data": [
                        {"x": 8, "y": 35, "r": 12},
                        {"x": 12, "y": -28, "r": 10},
                        {"x": 18, "y": 22, "r": 14},
                        {"x": 25, "y": 48, "r": 16},
                        {"x": 6, "y": -12, "r": 8},
                    ],
                    "backgroundColor": "rgba(192,57,43,0.55)",
                    "borderColor": "#c0392b",
                }
            ],
        },
    }

    chapter_1 = {
        "chapterId": "S1",
        "title": "Cover and table of contents",
        "anchor": "overview",
        "order": 10,
        "blocks": [
            {"type": "heading", "level": 2, "text": "I. Cover and table of contents", "anchor": "overview"},
            {
                "type": "paragraph",
                "inlines": [
                    {
                        "text": "A simulated summary of a social public hotspot event for quickly checking layout and typography effects.",
                    }
                ],
            },
            inline_demo,
            {
                "type": "kpiGrid",
                "items": [
                    {"label": "24h mentions", "value": "98K", "delta": "+41%", "deltaTone": "up"},
                    {"label": "Positive share", "value": "32%", "delta": "+5pp", "deltaTone": "up"},
                    {"label": "Negative share", "value": "18%", "delta": "-3pp", "deltaTone": "down"},
                    {"label": "Top channels", "value": "Short video / Weibo"},
                ],
                "cols": 4,
            },
            {"type": "toc"},
            {"type": "hr"},
        ],
    }

    chapter_2 = {
        "chapterId": "S2",
        "title": "Block type showcase",
        "anchor": "blocks-showcase",
        "order": 20,
        "blocks": [
            {
                "type": "heading",
                "level": 2,
                "text": "II. Block type showcase",
                "anchor": "blocks-showcase",
            },
            {
                "type": "paragraph",
                "inlines": [
                    {
                        "text": "The following content covers all block types one by one, including paragraph/list/table/swot/pest/widget.",
                    }
                ],
            },
            {
                "type": "heading",
                "level": 3,
                "text": "2.1 Lists and tables",
                "anchor": "lists-and-tables",
            },
            bullet_list,
            task_list,
            table_block,
            {
                "type": "heading",
                "level": 3,
                "text": "2.2 Chart component showcase",
                "anchor": "charts-demo",
            },
            {
                "type": "paragraph",
                "inlines": [
                    {
                        "text": "Line / bar (including horizontal and stacked) / pie / doughnut / radar / polar / scatter / bubble charts validate Chart.js compatibility.",
                    }
                ],
            },
            widget_block,
            stacked_bar_chart_block,
            horizontal_bar_chart_block,
            pie_chart_block,
            doughnut_chart_block,
            radar_chart_block,
            polar_area_chart_block,
            scatter_chart_block,
            bubble_chart_block,
            {
                "type": "heading",
                "level": 3,
                "text": "2.3 Advanced blocks and rich media",
                "anchor": "advanced-blocks",
            },
            blockquote_block,
            callout_block,
            engine_quote_block,
            swot_block,
            pest_block,
            code_block,
            math_block,
            figure_block,
            {
                "type": "hr",
                "variant": "dashed",
            },
            {
                "type": "paragraph",
                "align": "justify",
                "inlines": [
                    {
                        "text": "Inline math fallback validation for this chapter:",
                    },
                    {"text": "p(t)=p_0 e^{\\lambda t}", "marks": [{"type": "math"}]},
                    {"text": "; all allowed blocks and marks are covered above."},
                ],
            },
        ],
    }

    return [chapter_1, chapter_2]


def validate_chapters(chapters: list[dict]) -> None:
    """Validate chapter structure with IRValidator and raise on error."""
    validator = IRValidator()
    for chapter in chapters:
        ok, errors = validator.validate_chapter(chapter)
        if not ok:
            raise ValueError(f"{chapter.get('chapterId', 'unknown')} validation failed: {errors}")


def render_and_save(document_ir: dict, timestamp: str) -> tuple[Path, Path, Path, Path]:
    """Save IR as JSON, render HTML / PDF / Markdown, and return four output paths."""
    ir_dir = Path(settings.DOCUMENT_IR_OUTPUT_DIR)
    html_dir = Path(settings.OUTPUT_DIR) / "html"
    pdf_dir = Path(settings.OUTPUT_DIR) / "pdf"
    md_dir = Path(settings.OUTPUT_DIR) / "md"
    ir_dir.mkdir(parents=True, exist_ok=True)
    html_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    md_dir.mkdir(parents=True, exist_ok=True)

    ir_path = ir_dir / f"report_ir_all_blocks_demo_{timestamp}.json"
    ir_path.write_text(json.dumps(document_ir, ensure_ascii=False, indent=2), encoding="utf-8")

    html_renderer = HTMLRenderer()
    html_content = html_renderer.render(document_ir)
    html_path = html_dir / f"report_html_all_blocks_demo_{timestamp}.html"
    html_path.write_text(html_content, encoding="utf-8")

    pdf_renderer = PDFRenderer()
    pdf_path = pdf_dir / f"report_pdf_all_blocks_demo_{timestamp}.pdf"
    pdf_renderer.render_to_pdf(document_ir, pdf_path)

    md_renderer = MarkdownRenderer()
    md_content = md_renderer.render(document_ir, ir_file_path=str(ir_path))
    md_path = md_dir / f"report_md_all_blocks_demo_{timestamp}.md"
    md_path.write_text(md_content, encoding="utf-8")

    return ir_path, html_path, pdf_path, md_path


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_id = f"all-blocks-demo-{timestamp}"
    metadata = {
        "title": "Social Public Hotspot Rendering Test",
        "subtitle": "Sample data covering all IR block types, including multiple charts and PEST demonstration",
        "query": "Public-event rendering self-check / Chart & PEST",
        "toc": {"title": "Contents", "depth": 3},
        "hero": {
            "summary": "Used to validate compatibility of Report Engine with various blocks, Chart.js components, and PEST modules in HTML / PDF rendering.",
            "kpis": [
                {"label": "Demo block count", "value": "20+", "delta": "Includes PEST", "tone": "up"},
                {"label": "Chart count", "value": "7", "delta": "More chart types", "tone": "neutral"},
            ],
            "highlights": ["All blocks covered", "Inline/block math included", "Multiple Chart.js types", "PEST + SWOT"],
            "actions": ["Regenerate", "Export PDF"],
        },
    }

    chapters = build_chapters()
    validate_chapters(chapters)

    composer = DocumentComposer()
    document_ir = composer.build_document(report_id, metadata, chapters)

    ir_path, html_path, pdf_path, md_path = render_and_save(document_ir, timestamp)

    print("✅ Demo IR generation completed")
    print(f"IR:   {ir_path}")
    print(f"HTML: {html_path}")
    print(f"PDF:  {pdf_path}")
    print(f"MD:   {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
