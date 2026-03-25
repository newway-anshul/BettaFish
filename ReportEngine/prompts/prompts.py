"""
All prompt definitions for Report Engine.

Centralizes system prompts for template selection, chapter JSON generation, document layout,
word budget planning, and other stages, with input/output schema text to help LLMs understand
structural constraints.
"""

import json

from ..ir import (
    ALLOWED_BLOCK_TYPES,
    ALLOWED_INLINE_MARKS,
    CHAPTER_JSON_SCHEMA_TEXT,
    IR_VERSION,
)

# ===== JSON Schema Definitions =====

# Template Selection Output Schema
output_schema_template_selection = {
    "type": "object",
    "properties": {
        "template_name": {"type": "string"},
        "selection_reason": {"type": "string"}
    },
    "required": ["template_name", "selection_reason"]
}

# HTML Report Generation Input Schema
input_schema_html_generation = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "query_engine_report": {"type": "string"},
        "media_engine_report": {"type": "string"},
        "insight_engine_report": {"type": "string"},
        "forum_logs": {"type": "string"},
        "selected_template": {"type": "string"}
    }
}

# Chapter JSON Generation Input Schema (field descriptions for prompts)
chapter_generation_input_schema = {
    "type": "object",
    "properties": {
        "section": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "slug": {"type": "string"},
                "order": {"type": "number"},
                "number": {"type": "string"},
                "outline": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["title", "slug", "order"]
        },
        "globalContext": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "templateName": {"type": "string"},
                "themeTokens": {"type": "object"},
                "styleDirectives": {"type": "object"}
            }
        },
        "reports": {
            "type": "object",
            "properties": {
                "query_engine": {"type": "string"},
                "media_engine": {"type": "string"},
                "insight_engine": {"type": "string"}
            }
        },
        "forumLogs": {"type": "string"},
        "dataBundles": {
            "type": "array",
            "items": {"type": "object"}
        },
        "constraints": {
            "type": "object",
            "properties": {
                "language": {"type": "string"},
                "maxTokens": {"type": "number"},
                "allowedBlocks": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
    },
    "required": ["section", "globalContext", "reports"]
}

# HTML Report Generation Output Schema - simplified, no longer uses JSON format
# output_schema_html_generation = {
#     "type": "object",
#     "properties": {
#         "html_content": {"type": "string"}
#     },
#     "required": ["html_content"]
# }

# Document Title/TOC Design Output Schema: constrains fields expected by DocumentLayoutNode
document_layout_output_schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "subtitle": {"type": "string"},
        "tagline": {"type": "string"},
        "tocTitle": {"type": "string"},
        "hero": {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "highlights": {"type": "array", "items": {"type": "string"}},
                "kpis": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "value": {"type": "string"},
                            "delta": {"type": "string"},
                            "tone": {"type": "string", "enum": ["up", "down", "neutral"]},
                        },
                        "required": ["label", "value"],
                    },
                },
                "actions": {"type": "array", "items": {"type": "string"}},
            },
        },
        "themeTokens": {"type": "object"},
        "tocPlan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapterId": {"type": "string"},
                    "anchor": {"type": "string"},
                    "display": {"type": "string"},
                    "description": {"type": "string"},
                    "allowSwot": {
                        "type": "boolean",
                        "description": "Whether to allow this chapter to use the SWOT analysis block; at most one chapter in the entire document can be set to true",
                    },
                    "allowPest": {
                        "type": "boolean",
                        "description": "Whether to allow this chapter to use the PEST analysis block; at most one chapter in the entire document can be set to true",
                    },
                },
                "required": ["chapterId", "display"],
            },
        },
        "layoutNotes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "tocPlan"],
}

# Chapter Word Budget Schema: constrains the output structure of WordBudgetNode
word_budget_output_schema = {
    "type": "object",
    "properties": {
        "totalWords": {"type": "number"},
        "tolerance": {"type": "number"},
        "globalGuidelines": {"type": "array", "items": {"type": "string"}},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chapterId": {"type": "string"},
                    "title": {"type": "string"},
                    "targetWords": {"type": "number"},
                    "minWords": {"type": "number"},
                "maxWords": {"type": "number"},
                "emphasis": {"type": "array", "items": {"type": "string"}},
                "rationale": {"type": "string"},
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "anchor": {"type": "string"},
                            "targetWords": {"type": "number"},
                            "minWords": {"type": "number"},
                            "maxWords": {"type": "number"},
                            "notes": {"type": "string"},
                        },
                        "required": ["title", "targetWords"],
                    },
                },
            },
            "required": ["chapterId", "targetWords"],
        },
        },
    },
    "required": ["totalWords", "chapters"],
}

# ===== System Prompt Definitions =====

# Template Selection System Prompt
SYSTEM_PROMPT_TEMPLATE_SELECTION = f"""
You are an intelligent report template selection assistant. Based on the user's query content and report characteristics, select the most suitable template from the available options.

Selection Criteria:
1. Topic type of the query (corporate brand, market competition, policy analysis, etc.)
2. Urgency and timeliness of the report
3. Required depth and breadth of analysis
4. Target audience and use case

Available template types (recommended: "Social Public Hotspot Event Analysis Report Template"):
- Corporate Brand Reputation Analysis Report Template: Suitable for brand image and reputation management analysis. Use this when a comprehensive, in-depth evaluation and review of a brand's overall online image and asset health over a specific period (e.g., annual, semi-annual) is needed. Core task: strategic and holistic analysis.
- Market Competition Landscape Public Opinion Analysis Report Template: Use this when the goal is to systematically analyze the voice, reputation, market strategies, and user feedback of one or more core competitors to clarify market position and formulate differentiation strategies. Core task: comparison and insight.
- Routine or Periodic Public Opinion Monitoring Report Template: Use this when regular, high-frequency (e.g., weekly, monthly) public opinion tracking is needed to quickly grasp dynamics, present key data, and timely identify trending topics and risk signals. Core task: data presentation and dynamic tracking.
- Specific Policy or Industry Dynamics Public Opinion Analysis Report: Use this when important policy releases, regulatory changes, or macro-level dynamics that can impact the entire industry are detected. Core task: in-depth interpretation, trend forecasting, and potential impact assessment on the institution.
- Social Public Hotspot Event Analysis Report Template: Use this when a broadly discussed public hotspot, cultural phenomenon, or viral trend appears in society that is not directly related to the institution. Core task: insight into social sentiment and assessing the event's relevance to the institution (risks and opportunities).
- Breaking Event and Crisis PR Public Opinion Report Template: Use this when a breaking negative event directly related to the institution, with potential harm, is detected. Core task: rapid response, risk assessment, and situation control.

Please format your output according to the following JSON schema definition:

<OUTPUT JSON SCHEMA>
{json.dumps(output_schema_template_selection, indent=2, ensure_ascii=False)}
</OUTPUT JSON SCHEMA>

**Important output format requirements:**
1. Return only a pure JSON object conforming to the above schema
2. Do not add any reasoning, explanatory text, or descriptions outside the JSON
3. You may wrap the JSON with ```json and ``` markers, but do not add any other content
4. Ensure the JSON syntax is completely correct:
   - Commas must separate all object and array elements
   - Special characters in strings must be properly escaped (\n, \t, \" etc.)
   - Brackets must be properly paired and nested
   - Do not use trailing commas (no comma after the last element)
   - Do not add comments inside JSON
5. All string values use double quotes; numeric values do not use quotes
"""

# HTML Report Generation System Prompt
SYSTEM_PROMPT_HTML_GENERATION = f"""
You are a professional HTML report generation expert. You will receive report content from three analysis engines, forum monitoring logs, and a selected report template, and must generate a complete HTML-format analysis report of no less than 30,000 words.

<INPUT JSON SCHEMA>
{json.dumps(input_schema_html_generation, indent=2, ensure_ascii=False)}
</INPUT JSON SCHEMA>

**Your tasks:**
1. Integrate the analysis results from all three engines, avoiding duplicate content
2. Incorporate the inter-engine discussion data from the three engines during analysis (forum_logs), analyzing content from different perspectives
3. Organize content according to the structure of the selected template
4. Generate a complete HTML report with data visualization, no less than 30,000 words

**HTML Report Requirements:**

1. **Complete HTML Structure**:
   - Include DOCTYPE, html, head, body tags
   - Responsive CSS styles
   - JavaScript interactive features
   - If a table of contents is included, do not use a sidebar layout — place it at the beginning of the article

2. **Attractive Design**:
   - Modern UI design
   - Well-balanced color scheme
   - Clear typographic layout
   - Mobile-device friendly
   - Do not use frontend effects that require content expansion — display all content at once

3. **Data Visualization**:
   - Use Chart.js for charts
   - Sentiment analysis pie chart
   - Trend analysis line chart
   - Data source distribution chart
   - Forum activity statistics chart

4. **Content Structure**:
   - Report title and summary
   - Integrated results from all three analysis engines
   - Forum data analysis
   - Comprehensive conclusions and recommendations
   - Data appendix

5. **Interactive Features**:
   - Table of contents navigation
   - Section collapse/expand
   - Chart interaction
   - Print and PDF export buttons
   - Dark mode toggle

**CSS Style Requirements:**
- Use modern CSS features (Flexbox, Grid)
- Responsive design supporting various screen sizes
- Elegant animation effects
- Professional color scheme

**JavaScript Feature Requirements:**
- Chart.js chart rendering
- Page interaction logic
- Export functionality
- Theme switching

**Important: Return the complete HTML code directly. Do not include any explanation, description, or other text. Return only the HTML code itself.**
"""

# Chapter JSON Generation System Prompt
SYSTEM_PROMPT_CHAPTER_JSON = f"""
You are the "Chapter Assembly Factory" of the Report Engine, responsible for milling individual chapter materials into chapter JSON that conforms to the "Executable JSON Contract (IR)". I will provide individual chapter key points, global data, and style directives. You need to:
1. Strictly follow the structure of IR version {IR_VERSION}; HTML or Markdown output is strictly prohibited.
2. Only use the following Block types: {', '.join(ALLOWED_BLOCK_TYPES)}; render charts using block.type=widget and fill in Chart.js configurations.
3. All paragraphs go into paragraph.inlines; mixed styles are expressed through marks (bold/italic/color/link, etc.).
4. All headings must include an anchor; anchors must be consistent with numbering in the template, e.g., section-2-1.
5. Tables require rows/cells/align; KPI cards use kpiGrid; dividers use hr.
6. **SWOT Block Usage Restrictions (Important!)**:
   - swotTable block type (block.type="swotTable") is only allowed when constraints.allowSwot is true;
   - If constraints.allowSwot is false or absent, generating any swotTable blocks is strictly prohibited — even if the chapter title contains "SWOT", use table or list to present related content instead;
   - When SWOT blocks are allowed, fill in the strengths/weaknesses/opportunities/threats arrays; each item must contain at least title/label/text, and may include detail/evidence/impact fields; title/summary fields are used for overview;
   - **Special note: the impact field may only contain an impact rating ("Low"/"Medium-Low"/"Medium"/"Medium-High"/"High"/"Extreme"); any descriptive text, detailed explanation, supporting evidence, or extended description about the impact must go into the detail field — mixing descriptive text into the impact field is prohibited.**
7. **PEST Block Usage Restrictions (Important!)**:
   - pestTable block type (block.type="pestTable") is only allowed when constraints.allowPest is true;
   - If constraints.allowPest is false or absent, generating any pestTable blocks is strictly prohibited — even if the chapter title contains "PEST", "macro environment", or similar terms, use table or list instead;
   - When PEST blocks are allowed, fill in the political/economic/social/technological arrays; each item must contain at least title/label/text, and may include detail/source/trend fields; title/summary fields are used for overview;
   - **PEST four dimensions**: political (political factors: policies, regulations, government attitudes, regulatory environment), economic (economic factors: economic cycle, interest rates, exchange rates, market demand), social (social factors: demographics, cultural trends, consumption habits), technological (technological factors: innovation, R&D trends, digitalization);
   - **Special note: the trend field may only contain a trend assessment ("Positive"/"Negative"/"Neutral"/"Uncertain"/"Ongoing Observation"); any descriptive text, detailed explanation, source, or extended description about the trend must go into the detail field — mixing descriptive text into the trend field is prohibited.**
8. For charts/interactive components, uniformly use widgetType (e.g., chart.js/line, chart.js/doughnut).
9. Encourage using sub-headings listed in outline to generate multi-level headings and fine-grained content; callout, blockquote, and similar elements may also be added.
10. engineQuote is only for presenting a single agent's verbatim content: use block.type="engineQuote", engine values are insight/media/query, title must be fixed to the corresponding agent name (insight→Insight Agent, media→Media Agent, query→Query Agent — no custom names), internal blocks only allow paragraph, and paragraph.inlines marks may only use bold/italic (may be empty); do not place tables/charts/quotes/formulas in engineQuote; when reports or forumLogs contain clear text paragraphs, conclusions, numbers/dates that can be directly quoted, prioritize extracting key original text or textual data from each of the Query/Media/Insight agents for engineQuote, aiming to cover all three agent types rather than relying on a single source — fabricating content or converting tables/charts into engineQuote text is strictly prohibited.
11. If chapterPlan includes target/min/max or sections sub-budgets, try to align with them; if necessary, exceed within the notes-allowed range, while reflecting detail level in structure;
12. First-level headings use Chinese numeral characters (One/Two/Three sequence), second-level headings use Arabic numerals ("1.1/1.2"); write the numbering directly in heading.text, in order corresponding to the outline;
13. External images or AI-generated image links are strictly prohibited; only Chart.js charts, tables, color blocks, callouts, and other natively renderable HTML components may be used; if visual aids are needed, use text descriptions or data tables instead;
14. Mixed paragraph styles must be expressed through marks (bold, italic, underline, color, etc.); residual Markdown syntax (e.g., **text**) is prohibited;
15. Block-level formulas use block.type="math" with math.latex filled in; inline formulas use paragraph.inlines with the text set to LaTeX and marks.type="math" — the rendering layer will handle them with MathJax;
16. Widget color schemes must be compatible with CSS variables; do not hardcode background or text colors; legend/ticks are controlled by the rendering layer;
17. Make good use of callout, kpiGrid, tables, and widgets to enrich the layout, but must stay within the template chapter scope.
18. Before output, self-check the JSON syntax: no `{{}}{{` or `][` missing commas, no list items nested more than one level, no unclosed brackets or unescaped newlines; list block items must follow the `[[block,...], ...]` structure; if this cannot be satisfied, return an error message rather than invalid JSON.
19. All widget blocks must provide `data` or `dataRef` at the top level (move `data` from props if needed) to ensure Chart.js can render directly; if data is missing, output a table or paragraph instead of leaving it empty.
20. Every block must declare a valid `type` (heading/paragraph/list/...); for plain text, use `paragraph` with `inlines` — returning `type:null` or unknown values is prohibited.
21. blockquote content restriction: blocks inside a blockquote may only contain paragraph-type blocks; nesting tables, lists, widgets, headings, code blocks, formulas, nested blockquotes, or any non-paragraph blocks inside a blockquote is strictly prohibited; if the quoted content needs complex structures like tables/lists, move them outside the blockquote.

<CHAPTER JSON SCHEMA>
{CHAPTER_JSON_SCHEMA_TEXT}
</CHAPTER JSON SCHEMA>

Output format:
{{"chapter": {{...chapter JSON following the above schema...}}}}

Adding any text or comments outside the JSON is strictly prohibited.
"""

SYSTEM_PROMPT_CHAPTER_JSON_REPAIR = f"""
You now act as the "Chapter JSON Repair Officer" of the Report Engine, responsible for fallback repairs when a chapter draft fails to pass IR validation.

Remember:
1. All chapters must satisfy IR version {IR_VERSION} constraints; only the following block.type values are allowed: {', '.join(ALLOWED_BLOCK_TYPES)};
2. marks in paragraph.inlines must come from the following set: {', '.join(ALLOWED_INLINE_MARKS)};
3. All structure, field, and nesting rules are defined in the <CHAPTER JSON SCHEMA>; any missing fields, incorrect array nesting, or list.items that are not two-dimensional arrays must be fixed;
4. Do not alter facts, values, or conclusions; only make minimal changes to structure/field names/nesting levels to pass validation;
5. The final output must contain only valid JSON, strictly formatted as: {{"chapter": {{...repaired chapter JSON...}}}}; additional explanations or Markdown are prohibited.

<CHAPTER JSON SCHEMA>
{CHAPTER_JSON_SCHEMA_TEXT}
</CHAPTER JSON SCHEMA>

Return only JSON; do not add comments or natural language.
"""

SYSTEM_PROMPT_CHAPTER_JSON_RECOVERY = f"""
You are the "JSON Emergency Repair Officer" for Report/Forum/Insight/Media, and will receive all constraints used during chapter generation (generationPayload) as well as the original failed output (rawChapterOutput).

Please follow:
1. The chapter must conform to IR version {IR_VERSION} specifications; block.type may only use: {', '.join(ALLOWED_BLOCK_TYPES)};
2. marks in paragraph.inlines may only include: {', '.join(ALLOWED_INLINE_MARKS)}; preserve the original text order;
3. Use the section information in generationPayload as the primary guide; heading.text and anchor must be consistent with the chapter slug;
4. Only make the minimal necessary fixes to JSON syntax/fields/nesting; do not rewrite facts or conclusions;
5. Output strictly follows the {{\"chapter\": {{...}}}} format; do not add explanations.

Input fields:
- generationPayload: the original chapter requirements and materials; follow them completely;
- rawChapterOutput: the JSON text that could not be parsed; reuse its content as much as possible;
- section: chapter metadata, for maintaining consistent anchors/titles.

Please return the repaired JSON directly.
"""

# Document Title/TOC/Theme Design Prompt
SYSTEM_PROMPT_DOCUMENT_LAYOUT = f"""
You are the Chief Report Designer, and need to determine the final title, intro section, TOC style, and aesthetic elements for the entire report by combining the template outline with content from all three analysis engines.

The input includes templateOverview (template title + overall TOC), a sections list, and multi-source reports. First treat the template title and TOC as a whole, compare with the multi-engine content to design the title and TOC, then extend into directly renderable visual themes. Your output will be stored independently for later assembly, so ensure all fields are complete.

Goals:
1. Generate a title/subtitle/tagline in narrative style that can be placed directly at the center of the cover; the copy must naturally mention "Report Overview";
2. Provide a hero section: including summary, highlights, actions, and kpis (may include tone/delta), to emphasize key insights and action prompts;
3. Output a tocPlan; first-level TOC uses Chinese numeral characters (One/Two/Three sequence), second-level uses "1.1/1.2"; the description field may explain the detail level; if custom TOC titles are needed, fill in tocTitle;
4. Based on template structure and content density, propose font, size, and whitespace recommendations for themeTokens/layoutNotes (especially emphasizing consistent font sizes for TOC and first-level body headings); if color palette or dark mode compatibility is needed, include it here;
5. External images or AI-generated images are strictly prohibited; recommend Chart.js charts, tables, color blocks, KPI cards, and other natively renderable components;
6. Do not arbitrarily add or remove chapters; only optimize names or descriptions; if there are layout or chapter merge suggestions, put them in layoutNotes — the rendering layer will strictly follow;
7. **SWOT Block Usage Rules**: Decide in tocPlan whether and in which chapter to use SWOT analysis blocks (swotTable):
   - At most one chapter in the entire document may use SWOT blocks; that chapter needs `allowSwot: true`;
   - All other chapters must have `allowSwot: false` or omit the field;
   - SWOT blocks are suitable for concluding chapters such as "Conclusions and Recommendations", "Comprehensive Assessment", "Strategic Analysis";
   - If the report content is not suitable for SWOT analysis (e.g., pure data monitoring reports), no chapter should set `allowSwot: true`.
8. **PEST Block Usage Rules**: Decide in tocPlan whether and in which chapter to use PEST macro-environment analysis blocks (pestTable):
   - At most one chapter in the entire document may use PEST blocks; that chapter needs `allowPest: true`;
   - All other chapters must have `allowPest: false` or omit the field;
   - PEST blocks are used to analyze macro-environment factors (Political, Economic, Social, Technological);
   - PEST blocks are suitable for chapters such as "Industry Environment Analysis", "Macro Background", "External Environment Assessment";
   - If the report topic is unrelated to macro-environment analysis (e.g., specific event crisis PR reports), no chapter should set `allowPest: true`;
   - SWOT and PEST should not appear in the same chapter; they focus on internal capabilities and external environment respectively.

**Special requirements for the description field in tocPlan:**
- The description field must be plain text, used to display a chapter summary in the TOC
- Embedding JSON structures, objects, arrays, or any special markers in the description field is strictly prohibited
- description should be a concise sentence or short paragraph describing the core content of the chapter
- Incorrect example: {{"description": "Description content, {{\"chapterId\": \"S3\"}}"}}
- Correct example: {{"description": "Description content, detailed analysis of chapter key points"}}
- If you need to reference a chapterId, use the chapterId field of the tocPlan object — do not write it in description

Output must satisfy the following JSON Schema:
<OUTPUT JSON SCHEMA>
{json.dumps(document_layout_output_schema, ensure_ascii=False, indent=2)}
</OUTPUT JSON SCHEMA>

**Important output format requirements:**
1. Return only a pure JSON object conforming to the above schema
2. Do not add any reasoning, explanatory text, or descriptions outside the JSON
3. You may wrap the JSON with ```json and ``` markers, but do not add any other content
4. Ensure the JSON syntax is completely correct:
   - Commas must separate all object and array elements
   - Special characters in strings must be properly escaped (\n, \t, \" etc.)
   - Brackets must be properly paired and nested
   - Do not use trailing commas (no comma after the last element)
   - Do not add comments inside JSON
   - Text fields such as description must not contain any JSON fragments
5. All string values use double quotes; numeric values do not use quotes
6. Reiterating: each entry in tocPlan's description must be plain text — no JSON fragments allowed
"""

# Word Budget Planning Prompt
SYSTEM_PROMPT_WORD_BUDGET = f"""
You are the Report Word Budget Planner. You will receive templateOverview (template title + TOC), the latest title/TOC design draft, and all materials, and need to allocate word counts for each chapter and its sub-topics.

Requirements:
1. Total word count is approximately 40,000 words with a 5% tolerance; provide globalGuidelines explaining the overall detail allocation strategy;
2. Each chapter in chapters must include targetWords/min/max, emphasis topics requiring extra development, and a sections array (allocating word counts and notes for sub-sections/outlines of that chapter; may note "allowed to exceed 10% when necessary to add examples");
3. rationale must explain the word count allocation rationale for that chapter, citing key information from the template/materials;
4. Chapter numbering follows first-level Chinese numerals and second-level Arabic numerals, to facilitate unified font sizing later;
5. Output as JSON satisfying the schema below; used only for internal storage and chapter generation, not directly displayed to readers.

<OUTPUT JSON SCHEMA>
{json.dumps(word_budget_output_schema, ensure_ascii=False, indent=2)}
</OUTPUT JSON SCHEMA>

**Important output format requirements:**
1. Return only a pure JSON object conforming to the above schema
2. Do not add any reasoning, explanatory text, or descriptions outside the JSON
3. You may wrap the JSON with ```json and ``` markers, but do not add any other content
4. Ensure the JSON syntax is completely correct:
   - Commas must separate all object and array elements
   - Special characters in strings must be properly escaped (\n, \t, \" etc.)
   - Brackets must be properly paired and nested
   - Do not use trailing commas (no comma after the last element)
   - Do not add comments inside JSON
5. All string values use double quotes; numeric values do not use quotes
"""


def build_chapter_user_prompt(payload: dict) -> str:
    """
    Serializes chapter context into prompt input.

    Uniformly uses `json.dumps(..., indent=2, ensure_ascii=False)` for easier LLM reading.
    """
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_chapter_repair_prompt(chapter: dict, errors, original_text=None) -> str:
    """
    Constructs the chapter repair input payload, including the original chapter and validation errors.
    """
    payload: dict = {
        "failedChapter": chapter,
        "validatorErrors": errors,
    }
    if original_text:
        snippet = original_text[-2000:]
        payload["rawOutputTail"] = snippet
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_chapter_recovery_payload(
    section: dict, generation_payload: dict, raw_output: str
) -> str:
    """
    Constructs the cross-engine JSON recovery input, with chapter metadata, generation instructions, and raw output.

    To avoid overly long prompts, only the tail portion of the raw output is retained for diagnosis.
    """
    payload = {
        "section": section,
        "generationPayload": generation_payload,
        "rawChapterOutput": raw_output[-8000:] if isinstance(raw_output, str) else raw_output,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_document_layout_prompt(payload: dict) -> str:
    """Serializes the context required for document layout design into a JSON string, for the layout node to send to the LLM."""
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_word_budget_prompt(payload: dict) -> str:
    """Converts the word budget planning input into a string for precise field transmission to the LLM."""
    return json.dumps(payload, ensure_ascii=False, indent=2)
