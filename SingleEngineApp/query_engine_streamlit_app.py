"""
Streamlit Web Interface
Provides a user-friendly web interface for Query Agent
"""

import os
import sys
import streamlit as st
from datetime import datetime
import json
import locale
from loguru import logger

# Set UTF-8 encoding environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Set system encoding
try:
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
except locale.Error:
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        pass

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from QueryEngine import DeepSearchAgent, Settings
from config import settings
from utils.github_issues import error_with_issue_link


def main():
    """Main function."""
    st.set_page_config(
        page_title="Query Agent",
        page_icon="",
        layout="wide"
    )

    st.title("Query Agent")
    st.markdown("An AI agent with powerful web search capabilities")
    st.markdown("Broadly crawls official reports and news, combining domestic and international resources for public-opinion understanding")

    # Check URL parameters
    try:
        # Try using the newer query_params API
        query_params = st.query_params
        auto_query = query_params.get('query', '')
        auto_search = query_params.get('auto_search', 'false').lower() == 'true'
    except AttributeError:
        # Backward compatibility for older versions
        query_params = st.experimental_get_query_params()
        auto_query = query_params.get('query', [''])[0]
        auto_search = query_params.get('auto_search', ['false'])[0].lower() == 'true'

    # ----- Configuration is hardcoded -----
    # Force use of DeepSeek
    model_name = settings.QUERY_ENGINE_MODEL_NAME or "deepseek-chat"
    # Default advanced configuration
    max_reflections = 2
    max_content_length = 20000

    # Simplified research query display area

    # If auto query exists, use it as default value; otherwise show placeholder
    display_query = auto_query if auto_query else "Waiting to receive analysis content from the main page..."

    # Read-only query display area
    st.text_area(
        "Current Query",
        value=display_query,
        height=100,
        disabled=True,
        help="Query content is controlled by the search box on the main page",
        label_visibility="hidden"
    )

    # Auto search logic
    start_research = False
    query = auto_query

    if auto_search and auto_query and 'auto_search_executed' not in st.session_state:
        st.session_state.auto_search_executed = True
        start_research = True
    elif auto_query and not auto_search:
        st.warning("Waiting for search start signal...")

    # Validate configuration
    if start_research:
        if not query.strip():
            st.error("Please enter a research query")
            return

        # Since DeepSeek is enforced, check related API keys
        if not settings.QUERY_ENGINE_API_KEY:
            st.error("Please set QUERY_ENGINE_API_KEY in your environment variables")
            return
        if not settings.TAVILY_API_KEY:
            st.error("Please set TAVILY_API_KEY in your environment variables")
            return

        # Automatically use API keys from the configuration file
        engine_key = settings.QUERY_ENGINE_API_KEY
        tavily_key = settings.TAVILY_API_KEY

        # Create configuration
        config = Settings(
            QUERY_ENGINE_API_KEY=engine_key,
            QUERY_ENGINE_BASE_URL=settings.QUERY_ENGINE_BASE_URL,
            QUERY_ENGINE_MODEL_NAME=model_name,
            TAVILY_API_KEY=tavily_key,
            MAX_REFLECTIONS=max_reflections,
            SEARCH_CONTENT_MAX_LENGTH=max_content_length,
            OUTPUT_DIR="query_engine_streamlit_reports"
        )

        # Execute research
        execute_research(query, config)


def execute_research(query: str, config: Settings):
    """Execute research."""
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Initialize agent
        status_text.text("Initializing Agent...")
        agent = DeepSearchAgent(config)
        st.session_state.agent = agent

        progress_bar.progress(10)

        # Generate report structure
        status_text.text("Generating report structure...")
        agent._generate_report_structure(query)
        progress_bar.progress(20)

        # Process paragraphs
        total_paragraphs = len(agent.state.paragraphs)
        for i in range(total_paragraphs):
            status_text.text(f"Processing paragraph {i + 1}/{total_paragraphs}: {agent.state.paragraphs[i].title}")

            # Initial search and summary
            agent._initial_search_and_summary(i)
            progress_value = 20 + (i + 0.5) / total_paragraphs * 60
            progress_bar.progress(int(progress_value))

            # Reflection loop
            agent._reflection_loop(i)
            agent.state.paragraphs[i].research.mark_completed()

            progress_value = 20 + (i + 1) / total_paragraphs * 60
            progress_bar.progress(int(progress_value))

        # Generate final report
        status_text.text("Generating final report...")
        final_report = agent._generate_final_report()
        progress_bar.progress(90)

        # Save report
        status_text.text("Saving report...")
        agent._save_report(final_report)
        progress_bar.progress(100)

        status_text.text("Research completed!")

        # Display results
        display_results(agent, final_report)

    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        error_display = error_with_issue_link(
            f"An error occurred during research: {str(e)}",
            error_traceback,
            app_name="Query Engine Streamlit App"
        )
        st.error(error_display)
        logger.exception(f"An error occurred during research: {str(e)}")


def display_results(agent: DeepSearchAgent, final_report: str):
    """Display research results."""
    st.header("Research Results")

    # Result tabs (download option removed)
    tab1, tab2 = st.tabs(["Research Summary", "Reference Information"])

    with tab1:
        st.markdown(final_report)

    with tab2:
        # Paragraph details
        st.subheader("Paragraph Details")
        for i, paragraph in enumerate(agent.state.paragraphs):
            with st.expander(f"Paragraph {i + 1}: {paragraph.title}"):
                st.write("**Expected Content:**", paragraph.content)
                st.write("**Final Content:**", paragraph.research.latest_summary[:300] + "..."
                if len(paragraph.research.latest_summary) > 300
                else paragraph.research.latest_summary)
                st.write("**Search Count:**", paragraph.research.get_search_count())
                st.write("**Reflection Count:**", paragraph.research.reflection_iteration)

        # Search history
        st.subheader("Search History")
        all_searches = []
        for paragraph in agent.state.paragraphs:
            all_searches.extend(paragraph.research.search_history)

        if all_searches:
            for i, search in enumerate(all_searches):
                with st.expander(f"Search {i + 1}: {search.query}"):
                    st.write("**URL:**", search.url)
                    st.write("**Title:**", search.title)
                    st.write("**Content Preview:**",
                             search.content[:200] + "..." if len(search.content) > 200 else search.content)
                    if search.score:
                        st.write("**Relevance Score:**", search.score)


if __name__ == "__main__":
    main()
