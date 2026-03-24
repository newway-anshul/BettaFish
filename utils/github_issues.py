"""
GitHub Issues utility module.

Provides functions to create GitHub Issues URLs and display error messages with links.
Data model definition location:
- No data model.
"""

from datetime import datetime
from urllib.parse import quote

# GitHub repository information.
GITHUB_REPO = "666ghj/BettaFish"
GITHUB_ISSUES_URL = f"https://github.com/{GITHUB_REPO}/issues/new"


def create_issue_url(title: str, body: str = "") -> str:
    """Create a GitHub Issues URL with prefilled title and body.

    Args:
        title: Issue title.
        body: Issue body (optional).

    Returns:
        Full GitHub Issues URL.
    """
    encoded_title = quote(title)
    encoded_body = quote(body) if body else ""
    
    if encoded_body:
        return f"{GITHUB_ISSUES_URL}?title={encoded_title}&body={encoded_body}"
    else:
        return f"{GITHUB_ISSUES_URL}?title={encoded_title}"


def error_with_issue_link(
    error_message: str,
    error_details: str = "",
    app_name: str = "Streamlit App"
) -> str:
    """Generate an error message string with a GitHub Issues link.

    Intended for generic exception handling, not user configuration errors.

    Args:
        error_message: Error message.
        error_details: Error details (optional, included in issue body).
        app_name: Application name used to identify error source.

    Returns:
        Markdown string containing the error and a GitHub Issues link.
    """
    issue_title = f"[{app_name}] {error_message[:50]}"
    issue_body = f"## Error Message\n\n{error_message}\n\n"
    
    if error_details:
        issue_body += f"## Error Details\n\n```\n{error_details}\n```\n\n"
    
    issue_body += f"## Environment\n\n- App: {app_name}\n- Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    issue_url = create_issue_url(issue_title, issue_body)
    
    # Add hyperlink in markdown format.
    error_display = f"{error_message}\n\n[📝 Report this issue]({issue_url})"
    
    if error_details:
        error_display = f"{error_message}\n\n```\n{error_details}\n```\n\n[📝 Report this issue]({issue_url})"
    
    return error_display


__all__ = [
    "create_issue_url",
    "error_with_issue_link",
    "GITHUB_REPO",
    "GITHUB_ISSUES_URL",
]

