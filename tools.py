"""
Original custom Python tools for academic workflows.
Includes Academic Deadline Calculator and Study Schedule Calculator.
"""

from datetime import datetime, date
from typing import Dict, Any, List
import re
from langchain_core.tools import tool


def _parse_flexible_date(date_str: str) -> date:
    """Parse dates in various human and standard formats."""
    cleaned = date_str.strip().strip("'\"")
    # Formats to attempt
    formats = [
        "%Y-%m-%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y/%m/%d",
        "%B %d %Y",
        "%b %d %Y",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue
    
    # Try regex matching for common date expressions like "October 24, 2026" or "15 December 2026"
    month_regex = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    match = re.search(rf"{month_regex}\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", cleaned, re.IGNORECASE)
    if match:
        m, d, y = match.groups()
        try:
            return datetime.strptime(f"{m} {d} {y}", "%B %d %Y").date()
        except ValueError:
            pass

    match_dmy = re.search(rf"(\d{{1,2}})(?:st|nd|rd|th)?\s+{month_regex},?\s+(\d{{4}})", cleaned, re.IGNORECASE)
    if match_dmy:
        d, m, y = match_dmy.groups()
        try:
            return datetime.strptime(f"{d} {m} {y}", "%d %B %Y").date()
        except ValueError:
            pass

    raise ValueError(
        f"Unable to parse date '{date_str}'. Please use standard formats such as "
        f"'YYYY-MM-DD' (e.g., 2026-12-15) or 'Month DD, YYYY' (e.g., December 15, 2026)."
    )


@tool
def days_until_deadline(deadline: str) -> str:
    """Calculate the number of days remaining until an academic deadline."""
    if not deadline or not isinstance(deadline, str) or not deadline.strip():
        return "Error: Deadline string cannot be empty. Please provide a valid date string."

    try:
        target_date = _parse_flexible_date(deadline)
    except ValueError as e:
        return f"Error: {str(e)}"

    today = date.today()
    delta_days = (target_date - today).days

    formatted_target = target_date.strftime("%B %d, %Y")

    if delta_days == 0:
        return f"The deadline ({formatted_target}) is TODAY! 0 days remaining."
    elif delta_days > 0:
        day_str = "day" if delta_days == 1 else "days"
        return f"There are {delta_days} {day_str} remaining until the deadline ({formatted_target})."
    else:
        past_days = abs(delta_days)
        day_str = "day" if past_days == 1 else "days"
        return f"The deadline ({formatted_target}) passed {past_days} {day_str} ago."


@tool
def calculate_study_schedule(topics: str, available_hours: float) -> str:
    """Distribute available study hours across requested topics and return a structured study plan."""
    # Input validation
    if not topics or not isinstance(topics, str) or not topics.strip():
        return "Error: Topics list cannot be empty. Please provide at least one topic (e.g., 'RAG, Vector DBs, Prompting')."

    try:
        hours = float(available_hours)
    except (ValueError, TypeError):
        return f"Error: Available hours must be a valid positive number. Received: {available_hours}"

    if hours <= 0:
        return f"Error: Available hours must be greater than zero. Received: {hours} hours."
    if hours > 168:
        return f"Error: Available hours ({hours}) cannot exceed total hours in a week (168 hours)."

    # Split topics by comma, semicolon, or newline
    topic_list = [t.strip() for t in re.split(r"[,;\n]+", topics) if t.strip()]
    if not topic_list:
        return "Error: No valid topics found after parsing input."

    num_topics = len(topic_list)
    total_minutes = int(hours * 60)
    minutes_per_topic = total_minutes // num_topics
    remainder_minutes = total_minutes % num_topics

    # Calculate pomodoro blocks (25 min study + 5 min break = 30 min cycle)
    pomodoro_cycles = total_minutes // 30

    plan_lines: List[str] = [
        f"### Structured Study Plan: {hours:.1f} Total Hours ({total_minutes} mins) for {num_topics} Topics",
        "",
        "| # | Topic | Allocated Time | Recommended Sessions |",
        "|---|---|---|---|",
    ]

    for idx, topic in enumerate(topic_list, start=1):
        # Distribute remaining minutes to the first few topics
        allocated = minutes_per_topic + (1 if idx <= remainder_minutes else 0)
        topic_hours = allocated // 60
        topic_mins = allocated % 60
        time_str = f"{topic_hours}h {topic_mins}m" if topic_hours > 0 else f"{topic_mins}m"
        sessions = max(1, round(allocated / 25))
        plan_lines.append(f"| {idx} | **{topic}** | {time_str} | ~{sessions} x 25-min focus blocks |")

    plan_lines.extend([
        "",
        f"**Efficiency Tips:**",
        f"- Target ~{pomodoro_cycles} Pomodoro cycles (25 min focus + 5 min rest).",
        f"- Take an extended 15-minute break after every 4 cycles.",
        f"- Review summary notes during the final 10 minutes of each topic block."
    ])

    return "\n".join(plan_lines)


# List of custom tools for export
CUSTOM_TOOLS = [days_until_deadline, calculate_study_schedule]
TOOL_MAP = {t.name: t for t in CUSTOM_TOOLS}
