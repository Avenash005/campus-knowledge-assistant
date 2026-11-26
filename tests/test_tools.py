"""
Unit tests for custom Python academic tools.
Tests days_until_deadline and calculate_study_schedule.
"""

import pytest
from datetime import date, timedelta
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import days_until_deadline, calculate_study_schedule


class TestDaysUntilDeadline:
    """Test suite for Academic Deadline Calculator."""

    def test_future_date_standard_format(self):
        target = date.today() + timedelta(days=10)
        res = days_until_deadline.invoke({"deadline": target.strftime("%Y-%m-%d")})
        assert "There are 10 days remaining" in res
        assert target.strftime("%B %d, %Y") in res

    def test_today_date(self):
        today_str = date.today().strftime("%Y-%m-%d")
        res = days_until_deadline.invoke({"deadline": today_str})
        assert "is TODAY! 0 days remaining" in res

    def test_past_date(self):
        past = date.today() - timedelta(days=5)
        res = days_until_deadline.invoke({"deadline": past.strftime("%Y-%m-%d")})
        assert "passed 5 days ago" in res

    def test_flexible_date_formats(self):
        # Month DD, YYYY
        res1 = days_until_deadline.invoke({"deadline": "December 25, 2026"})
        assert "Error" not in res1
        assert "December 25, 2026" in res1

        # DD Month YYYY
        res2 = days_until_deadline.invoke({"deadline": "15 October 2026"})
        assert "Error" not in res2

    def test_invalid_date_format(self):
        res = days_until_deadline.invoke({"deadline": "not-a-valid-date"})
        assert "Error: Unable to parse date" in res

    def test_empty_deadline_input(self):
        res = days_until_deadline.invoke({"deadline": "   "})
        assert "Error: Deadline string cannot be empty" in res


class TestCalculateStudySchedule:
    """Test suite for Study Schedule Calculator."""

    def test_valid_schedule_distribution(self):
        topics = "RAG Systems, Vector Databases, Prompt Engineering"
        hours = 6.0
        res = calculate_study_schedule.invoke({"topics": topics, "available_hours": hours})
        assert "Structured Study Plan: 6.0 Total Hours" in res
        assert "3 Topics" in res
        assert "RAG Systems" in res
        assert "Vector Databases" in res
        assert "Prompt Engineering" in res
        assert "2h 0m" in res
        assert "Pomodoro cycles" in res

    def test_single_topic(self):
        res = calculate_study_schedule.invoke({"topics": "Algorithms", "available_hours": 2.5})
        assert "Structured Study Plan: 2.5 Total Hours" in res
        assert "Algorithms" in res
        assert "2h 30m" in res

    def test_empty_topics_validation(self):
        res = calculate_study_schedule.invoke({"topics": "   ", "available_hours": 5.0})
        assert "Error: Topics list cannot be empty" in res

    def test_zero_or_negative_hours(self):
        res_zero = calculate_study_schedule.invoke({"topics": "Math", "available_hours": 0})
        assert "Error: Available hours must be greater than zero" in res_zero

        res_neg = calculate_study_schedule.invoke({"topics": "Math", "available_hours": -4.5})
        assert "Error: Available hours must be greater than zero" in res_neg

    def test_excessive_hours_limit(self):
        res = calculate_study_schedule.invoke({"topics": "Math", "available_hours": 200.0})
        assert "cannot exceed total hours in a week" in res
