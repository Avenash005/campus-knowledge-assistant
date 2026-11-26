# Project Plan — Campus Knowledge Assistant (`plan.md`)
LU10 AI Capstone Project

## 1. Problem Statement
College and university students face significant friction navigating scattered academic resources across diverse platforms. Information about course syllabi, examination schedules, grading policies, late penalties, regrading appeals, laboratory hours, and campus wellness support is routinely spread across static PDFs, email announcements, and department websites. Furthermore, students frequently need to calculate deadlines and structure study routines around upcoming coursework.

## 2. Target Users & Job-to-be-Done
- **Primary User:** Undergraduate and graduate university students.
- **Core Job-to-be-Done:** Provide an intelligent, conversational campus knowledge assistant that quickly answers academic policy and course questions with verifiable citations, accurately resolves follow-up questions using conversation history, and calculates deadlines and study schedules via specialized tools.

## 3. LangGraph Workflow & Three Routing Branches
The application is architected around a stateful LangGraph directed acyclic graph (DAG) with three conditional routing branches:

```
[User Query]
     │
     ▼
[Question Rewriter] ── (Resolves conversational follow-ups & coreferences)
     │
     ▼
[Intent Router]
     │
     ├─────────────┬─────────────┐
     ▼             ▼             ▼
[Route A: DOCS] [Route B: TOOLS] [Route C: DECLINE]
     │             │             │
[Retrieval]   [Tool Agent]  [Decline Node]
     │             │             │
[Context]     [Execution]   [Standard Decline]
     │             │             │
[Response]    [Response]         │
     │             │             │
     └─────────────┴─────────────┘
                   │
                   ▼
                 [END]
```

### Route A — DOCS
- **Purpose:** Answers questions requiring knowledge from campus documents.
- **Pipeline:** `Question Rewriter` ➔ `Intent Router` ➔ `Knowledge Base Retrieval` ➔ `Response Generator` ➔ `END`.
- **Groundedness Guarantee:** Citations to source files are strictly attached. If the question cannot be answered from the retrieved passages, the exact fallback `"I couldn't find that information in the available documents."` is returned.

### Route B — TOOLS
- **Purpose:** Executes deterministic academic calculations and scheduling requests.
- **Pipeline:** `Question Rewriter` ➔ `Intent Router` ➔ `Tool Execution Agent` ➔ `END`.
- **Step Tracking:** Every tool execution records parameters and outputs into an expandable "Agent Steps" view in the UI.

### Route C — DECLINE
- **Purpose:** Safely intercepts and rejects unrelated, unsafe, or adversarial queries (e.g. cooking recipes, cryptocurrency, prompt injection attempts).
- **Pipeline:** `Question Rewriter` ➔ `Intent Router` ➔ `Decline Node` ➔ `END`.
- **Standard Response:** `"I'm designed to assist with campus and academic information. I can't help with that request."`

---

## 4. Knowledge Base Documents
Three original, realistic, topic-specific markdown documents located in `knowledge_base/`:
1. `course_overview.md`: Comprehensive syllabus for CS 402 (Applied AI & Intelligent Systems) covering instructor/TA info, weekly curriculum, 14-week schedule, 4-assignment deadlines, midterm exam date (October 24, 2026), and Capstone submission deadline (December 15, 2026).
2. `academic_policies.md`: Institutional academic policies including 80% attendance mandate, late submission penalty tiers (10% per 24 hours up to 72 hours cutoff), plagiarism rules (Turnitin/MOSS threshold 15%), generative AI disclosure statement requirement, regrade appeal deadline (7 calendar days), and course withdrawal deadlines.
3. `student_guide.md`: Student services and campus facilities including Turing Computing Lab hardware (RTX 4090 GPUs) and operating hours (Mon-Fri 8am-10pm, Sat 10am-6pm), library study room reservations (up to 3 hours/day), Peer Tutoring Center, Student Counseling & Wellness Center (24/7 hotline 555-0199), and parking/shuttle services.

---

## 5. Custom Python Tools
Two original, non-trivial Python tools implemented with strict type annotations, one-sentence docstrings, input validation, and zero side-effects:

1. `days_until_deadline(deadline: str) -> str`:
   - Calculates the exact number of days remaining until an academic deadline or date.
   - Handles future dates, today's date, past dates (indicating how many days ago it passed), and invalid date strings with helpful formatting guidance.

2. `calculate_study_schedule(topics: str, available_hours: float) -> str`:
   - Proportionally distributes available study hours across requested topics.
   - Generates a structured markdown study table detailing hours, minutes, recommended Pomodoro 25-minute focus intervals, and efficiency tips.
   - Validates available hours (>0 and <=168 hours) and non-empty topic strings.

---

## 6. Extra Feature: Interactive Suggested Questions
Four interactive one-click question buttons rendered prominently on the homepage:
1. *"What topics are covered in the course?"*
2. *"What are the academic policies?"*
3. *"How many days remain until my deadline?"*
4. *"Help me create a study schedule."*

Clicking any button passes the query through the full LangGraph state machine and execution workflow rather than returning hardcoded text.
