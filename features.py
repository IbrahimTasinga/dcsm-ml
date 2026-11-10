"""Feature definitions shared by training and prediction. All features are numeric; None = unknown."""

FEATURES = [
    "att_pct", "att_recent_pct", "att_missed_streak", "sessions_held",
    "cw_pct", "cw_done", "prev_avg_pct", "prev_fail_count",
    "discussion_posts", "group_posts", "note_downloads",
    "years_experience", "degree_level", "employed", "is_elective", "semester_no",
]

LABELS = {
    "att_pct": "attendance so far",
    "att_recent_pct": "recent attendance",
    "att_missed_streak": "consecutive sessions missed",
    "sessions_held": "number of sessions held",
    "cw_pct": "coursework average",
    "cw_done": "assessments completed",
    "prev_avg_pct": "results in earlier semesters",
    "prev_fail_count": "modules failed before",
    "discussion_posts": "discussion participation",
    "group_posts": "study-group participation",
    "note_downloads": "notes downloaded",
    "years_experience": "years of work experience",
    "degree_level": "level of previous degree",
    "employed": "employment status",
    "is_elective": "elective module",
    "semester_no": "semester",
}
