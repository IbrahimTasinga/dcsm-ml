"""Sample data used ONLY until real history is available. Generated from simple, plausible relationships
(attendance and coursework dominate, participation helps a little). It is NOT real student data."""
import numpy as np
from features import FEATURES


def generate(n=600, seed=7):
    r = np.random.default_rng(seed)
    att = np.clip(r.normal(78, 18, n), 0, 100)
    cw = np.clip(0.45 * att + r.normal(30, 16, n), 0, 100)
    prev = np.where(r.random(n) < 0.35, np.nan, np.clip(0.3 * cw + r.normal(38, 12, n), 0, 100))
    disc = r.poisson(np.clip(att / 25, 0.2, 5))
    grp = r.poisson(np.clip(att / 30, 0.1, 4))
    dl = r.poisson(np.clip(att / 20, 0.3, 6))
    yrs = np.clip(r.normal(4, 3, n), 0, 25).round()
    sessions = r.integers(4, 16, n)
    recent = np.clip(att + r.normal(0, 12, n), 0, 100)
    streak = np.clip(np.round((100 - recent) / 25 + r.normal(0, 0.7, n)), 0, 8)
    deg = r.integers(1, 5, n)
    emp = (r.random(n) < 0.6).astype(float)
    elec = (r.random(n) < 0.3).astype(float)
    sem = r.integers(1, 4, n)
    prevfail = np.where(np.isnan(prev), 0, (prev < 40).astype(float))
    z = -5.0 + 0.035 * att + 0.045 * cw + 0.01 * np.nan_to_num(prev, nan=55) + 0.08 * np.minimum(disc, 5) - 0.25 * streak + r.normal(0, 0.9, n)
    y = (z > 0).astype(int)
    cols = dict(att_pct=att, att_recent_pct=recent, att_missed_streak=streak, sessions_held=sessions, cw_pct=cw,
                cw_done=r.integers(1, 5, n).astype(float), prev_avg_pct=prev, prev_fail_count=prevfail,
                discussion_posts=disc, group_posts=grp, note_downloads=dl, years_experience=yrs, degree_level=deg.astype(float),
                employed=emp, is_elective=elec, semester_no=sem.astype(float))
    X = np.column_stack([np.asarray(cols[f], dtype=float) for f in FEATURES])
    return X, y
