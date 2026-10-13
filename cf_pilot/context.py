"""Verdict-free historical summaries from strictly earlier completed contests."""
import math
from statistics import median

from .benchmark import FEATURE_COLUMNS

HISTORY_COLUMNS = (
    "history_count_log", "history_recency_days_log", "history_attempts_log_median",
    "history_problems_log_median", "history_first_median", "history_last_median",
    "history_gap_log_median", "attempts_history_log_deviation",
    "problems_history_log_deviation", "first_history_deviation",
    "last_history_deviation", "gap_history_log_deviation", "retry_history_deviation",
)
CONTEXT_COLUMNS = FEATURE_COLUMNS + HISTORY_COLUMNS


def compact_history(row, end):
    """Store only allowlisted behaviour; never outcomes or administrative fields."""
    values = [row[n] for n in ("attempts", "attempted_problems",
              "first_submission_fraction", "last_submission_fraction",
              "gap_median_fraction", "retry_fraction")]
    return (end, *values)


def history_features(row, previous, start):
    if any(p[0] >= start for p in previous):
        raise ValueError("History must end strictly before the target contest")
    answer = dict.fromkeys(HISTORY_COLUMNS)
    answer["history_count_log"] = math.log1p(len(previous))
    if not previous:
        return answer
    answer["history_recency_days_log"] = math.log1p((start - previous[-1][0]) / 86400)
    attempts = median(math.log1p(p[1]) for p in previous)
    problems = median(math.log1p(p[2]) for p in previous)
    first, last = median(p[3] for p in previous), median(p[4] for p in previous)
    gaps = [math.log(max(p[5], 1e-6)) for p in previous if p[5] is not None]
    gap = median(gaps) if gaps else None
    answer.update(history_attempts_log_median=attempts,
                  history_problems_log_median=problems,
                  history_first_median=first, history_last_median=last,
                  history_gap_log_median=gap,
                  attempts_history_log_deviation=math.log1p(row["attempts"]) - attempts,
                  problems_history_log_deviation=math.log1p(row["attempted_problems"]) - problems,
                  first_history_deviation=row["first_submission_fraction"] - first,
                  last_history_deviation=row["last_submission_fraction"] - last,
                  gap_history_log_deviation=(math.log(max(row["gap_median_fraction"], 1e-6)) - gap
                      if row["gap_median_fraction"] is not None and gap is not None else None),
                  retry_history_deviation=row["retry_fraction"] - median(p[6] for p in previous))
    if any(v is not None and not math.isfinite(v) for v in answer.values()):
        raise ValueError("Non-finite history feature")
    return answer
