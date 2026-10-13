# Project guidance

This is a reinforcement learning course mini project for post-contest Codeforces review. Read PROJECT_PLAN.md and STATUS.md when present before continuing project work.

- Analyse completed participant-contest sessions. The complete target contest is available at review time.
- The active strategy samples a complete contest, without collecting college/country. The older organisation pilot remains a reproducible checkpoint. Organisation/college must never become a model observation.
- Coverage groups refer only to the frozen earlier-contest panel. Missing or incomplete history is never silently classified as no history. Preserve the full target population outside the development sample.
- Collection currently uses the Python standard library. Run it from this directory.
- Preserve raw responses, request provenance, checkpoints and dataset versions. Do not hide incomplete collection or change a frozen run's settings.
- Unusual behaviour and administrative outcomes do not automatically establish misconduct labels. Unknown labels remain unknown until independently reviewed.
- Earlier behavioural context must predate the target contest. Future records cannot influence earlier features.
- Keep manual label worksheets intact when refreshing reports.
- Run python -m unittest discover -s tests -v after relevant changes to collection/session logic.
- The user authorized an automated 2026 recorded-outcome benchmark without manual source review. Training may use explicitly named outcome proxy targets after eligibility, split and leakage audits. Such targets are separate from unknown misconduct labels and do not validate cheating or AI-use accuracy. Preserve the older 2025 and contest-002 checkpoints.
- Record concrete completed work and unresolved issues in STATUS.md. Do not claim tests or live checks that were not run.

- The active project strictly uses RL algorithms: Q-learning, SARSA, DQN and dueling Double DQN. Read docs/DEEP_RL.md before continuing model work. Preserve historical classifier artifacts, but do not add supervised models to the active comparison.
- deep-rl-2026-001 fresh test contest 2274 has now been evaluated. Do not claim it is untouched in subsequent tuning. Previously tested 2264/2267/2269 are explicitly validation development data. Recent administrative outcomes may change.
