"""Additional offline feature and evidence commands."""

import json

from .features import VERSION, build_feature_report
from .review import export_review_context, export_review_register, import_reviews, review_audit


def add_commands(commands):
    from pathlib import Path
    features = commands.add_parser("features", help="Audit and build deterministic development features offline")
    features.add_argument("--run-id", required=True)
    features.add_argument("--output", type=Path)
    review = commands.add_parser("review-export", help="Create a random evidence review batch without overwriting it")
    review.add_argument("--run-id", required=True)
    review.add_argument("--output", type=Path)
    review.add_argument("--limit", type=int, default=100)
    review.add_argument("--seed", type=int, default=736)
    load = commands.add_parser("review-import", help="Validate an evidence register; --apply records reviews and labels")
    load.add_argument("--run-id", required=True)
    load.add_argument("--input", type=Path, required=True)
    load.add_argument("--apply", action="store_true")
    audit = commands.add_parser("review-check", help="Count current review evidence and labelled session coverage")
    audit.add_argument("--run-id", required=True)


def dispatch(store, args):
    if args.command == "features":
        output = args.output or args.reports_dir / args.run_id / VERSION
        result = build_feature_report(store, args.run_id, output)
        result = {key: result[key] for key in ("run_id", "feature_version", "rows", "used_eligible_submissions", "training_ready")} | {"output": str(output)}
    elif args.command == "review-export":
        output = args.output or args.reports_dir / args.run_id / "evidence_register.csv"
        result = export_review_register(store, args.run_id, output, args.limit, args.seed)
        if not result["preserved_existing"]:
            result["context"] = export_review_context(store, args.run_id, output, output.with_suffix(".context.json"))
    elif args.command == "review-import":
        result = import_reviews(store, args.run_id, args.input, args.apply)
    else:
        result = review_audit(store, args.run_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0
