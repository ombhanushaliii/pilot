"""Build or resume the frozen 2026 benchmark using verified local archives only."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.benchmark import build_benchmark

if __name__ == "__main__":
    config = json.loads((ROOT / "config/benchmark-2026.json").read_text(encoding="utf-8"))
    result = build_benchmark(ROOT, config)
    print(json.dumps({k: result[k] for k in ("benchmark_id", "complete", "partitions", "network_requests")}, indent=2))
