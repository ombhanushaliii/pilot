"""Train a frozen experiment with one BLAS thread and no browser/network work."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.training import train_run

if __name__ == "__main__":
    config = json.loads((ROOT / "config/training-2026.json").read_text(encoding="utf-8"))
    result = train_run(ROOT, config)
    print(json.dumps({"selected_rl": result["selected_rl"], "seed_aggregate": result["rl_seed_aggregate"]}, indent=2))
