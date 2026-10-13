"""Run the local cached-contest review portal."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cf_pilot.portal import serve

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local post-contest review portal using saved RL weights")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Use a port between 1024 and 65535")
    try:
        serve(ROOT, args.port)
    except (OSError, ValueError, KeyError) as error:
        print("Portal could not start:", error, file=sys.stderr)
        print("The demo requires the existing verified datasets and model files. See docs/PORTAL.md.", file=sys.stderr)
        sys.exit(1)
