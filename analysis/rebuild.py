"""One-command rebuild of the entire artifact chain (deterministic, seed 42).

Regenerates, in dependency order:
  1. synthetic augmentation -> data/augmented_orders.csv   (EDA/dashboard only)
  2. trained models + leaderboard + report artifacts        (models/*)
  3. the dashboard's app/public/predictions.json
  4. a copy of the served models into api/models/           (for the Railway deploy)
  5. (optional, --report) renders report.qmd -> report.html

Run from analysis/:
    python rebuild.py
    python rebuild.py --report
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src import augment, models           # noqa: E402
import export_predictions                 # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", action="store_true", help="also render report.qmd to HTML")
    args = ap.parse_args()

    print("[1/4] synthetic augmentation ...")
    augment.augment_orders().to_csv(augment.OUT_CSV, index=False)

    print("[2/4] training model ladders (XGBoost search — a minute or two) ...")
    models.train_all(write=True)

    print("[3/4] exporting predictions.json ...")
    payload = export_predictions.build_payload()
    export_predictions.OUT.parent.mkdir(parents=True, exist_ok=True)
    export_predictions.OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("[4/4] syncing served models into api/models/ ...")
    api_models = HERE.parent / "api" / "models"
    api_models.mkdir(parents=True, exist_ok=True)
    for fname in ("daily_revenue.joblib", "high_demand.joblib"):
        (api_models / fname).write_bytes((models.MODELS_DIR / fname).read_bytes())

    if args.report:
        print("[+] rendering report.qmd ...")
        subprocess.run(["quarto", "render", "report.qmd"], cwd=HERE, check=True)

    print("\nDone. Regenerated: models/, predictions.json, api/models/"
          + (", report.html" if args.report else ""))


if __name__ == "__main__":
    main()
