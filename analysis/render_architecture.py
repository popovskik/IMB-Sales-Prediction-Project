"""Render the D0 solution-architecture diagram to architecture.png.

The mermaid MCP / mmdc renderer fails on this Windows machine (npx/puppeteer),
so we draw the same diagram (same nodes, edges, and offline/runtime split as
architecture.md's Mermaid source) deterministically with matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "architecture.png"

# Palette
OFFLINE_BG = "#eaf1fd"
RUNTIME_BG = "#e7f6ed"
PROC = "#ffffff"
MODEL = "#fff0ea"      # brand-orange tint for serialized models
MODEL_EDGE = "#ff4800"
USER = "#f2f4f7"
INK = "#101828"
ARROW = "#475467"

fig, ax = plt.subplots(figsize=(12, 9))
ax.set_xlim(0, 12)
ax.set_ylim(0, 9)
ax.axis("off")

# --- Region backgrounds -----------------------------------------------------
ax.add_patch(FancyBboxPatch((0.3, 3.7), 11.4, 5.0, boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc=OFFLINE_BG, ec="#aac4f0", lw=1.2, zorder=0))
ax.add_patch(FancyBboxPatch((0.3, 0.4), 11.4, 3.0, boxstyle="round,pad=0.02,rounding_size=0.15",
                            fc=RUNTIME_BG, ec="#a6d8bb", lw=1.2, zorder=0))
ax.text(0.55, 8.45, "OFFLINE  ·  trained once on the developer machine",
        fontsize=12, fontweight="bold", color="#2360c4", zorder=1)
ax.text(0.55, 3.12, "RUNTIME  ·  the live app",
        fontsize=12, fontweight="bold", color="#1c8a4e", zorder=1)

boxes: dict[str, tuple[float, float]] = {}


def box(key, x, y, w, h, text, fc=PROC, ec="#cdd5e0", text_color=INK, fontsize=10, lw=1.3):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.03,rounding_size=0.08",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize, color=text_color, zorder=3)
    boxes[key] = (x, y, w, h)


def arrow(k1, k2, label="", style="-|>", rad=0.0, color=ARROW, label_dx=0.15, label_dy=0.0):
    x1, y1, w1, h1 = boxes[k1]
    x2, y2, w2, h2 = boxes[k2]
    # connect from edge midpoints (vertical bias)
    if y1 > y2:
        p1 = (x1, y1 - h1 / 2); p2 = (x2, y2 + h2 / 2)
    elif y1 < y2:
        p1 = (x1, y1 + h1 / 2); p2 = (x2, y2 - h2 / 2)
    else:
        if x1 < x2:
            p1 = (x1 + w1 / 2, y1); p2 = (x2 - w2 / 2, y2)
        else:
            p1 = (x1 - w1 / 2, y1); p2 = (x2 + w2 / 2, y2)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                                 lw=1.5, color=color, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if label:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx + label_dx, my + label_dy, label, ha="center", fontsize=8.5,
                color=color, style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.85))


# --- Offline nodes ----------------------------------------------------------
box("csv", 2.6, 7.9, 3.2, 0.85, "Maven Pizza Sales CSVs\norders / order_details /\npizzas / pizza_types",
    fc="#dfe9fb", fontsize=9)
box("agg", 2.6, 6.7, 3.2, 0.7, "Join + aggregate\n-> 365 day-rows", fontsize=9.5)
box("feat", 2.6, 5.5, 3.2, 0.7, "Feature engineering\ncalendar + sin/cos (leakage-safe)", fontsize=9)
box("reg", 6.6, 6.5, 3.4, 0.7, "Regression ladder\nDummy -> Ridge -> Tree -> XGBoost", fontsize=9)
box("clf", 6.6, 5.4, 3.4, 0.7, "Classification ladder\nDummy -> LogReg -> Tree -> XGBoost", fontsize=9)
box("mreg", 5.6, 4.25, 2.6, 0.6, "daily_revenue.joblib", fc=MODEL, ec=MODEL_EDGE, fontsize=9)
box("mclf", 8.6, 4.25, 2.6, 0.6, "high_demand.joblib", fc=MODEL, ec=MODEL_EDGE, fontsize=9)
box("pred", 10.4, 6.0, 2.4, 0.7, "predictions.json\n(charts + scorecard)", fc="#fdf1d8", ec="#e0c074", fontsize=9)

arrow("csv", "agg")
arrow("agg", "feat")
arrow("feat", "reg", rad=-0.15)
arrow("feat", "clf", rad=-0.05)
arrow("reg", "mreg", rad=0.0)
arrow("clf", "mclf", rad=0.0)
arrow("reg", "pred", rad=0.1)
arrow("clf", "pred", rad=0.2)

# --- Runtime nodes ----------------------------------------------------------
box("api", 3.0, 1.7, 3.2, 0.8, "FastAPI  POST /predict\n(Railway)", fc="#ffffff", ec="#cdd5e0", fontsize=9.5)
box("app", 7.2, 1.7, 3.0, 0.8, "React dashboard\n(Vercel)", fc="#ffffff", ec="#cdd5e0", fontsize=9.5)
box("user", 10.6, 1.7, 1.8, 0.8, "Restaurant\nowner", fc=USER, ec="#b0b8c4", fontsize=9.5)

# models load into the API at startup (crosses the offline/runtime divide)
arrow("mreg", "api", rad=0.1)
arrow("mclf", "api", rad=0.15)
# user -> dashboard
arrow("user", "app")
# dashboard <-> api (separate the two arcs so labels don't collide)
arrow("app", "api", label="date", rad=0.32, label_dy=0.5)
arrow("api", "app", label="revenue + demand", rad=0.32, label_dy=-0.5)
# predictions.json feeds the dashboard (charts + fallback)
arrow("pred", "app", label="baked data / fallback", rad=-0.25, color="#8c5a00")

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
