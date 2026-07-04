#!/usr/bin/env python3
"""Open-model size↔quality Pareto frontier for MTEB(por), highlighting Colibri."""
import json, re, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageDraw, ImageFont

def emoji_img(ch="🐦", size=160):
    """Rasterize a color emoji (Apple Color Emoji) to an RGBA array for matplotlib."""
    font = ImageFont.truetype("/System/Library/Fonts/Apple Color Emoji.ttc", size)
    im = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((size / 2, size / 2), ch, font=font,
                            embedded_color=True, anchor="mm")
    return np.asarray(im)

PARETO_JS = "/Users/tardelli/Workplace/mteb-pt-org/data/pareto.js"
OUT = "/Users/tardelli/Workplace/improving-gemma-test/colibri_pareto.png"

# --- load the existing ecosystem data ---
raw = open(PARETO_JS).read()
data = json.loads(raw[raw.index("["): raw.rindex("]") + 1])
for d in data:
    d["colibri"] = False

# --- inject Colibri (this work) ---
data.append({"name": "Colibri", "params": 157.0, "mean": 0.6501, "colibri": True})

# --- recompute the Pareto frontier (min params, max mean) ---
def is_frontier(pt, pts):
    for o in pts:
        if o is pt:
            continue
        if o["params"] <= pt["params"] and o["mean"] >= pt["mean"] and (
           o["params"] < pt["params"] or o["mean"] > pt["mean"]):
            return False
    return True

for d in data:
    d["frontier"] = is_frontier(d, data)

front = sorted([d for d in data if d["frontier"]], key=lambda d: d["params"])
dom = [d for d in data if not d["frontier"]]

# --- shorter display names (keep labels tidy) ---
SHORT = {
    "Qwen3-Embedding-8B": "Qwen3-8B",
    "Qwen3-Embedding-4B": "Qwen3-4B",
    "BidirLM-1.7B-Embedding": "BidirLM-1.7B",
    "serafim-100m-portuguese-pt-sentence-encoder-ir": "Serafim-100M",
    "granite-embedding-97m-multilingual-r2": "Granite-97M-Multilingual",
    "serafim-900m-portuguese-pt-sentence-encoder-ir": "Serafim-900M",
    "multilingual-e5-large-instruct": "multilingual-e5-large-instruct",
    "gte-Qwen2-1.5B-instruct": "gte-Qwen2-1.5B",
}
def disp(name):
    return SHORT.get(name, name)

# --- notable dominated models to label (keep it readable) ---
LABEL_DOM = {
    "Qwen3-Embedding-8B", "harrier-oss-v1-27b", "embeddinggemma-300m",
    "multilingual-e5-large-instruct", "bge-m3", "LaBSE",
    "e5-mistral-7b-instruct", "mxbai-embed-large-v1",
    "serafim-900m-portuguese-pt-sentence-encoder-ir",
    "all-MiniLM-L6-v2",
    "F2LLM-v2-160M", "multilingual-e5-base", "Legal-BERTimbau-sts-large", "gte-Qwen2-1.5B-instruct",
}
# custom label offsets to de-clutter (name -> (dx, dy) in points)
DOM_OFFSET = {
    "multilingual-e5-large-instruct": (6, -12),
    "gte-Qwen2-1.5B-instruct": (6, -11),
    "multilingual-e5-base": (-6, -12),
    "F2LLM-v2-160M": (6, 4),
    "embeddinggemma-300m": (7, -12),
}

RED = "#d62728"
BLUE = "#4a90c2"
GOLD = "#f2a900"

fig, ax = plt.subplots(figsize=(13, 7.6), dpi=170)
ax.set_xscale("log")

# dashed frontier line
ax.plot([d["params"] for d in front], [d["mean"] for d in front],
        "--", color="#999", lw=1.3, zorder=1)

# dominated points
ax.scatter([d["params"] for d in dom], [d["mean"] for d in dom],
           s=42, c=BLUE, edgecolors="white", linewidths=0.6, zorder=2, alpha=0.9)
for d in dom:
    if d["name"] in LABEL_DOM:
        dx, dy = DOM_OFFSET.get(d["name"], (6, 4))
        ax.annotate(disp(d["name"]), (d["params"], d["mean"]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.7, color="#4a4a4a", zorder=5)

# frontier points (red) — excluding Colibri which gets its own style
for d in front:
    if d["colibri"]:
        continue
    ax.scatter(d["params"], d["mean"], s=64, c=RED, edgecolors="white",
               linewidths=0.8, zorder=4)
    ax.annotate(disp(d["name"]), (d["params"], d["mean"]), xytext=(7, 5),
                textcoords="offset points", fontsize=8.2, color=RED,
                fontweight="bold", zorder=6)

# --- Colibri: bird emoji marker + red label ABOVE (same style as other frontier labels) ---
c = next(d for d in data if d["colibri"])
ab = AnnotationBbox(OffsetImage(emoji_img("🐦", 160), zoom=0.20),
                    (c["params"], c["mean"]), frameon=False, zorder=10,
                    box_alignment=(0.5, 0.5))
ax.add_artist(ab)
ax.annotate("Colibri", (c["params"], c["mean"]), xytext=(0, 20),
            textcoords="offset points", fontsize=10.5, color=RED,
            fontweight="bold", ha="center", va="bottom", zorder=11)

# axes cosmetics
ax.set_xlabel("Parameters (log scale)", fontsize=11.5)
ax.set_ylabel("MTEB(por) mean", fontsize=11.5)
ax.set_title("Open-model size ↔ quality frontier on MTEB(por) — Colibri anchors the knee",
             fontsize=13.5, fontweight="bold", pad=14)
ax.set_ylim(0.35, 0.70)
ax.set_xlim(18, 34000)
ticks = [23, 100, 300, 1000, 4000, 10000, 34000]
ax.set_xticks(ticks)
ax.xaxis.set_major_formatter(FuncFormatter(
    lambda v, _: f"{v/1000:.0f}B" if v >= 1000 else f"{v:.0f}M"))
ax.grid(True, which="both", ls=":", lw=0.5, color="#ddd", zorder=0)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# legend
from matplotlib.lines import Line2D
leg = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor=RED, markersize=9,
           label="Pareto frontier"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=BLUE, markersize=8,
           label="dominated models"),
]
ax.legend(handles=leg, loc="lower right", fontsize=9.5, frameon=True, framealpha=0.95,
          title="Colibri (this work) = bird marker")

plt.tight_layout()
plt.savefig(OUT, bbox_inches="tight", facecolor="white")
print("saved:", OUT)
print("\nNEW frontier (params asc):")
for d in front:
    tag = "  <-- COLIBRI" if d["colibri"] else ""
    print(f"  {d['params']:>7.0f}M  {d['mean']:.4f}  {d['name']}{tag}")
