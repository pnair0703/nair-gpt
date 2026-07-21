"""plot_loss.py — render the training/validation loss curve for the README.

Reads history.json (the current run) and overlays the FIRST, un-regularized run
(recorded below) to visualize the overfitting fix. Writes loss_curve.png.

Design: color = which run (orange = first run, blue = regularized), line style =
which metric (solid = validation, dashed = train). The two colors are a
colorblind-safe pair; line style is the secondary encoding.
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BLUE, ORANGE, INK, MUTED = "#0072B2", "#D55E00", "#333333", "#888888"

# The first big run had no dropout / weight decay and overfit hard: train loss
# kept falling while val loss bottomed near step ~1300, then climbed to 2.84.
FIRST_STEPS = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3600]
FIRST_TRAIN = [4.2097, 2.3489, 1.7949, 1.4913, 1.3489, 1.2600, 1.1796, 1.0955, 1.0039, 0.8942, 0.7735, 0.6563, 0.5220, 0.4215, 0.3377, 0.3152]
FIRST_VAL = [4.2191, 2.3762, 1.9281, 1.6902, 1.5834, 1.5503, 1.5528, 1.5818, 1.6621, 1.7416, 1.9030, 2.0611, 2.2669, 2.5241, 2.7549, 2.8385]


def main():
    with open("history.json") as f:
        hist = json.load(f)
    steps = [h[0] for h in hist]
    train = [h[1] for h in hist]
    val = [h[2] for h in hist]

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # First (overfit) run — orange.  Regularized run — blue.  val solid, train dashed.
    ax.plot(FIRST_STEPS, FIRST_TRAIN, color=ORANGE, lw=2, ls="--", alpha=0.9)
    ax.plot(FIRST_STEPS, FIRST_VAL, color=ORANGE, lw=2, ls="-", alpha=0.9)
    ax.plot(steps, train, color=BLUE, lw=2, ls="--")
    ax.plot(steps, val, color=BLUE, lw=2, ls="-")

    # Short line-end labels in ink (identity comes from proximity + legend).
    ax.text(3670, FIRST_VAL[-1], f"val {FIRST_VAL[-1]:.2f}", color=INK, fontsize=9, va="center")
    ax.text(3670, FIRST_TRAIN[-1], f"train {FIRST_TRAIN[-1]:.2f}", color=INK, fontsize=9, va="center")
    ax.text(3670, val[-1] + 0.05, f"val {val[-1]:.2f}", color=INK, fontsize=9, va="center", weight="bold")
    ax.text(3670, train[-1] - 0.06, f"train {train[-1]:.2f}", color=INK, fontsize=9, va="center")

    ax.annotate(
        "overfits: val climbs while\ntrain keeps dropping",
        xy=(3000, 2.2669), xytext=(1950, 3.05),
        color=ORANGE, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2),
    )
    ax.annotate(
        "regularized: val holds,\nsmall stable gap",
        xy=(3200, 1.497), xytext=(1650, 0.55),
        color=BLUE, fontsize=9,
        arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.2),
    )

    ax.set_xlabel("training step", color=INK)
    ax.set_ylabel("cross-entropy loss", color=INK)
    ax.set_title(
        "Overfit vs. regularized run — 10.8M-param GPT on TinyShakespeare",
        color=INK, fontsize=12, weight="bold", pad=12,
    )
    ax.set_xlim(0, 4050)
    ax.set_ylim(0, 4.4)
    ax.grid(axis="y", color=MUTED, alpha=0.2, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK)

    handles = [
        Line2D([0], [0], color=BLUE, lw=2, label="regularized run (dropout + weight decay)"),
        Line2D([0], [0], color=ORANGE, lw=2, label="first run (no regularization)"),
        Line2D([0], [0], color=INK, lw=2, ls="-", label="validation loss"),
        Line2D([0], [0], color=INK, lw=2, ls="--", label="train loss"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right", labelcolor=INK)

    fig.tight_layout()
    fig.savefig("loss_curve.png", facecolor="white", bbox_inches="tight")
    print("wrote loss_curve.png")


if __name__ == "__main__":
    main()
