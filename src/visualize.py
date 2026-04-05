import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from src.config import LOCATIONS, AP_POSITIONS, LOCATION_NAMES, OUTPUT_DIR

def plot_confusion_matrices(results, y_test):
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    loc_names = [LOCATION_NAMES[i] for i in sorted(LOCATION_NAMES)]
    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(y_test, res["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=loc_names, yticklabels=loc_names, ax=ax)
        ax.set_title(f"{name.upper()} | Acc: {res['accuracy']*100:.1f}%", fontsize=12)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.close()

def plot_floor_map(best_model_name, y_pred, y_test):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f9fa")
    ax.set_title(f"Floor Map — {best_model_name.upper()} Predictions", fontsize=14, pad=15)
    ax.set_xlabel("X (meters)"); ax.set_ylabel("Y (meters)")
    for x in range(0, 7, 2):
        ax.axvline(x, color="#dee2e6", lw=0.8, ls="--")
    for y in range(0, 7, 2):
        ax.axhline(y, color="#dee2e6", lw=0.8, ls="--")
    for i, (ax_x, ax_y) in enumerate(AP_POSITIONS):
        ax.scatter(ax_x, ax_y, marker="^", s=200, color="#e63946",
                   zorder=5, edgecolors="white", lw=1)
        ax.annotate(f"AP{i+1}", (ax_x, ax_y),
                    textcoords="offset points", xytext=(6, 4), fontsize=8, color="#e63946")
    np.random.seed(0)
    for actual, predicted in zip(y_test[:40], y_pred[:40]):
        lx, ly = LOCATIONS[actual]
        jx = lx + np.random.uniform(-0.2, 0.2)
        jy = ly + np.random.uniform(-0.2, 0.2)
        ax.scatter(jx, jy, s=50, color="#4361ee", alpha=0.6, zorder=4)
        if actual != predicted:
            px, py = LOCATIONS[predicted]
            ax.annotate("", xy=(px, py), xytext=(jx, jy),
                        arrowprops=dict(arrowstyle="->", color="#f77f00", lw=1.3))
    for loc_id, (lx, ly) in LOCATIONS.items():
        ax.scatter(lx, ly, s=250, color="#2dc653", zorder=3,
                   edgecolors="white", lw=2)
        ax.annotate(LOCATION_NAMES[loc_id], (lx, ly),
                    textcoords="offset points", xytext=(8, -14),
                    fontsize=7, color="#333")
    legend = [
        mpatches.Patch(color="#4361ee", label="Predicted positions (test samples)"),
        mpatches.Patch(color="#2dc653", label="Reference points (ground truth)"),
        mpatches.Patch(color="#e63946", label="Access Points (APs)"),
        mpatches.Patch(color="#f77f00", label="Misclassification"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "floor_map.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.close()

def plot_model_comparison(results):
    names = list(results.keys())
    accs  = [r["accuracy"]*100 for r in results.values()]
    mdes  = [r["mde"] for r in results.values()]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    bars1 = ax1.bar(names, accs, color=["#4361ee","#2dc653","#e63946"], edgecolor="white", lw=1.5)
    ax1.set_ylabel("Accuracy (%)"); ax1.set_title("Model Accuracy Comparison")
    ax1.set_ylim(0, 105)
    for bar, val in zip(bars1, accs):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    bars2 = ax2.bar(names, mdes, color=["#4361ee","#2dc653","#e63946"], edgecolor="white", lw=1.5)
    ax2.set_ylabel("Mean Distance Error (m)"); ax2.set_title("Localization Error (lower = better)")
    for bar, val in zip(bars2, mdes):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02,
                 f"{val:.2f}m", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[SAVED] {path}")
    plt.close()
