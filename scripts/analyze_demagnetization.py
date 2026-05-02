"""Analyze demagnetization fault injection results.

Usage:
  python scripts/analyze_demagnetization.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Font setup — use Noto Sans CJK (system TTC) for Korean, fallback to English
# ---------------------------------------------------------------------------
from matplotlib.font_manager import FontProperties

_KO_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _KO_FONT_PATH.exists():
    KO_FONT = FontProperties(fname=str(_KO_FONT_PATH))
    USE_KO = True
else:
    KO_FONT = FontProperties()
    USE_KO = False

matplotlib.rcParams["axes.unicode_minus"] = False


def ko(korean: str, english: str) -> str:
    """Return Korean string if font available, else English."""
    return korean if USE_KO else english

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("results/demagnetization_rr_calf")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ["PD", "Native", "Coupled"]
DEMAG_FACTORS = [1.0, 0.8, 0.6, 0.4]
TASK_COLORS = {"PD": "#1f77b4", "Native": "#ff7f0e", "Coupled": "#2ca02c"}


def load_data(task: str, demag: float) -> dict | None:
    path = DATA_DIR / f"{task}_demag_{demag}.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def load_summary() -> list[dict]:
    path = DATA_DIR / "summary.json"
    with open(path) as f:
        return json.load(f)


# ===== Figure 1: Current time series (3x4 grid) =====
def fig1_current():
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    fig.suptitle(ko("Figure 1: 평균 |전류| 시계열", "Figure 1: Mean |Current| Time Series"),
                 fontsize=14, fontproperties=KO_FONT)

    for row, task in enumerate(TASKS):
        for col, demag in enumerate(DEMAG_FACTORS):
            ax = axes[row, col]
            data = load_data(task, demag)

            if task == "PD" or data is None:
                ax.text(0.5, 0.5, "N/A" if task == "PD" else "No data",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=12, color="gray")
            else:
                act = data["act"]  # (steps, nu)
                mean_abs = np.mean(np.abs(act), axis=-1)  # (steps,)
                ax.plot(mean_abs, linewidth=0.5, color=TASK_COLORS[task])
                ax.set_ylabel("|I| [A]") if col == 0 else None

            if row == 0:
                ax.set_title(f"demag={demag}")
            if col == 0:
                ax.annotate(task, xy=(-0.35, 0.5), xycoords="axes fraction",
                            fontsize=12, fontweight="bold", rotation=90, va="center")
            if row == 2:
                ax.set_xlabel("Step")

    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig1_current.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig1_current.png'}")
    plt.close(fig)


# ===== Figure 2: Torque time series (3x4 grid) =====
def fig2_torque():
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    fig.suptitle(ko("Figure 2: 평균 |토크| 시계열", "Figure 2: Mean |Torque| Time Series"),
                 fontsize=14, fontproperties=KO_FONT)

    for row, task in enumerate(TASKS):
        for col, demag in enumerate(DEMAG_FACTORS):
            ax = axes[row, col]
            data = load_data(task, demag)

            if data is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=12, color="gray")
            else:
                # Skip floating base 6 DOFs
                torque = data["qfrc_actuator"][:, 6:]  # (steps, 12)
                mean_abs = np.mean(np.abs(torque), axis=-1)
                ax.plot(mean_abs, linewidth=0.5, color=TASK_COLORS[task])
                ax.set_ylabel("|τ| [N·m]") if col == 0 else None

            if row == 0:
                ax.set_title(f"demag={demag}")
            if col == 0:
                ax.annotate(task, xy=(-0.35, 0.5), xycoords="axes fraction",
                            fontsize=12, fontweight="bold", rotation=90, va="center")
            if row == 2:
                ax.set_xlabel("Step")

    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig2_torque.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig2_torque.png'}")
    plt.close(fig)


# ===== Figure 3: Performance degradation (bar chart) =====
def fig3_performance():
    summary = load_summary()

    metrics = {
        "episode_return": ko("누적 보상", "Episode Return"),
        "vel_error_rms": ko("속도 추종 오차 (RMS)", "Velocity Error (RMS)"),
        "mean_survival_steps": ko("생존 스텝 수", "Survival Steps"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(ko("Figure 3: 성능 저하 요약 (demag=1.0 대비 정규화)",
                     "Figure 3: Performance Degradation (normalized to demag=1.0)"),
                 fontsize=14, fontproperties=KO_FONT)

    for ax_idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[ax_idx]
        x = np.arange(len(DEMAG_FACTORS))
        width = 0.25

        for task_idx, task in enumerate(TASKS):
            vals = []
            for demag in DEMAG_FACTORS:
                entry = next((s for s in summary
                              if s["task"] == task and s["demag_factor"] == demag), None)
                vals.append(entry[metric] if entry else 0.0)

            # Normalize to demag=1.0
            baseline = vals[0] if vals[0] != 0 else 1.0
            if metric == "vel_error_rms":
                # For error: lower is better, so invert normalization
                normalized = [baseline / v if v != 0 else 0 for v in vals]
            else:
                normalized = [v / baseline for v in vals]

            ax.bar(x + task_idx * width, normalized, width,
                   label=task, color=TASK_COLORS[task])

        ax.set_xlabel("Demag factor")
        ax.set_ylabel(ko("정규화 비율", "Normalized Ratio"), fontproperties=KO_FONT)
        ax.set_title(label, fontproperties=KO_FONT)
        ax.set_xticks(x + width)
        ax.set_xticklabels([str(d) for d in DEMAG_FACTORS])
        ax.legend()
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig3_performance.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig3_performance.png'}")
    plt.close(fig)


# ===== Figure 4: Current vs Torque scatter =====
def fig4_current_torque():
    summary = load_summary()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(ko("Figure 4: 전류-토크 관계 (감자 수준별)",
                     "Figure 4: Current-Torque Relationship (by demag level)"),
                 fontsize=14, fontproperties=KO_FONT)

    for task in ["Native", "Coupled"]:
        currents, torques, labels = [], [], []
        for demag in DEMAG_FACTORS:
            entry = next((s for s in summary
                          if s["task"] == task and s["demag_factor"] == demag), None)
            if entry:
                currents.append(entry["mean_abs_current"])
                torques.append(entry["mean_abs_torque"])
                labels.append(f"{demag}")

        ax.plot(currents, torques, "o-", label=task, color=TASK_COLORS[task],
                markersize=8, linewidth=2)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (currents[i], torques[i]),
                        textcoords="offset points", xytext=(8, 5), fontsize=9)

    ax.set_xlabel(ko("평균 |전류| [A]", "Mean |Current| [A]"), fontproperties=KO_FONT)
    ax.set_ylabel(ko("평균 |토크| [N·m]", "Mean |Torque| [N·m]"), fontproperties=KO_FONT)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_current_torque.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig4_current_torque.png'}")
    plt.close(fig)


# ===== Figure 5: Native vs Coupled difference =====
def fig5_native_vs_coupled():
    summary = load_summary()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(ko("Figure 5: Native vs Coupled 차이 (감자 수준별)",
                     "Figure 5: Native vs Coupled Difference (by demag level)"),
                 fontsize=14, fontproperties=KO_FONT)

    metrics = {
        "episode_return": ko("누적 보상", "Episode Return"),
        "vel_error_rms": ko("속도 추종 오차", "Velocity Error"),
        "mean_abs_torque": ko("평균 |토크|", "Mean |Torque|"),
    }

    for ax_idx, (metric, label) in enumerate(metrics.items()):
        ax = axes[ax_idx]
        diffs = []
        for demag in DEMAG_FACTORS:
            native = next((s for s in summary
                           if s["task"] == "Native" and s["demag_factor"] == demag), None)
            coupled = next((s for s in summary
                            if s["task"] == "Coupled" and s["demag_factor"] == demag), None)

            if native and coupled and coupled[metric] != 0:
                diff = abs(native[metric] - coupled[metric]) / abs(coupled[metric])
            else:
                diff = 0.0
            diffs.append(diff)

        ax.bar(range(len(DEMAG_FACTORS)), diffs, color="#9467bd")
        ax.set_xlabel("Demag factor")
        ax.set_ylabel("|Native - Coupled| / |Coupled|")
        ax.set_title(label, fontproperties=KO_FONT)
        ax.set_xticks(range(len(DEMAG_FACTORS)))
        ax.set_xticklabels([str(d) for d in DEMAG_FACTORS])

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig5_native_vs_coupled.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig5_native_vs_coupled.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Generating figures...")
    fig1_current()
    fig2_torque()
    fig3_performance()
    fig4_current_torque()
    fig5_native_vs_coupled()
    print(f"\nAll figures saved to: {FIG_DIR}")

    # Print summary table
    summary = load_summary()
    print(f"\n{'Task':<10} {'Demag':<8} {'Return':<10} {'VelErr':<10} "
          f"{'|I|':<8} {'|τ|':<8} {'Survival':<10}")
    print("-" * 64)
    for s in summary:
        print(f"{s['task']:<10} {s['demag_factor']:<8.1f} "
              f"{s['episode_return']:<10.1f} {s['vel_error_rms']:<10.4f} "
              f"{s['mean_abs_current']:<8.3f} {s['mean_abs_torque']:<8.3f} "
              f"{s['mean_survival_steps']:<10.0f}")


if __name__ == "__main__":
    main()
