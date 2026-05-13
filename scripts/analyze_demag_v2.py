"""Analyze demagnetization v2: RR_calf single-motor fault, 3 methods.

Usage:
  python scripts/analyze_demag_v2.py
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

# ── Font ───────────────────────────────────────────────────────────
_KO_FONT_PATH = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
KO_FONT = FontProperties(fname=str(_KO_FONT_PATH)) if _KO_FONT_PATH.exists() else FontProperties()
USE_KO = _KO_FONT_PATH.exists()
matplotlib.rcParams["axes.unicode_minus"] = False

def ko(kr, en):
    return kr if USE_KO else en

# ── Paths ──────────────────────────────────────────────────────────
DATA_DIR = Path("results/demagnetization_v2")
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["PD", "MethodA", "Aplus"]
DEMAGS = [1.0, 0.8, 0.6, 0.4]
COLORS = {"PD": "gray", "MethodA": "tab:blue", "Aplus": "tab:green"}
LSTYLES = {"PD": "--", "MethodA": "-", "Aplus": "-"}
MARKERS = {"PD": "s", "MethodA": "o", "Aplus": "D"}
LABELS = {"PD": "PD", "MethodA": "Method A (IE)", "Aplus": "Method A+ (FE)"}


def load_data(method, demag):
    p = DATA_DIR / f"{method}_demag_{demag}.npz"
    return dict(np.load(p)) if p.exists() else None

def load_summary():
    with open(DATA_DIR / "summary.json") as f:
        return json.load(f)


# ===== Figure 1: RR_calf current (4×4 grid) =====
def fig1():
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(ko("Figure 1: RR_calf 전류 시계열 (signed)", "Figure 1: RR_calf Current (signed)"),
                 fontsize=14, fontproperties=KO_FONT)
    for row, m in enumerate(METHODS):
        for col, d in enumerate(DEMAGS):
            ax = axes[row, col]
            data = load_data(m, d)
            if m == "PD" or data is None:
                ax.text(0.5, 0.5, "N/A" if m == "PD" else "No data",
                        ha="center", va="center", transform=ax.transAxes, color="gray")
            else:
                ax.plot(data["rr_calf_current"], lw=0.4, color=COLORS[m])
            if row == 0:
                ax.set_title(f"demag={d}")
            if col == 0:
                ax.annotate(LABELS[m], xy=(-0.4, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", rotation=90, va="center")
            if row == 3:
                ax.set_xlabel("Step")
    axes[0, 0].set_ylabel("I [A]")
    fig.tight_layout(rect=[0.05, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig1_rr_calf_current.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig1_rr_calf_current.png'}")
    plt.close(fig)


# ===== Figure 2: RR_calf torque (4×4 grid) =====
def fig2():
    fig, axes = plt.subplots(4, 4, figsize=(16, 12), sharex=True, sharey=True)
    fig.suptitle(ko("Figure 2: RR_calf 토크 시계열 (signed)", "Figure 2: RR_calf Torque (signed)"),
                 fontsize=14, fontproperties=KO_FONT)
    for row, m in enumerate(METHODS):
        for col, d in enumerate(DEMAGS):
            ax = axes[row, col]
            data = load_data(m, d)
            if data is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
            else:
                ax.plot(data["rr_calf_torque"], lw=0.4, color=COLORS[m])
            if row == 0:
                ax.set_title(f"demag={d}")
            if col == 0:
                ax.annotate(LABELS[m], xy=(-0.4, 0.5), xycoords="axes fraction",
                            fontsize=10, fontweight="bold", rotation=90, va="center")
            if row == 3:
                ax.set_xlabel("Step")
    axes[0, 0].set_ylabel("[N·m]")
    fig.tight_layout(rect=[0.05, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig2_rr_calf_torque.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig2_rr_calf_torque.png'}")
    plt.close(fig)


# ===== Figure 3: RR_calf (fault) vs RL_calf (normal) =====
def fig3():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey="row")
    fig.suptitle(ko("Figure 3: Fault(RR_calf) vs Normal(RL_calf) 전류 비교",
                     "Figure 3: Fault (RR_calf) vs Normal (RL_calf) Current"),
                 fontsize=14, fontproperties=KO_FONT)
    for col, d in enumerate(DEMAGS):
        for m in METHODS:
            if m == "PD":
                continue
            data = load_data(m, d)
            if data is None:
                continue
            mk = max(1, len(data["rr_calf_current"]) // 15)
            axes[0, col].plot(data["rr_calf_current"], lw=0.5, color=COLORS[m],
                              label=LABELS[m] if col == 0 else None)
            axes[1, col].plot(data["rl_calf_current"], lw=0.5, color=COLORS[m],
                              label=LABELS[m] if col == 0 else None)
        axes[0, col].set_title(f"demag={d}")
        if col == 0:
            axes[0, col].set_ylabel("RR_calf (fault) [A]")
            axes[1, col].set_ylabel("RL_calf (normal) [A]")
        axes[1, col].set_xlabel("Step")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig3_fault_vs_normal.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig3_fault_vs_normal.png'}")
    plt.close(fig)


# ===== Figure 4: Performance summary (line plot) =====
def fig4():
    summary = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(ko("Figure 4: 성능 저하 요약", "Figure 4: Performance Degradation"),
                 fontsize=14, fontproperties=KO_FONT)

    for m in METHODS:
        entries = [s for s in summary if s["task"] == m]
        dvals = [e["demag_factor"] for e in entries]
        returns = [e["episode_return"] for e in entries]
        survs = [e["mean_survival_steps"] for e in entries]

        axes[0].plot(dvals, returns, color=COLORS[m], ls=LSTYLES[m], marker=MARKERS[m],
                     ms=6, lw=2, label=LABELS[m])
        axes[1].plot(dvals, survs, color=COLORS[m], ls=LSTYLES[m], marker=MARKERS[m],
                     ms=6, lw=2, label=LABELS[m])

    axes[0].set_xlabel("Demag factor"); axes[0].set_ylabel("Episode Return")
    axes[0].set_title(ko("누적 보상", "Episode Return"), fontproperties=KO_FONT)
    axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()

    axes[1].set_xlabel("Demag factor"); axes[1].set_ylabel("Steps")
    axes[1].set_title(ko("생존 스텝 수", "Survival Steps"), fontproperties=KO_FONT)
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig4_performance.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig4_performance.png'}")
    plt.close(fig)


# ===== Figure 5: Method A vs A+ direct comparison =====
def fig5():
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey="row")
    fig.suptitle(ko("Figure 5: Method A vs A+ — RR_calf 전류 + 토크",
                     "Figure 5: Method A vs A+ — RR_calf Current + Torque"),
                 fontsize=14, fontproperties=KO_FONT)

    for col, d in enumerate(DEMAGS):
        for m, row_label in [("MethodA", "A"), ("Aplus", "A+")]:
            row = 0 if m == "MethodA" else 1
            data = load_data(m, d)
            if data is None:
                continue
            ax = axes[row, col]
            ax.plot(data["rr_calf_current"], lw=0.5, color="tab:orange", label="Current [A]")
            ax2 = ax.twinx()
            ax2.plot(data["rr_calf_torque"], lw=0.5, color="tab:purple", alpha=0.7, label="Torque [Nm]")
            if col == 0:
                ax.set_ylabel(f"{row_label}: Current [A]")
            if col == 3:
                ax2.set_ylabel("Torque [Nm]")
            if row == 0:
                ax.set_title(f"demag={d}")
            if row == 1:
                ax.set_xlabel("Step")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / "fig5_A_vs_Aplus.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig5_A_vs_Aplus.png'}")
    plt.close(fig)


# ===== Figure 6: Yaw bias =====
def fig6():
    summary = load_summary()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(ko("Figure 6: Yaw 편향 (단일 관절 비대칭 fault)",
                     "Figure 6: Yaw Bias (asymmetric single-joint fault)"),
                 fontsize=14, fontproperties=KO_FONT)

    # Left: yaw rate time series at demag=0.4
    ax = axes[0]
    for m in METHODS:
        data = load_data(m, 0.4)
        if data is None:
            continue
        ang_vel = data["base_ang_vel"]  # (steps, 3) or (steps,)
        if ang_vel.ndim == 2:
            yaw_rate = ang_vel[:, 2]
        else:
            yaw_rate = ang_vel
        ax.plot(yaw_rate, lw=0.5, color=COLORS[m], label=LABELS[m])
    ax.set_xlabel("Step"); ax.set_ylabel("Yaw rate [rad/s]")
    ax.set_title(ko("Yaw 각속도 (demag=0.4)", "Yaw Rate (demag=0.4)"), fontproperties=KO_FONT)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: mean |yaw rate| vs demag
    ax = axes[1]
    for m in METHODS:
        yaw_means = []
        for d in DEMAGS:
            data = load_data(m, d)
            if data is None:
                yaw_means.append(0)
                continue
            ang_vel = data["base_ang_vel"]
            yr = ang_vel[:, 2] if ang_vel.ndim == 2 else ang_vel
            yaw_means.append(float(np.mean(np.abs(yr))))
        ax.plot(DEMAGS, yaw_means, color=COLORS[m], ls=LSTYLES[m], marker=MARKERS[m],
                ms=6, lw=2, label=LABELS[m])
    ax.set_xlabel("Demag factor"); ax.set_ylabel("Mean |yaw rate| [rad/s]")
    ax.set_title(ko("평균 |yaw rate| vs 감자 수준", "Mean |Yaw Rate| vs Demag"),
                 fontproperties=KO_FONT)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / "fig6_yaw_bias.png", dpi=150)
    print(f"Saved: {FIG_DIR / 'fig6_yaw_bias.png'}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────
def main():
    print("Generating figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print(f"\nAll figures saved to: {FIG_DIR}")

    # Print summary
    summary = load_summary()
    print(f"\n{'Task':10s} {'dmg':>4s}  {'RR|I|':>7s}  {'RR|τ|':>7s}  {'RL|I|':>7s}  {'RL|τ|':>7s}  {'Return':>7s}  {'Surv':>5s}")
    print("-" * 65)
    for s in summary:
        print(f"{s['task']:10s} {s['demag_factor']:4.1f}  {s['rr_calf_mean_abs_I']:7.3f}  "
              f"{s['rr_calf_mean_abs_tau']:7.3f}  {s['rl_calf_mean_abs_I']:7.3f}  "
              f"{s['rl_calf_mean_abs_tau']:7.3f}  {s['episode_return']:7.1f}  "
              f"{s['mean_survival_steps']:5.0f}")


if __name__ == "__main__":
    main()
