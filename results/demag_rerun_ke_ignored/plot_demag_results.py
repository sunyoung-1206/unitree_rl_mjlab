#!/usr/bin/env python
"""Plot demag re-run results (v3: per-joint array schema).

Input  : results/demag_rerun/data/{pd,methoda}/*.npz
Output : results/demag_rerun/plots/
         ├── methoda_FL_timeseries.png
         ├── methoda_FR_timeseries.png
         ├── methoda_RL_timeseries.png
         ├── methoda_RR_timeseries.png
         ├── methoda_summary_grid.png
         └── policy_baseline_compare.png

Schema assumptions (new format):
  d["tau_cmd"]/["tau_actual"]/["I_cmd"]/["I_actual"]/["q_all"]/["qd_all"]
    shape (N, 12), column order = meta["q_all_joint_names"].
  d["base_pos"] (N,3)  d["base_quat"] (N,4)  d["base_rpy"] (N,3)
  d["base_lin_vel"] (N,3, body)  d["base_ang_vel"] (N,3, body)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(exist_ok=True)

LEGS = ("FL", "FR", "RL", "RR")
FACTORS = (1.0, 0.8, 0.6, 0.4)

FACTOR_STYLE = {
    1.0: {"color": "#000000", "lw": 2.0, "label": "healthy"},
    0.8: {"color": "#1f77b4", "lw": 1.2, "label": "×0.8"},
    0.6: {"color": "#ff7f0e", "lw": 1.2, "label": "×0.6"},
    0.4: {"color": "#d62728", "lw": 1.2, "label": "×0.4"},
}

EFFORT_LIMIT = 45.0
KT_NOMINAL = 0.128 * 6.33
I_MAX = EFFORT_LIMIT / KT_NOMINAL  # ≈ 55.5 A
SAT_THRESHOLD = 0.95 * I_MAX


# ─── Data loading ─────────────────────────────────────────────────────
def load_npz(path: Path) -> dict | None:
    if not path.exists():
        return None
    raw = np.load(path, allow_pickle=True)
    out = {k: raw[k] for k in raw.files if k != "meta"}
    out["meta"] = json.loads(str(raw["meta"]))
    dt = out["meta"]["dt"]
    out["time"] = np.arange(out["meta"]["num_steps"]) * dt
    return out


def col_of(d: dict, joint_suffix: str) -> int:
    """Return column index of a joint (e.g. 'FL_calf') in per-joint arrays."""
    names = d["meta"]["q_all_joint_names"]
    # Accept both 'FL_calf' and 'FL_calf_joint'.
    candidates = {joint_suffix, f"{joint_suffix}_joint"}
    for i, n in enumerate(names):
        if n in candidates:
            return i
    raise KeyError(f"{joint_suffix} not in {names}")


def calf_col(d: dict, leg: str) -> int:
    return col_of(d, f"{leg}_calf")


def load_methoda_matrix() -> dict:
    healthy = load_npz(DATA_DIR / "methoda" / "healthy.npz")
    matrix: dict = {leg: {} for leg in LEGS}
    for leg in LEGS:
        if healthy is not None:
            matrix[leg][1.0] = healthy
        for f in (0.8, 0.6, 0.4):
            d = load_npz(DATA_DIR / "methoda" / f"{leg}_{f:.1f}.npz")
            if d is not None:
                matrix[leg][f] = d
    return matrix


# ─── Helpers ──────────────────────────────────────────────────────────
def _shade_mask(ax, t, mask, color="#d62728", alpha=0.15):
    if not mask.any():
        return
    edges = np.diff(mask.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate([[0], starts])
    if mask[-1]:
        ends = np.concatenate([ends, [len(mask)]])
    for s, e in zip(starts, ends):
        ax.axvspan(t[s], t[min(e, len(t) - 1)],
                   color=color, alpha=alpha, linewidth=0)


def _leg_column_title(ax, leg: str, demag_leg: str) -> None:
    if leg == demag_leg:
        ax.set_title(f"{leg}_calf  ← demag",
                     backgroundcolor="#fff3b0", weight="bold")
    else:
        ax.set_title(f"{leg}_calf")


def _common_dual_legend(fig):
    color_handles = [
        Line2D([0], [0], color=FACTOR_STYLE[f]["color"],
               lw=FACTOR_STYLE[f]["lw"], label=FACTOR_STYLE[f]["label"])
        for f in FACTORS
    ]
    ls_handles = [
        Line2D([0], [0], color="0.3", lw=1.6, linestyle="--", label="τ_cmd"),
        Line2D([0], [0], color="0.3", lw=1.6, linestyle="-",  label="τ_actual"),
    ]
    fig.legend(handles=color_handles + ls_handles,
               loc="upper right", bbox_to_anchor=(0.995, 0.985),
               ncol=1, fontsize=8, framealpha=0.95)


# ─── Figure 1-4: per-leg 3×4 timeseries ──────────────────────────────
def plot_leg_timeseries(demag_leg: str, matrix: dict) -> None:
    cases = matrix[demag_leg]
    if not cases:
        print(f"[skip] no data for leg={demag_leg}")
        return

    fig, axes = plt.subplots(3, 4, figsize=(17, 9),
                             sharex=True, squeeze=False)

    for col, leg in enumerate(LEGS):
        ax_tau, ax_i, ax_q = axes[0, col], axes[1, col], axes[2, col]
        _leg_column_title(ax_tau, leg, demag_leg)

        for f in FACTORS:
            d = cases.get(f)
            if d is None:
                continue
            t = d["time"]
            style = FACTOR_STYLE[f]
            c = style["color"]; lw = style["lw"]
            idx = calf_col(d, leg)

            tau_cmd = d["tau_cmd"][:, idx]
            tau_act = d["tau_actual"][:, idx]
            I_cmd   = d["I_cmd"][:, idx]
            I_act   = d["I_actual"][:, idx]
            q       = d["q_all"][:, idx]

            ax_tau.plot(t, tau_cmd, color=c, lw=lw * 0.85,
                        linestyle="--", alpha=0.8)
            ax_tau.plot(t, tau_act, color=c, lw=lw * 1.15,
                        linestyle="-",  alpha=0.9)

            ax_i.plot(t, I_cmd, color=c, lw=lw, alpha=0.9)
            sat_mask = np.abs(I_act) >= SAT_THRESHOLD
            if sat_mask.any():
                _shade_mask(ax_i, t, sat_mask)

            ax_q.plot(t, q, color=c, lw=lw, alpha=0.9)

        for ax in (ax_tau, ax_i, ax_q):
            ax.grid(True, alpha=0.3)
        ax_q.set_xlabel("time [s]  (control dt = 20 ms)")

        if col == 0:
            ax_tau.set_ylabel("τ  [N·m]")
            ax_i.set_ylabel("I  [A]")
            ax_q.set_ylabel("calf q  [rad]")
        if col == 3:
            ax_i.text(0.98, 0.04, f"I_max ≈ {I_MAX:.1f} A",
                      transform=ax_i.transAxes,
                      ha="right", va="bottom", fontsize=8, alpha=0.7,
                      bbox=dict(boxstyle="round,pad=0.2",
                                fc="white", ec="0.8", alpha=0.8))

    fig.suptitle(f"MethodA — {demag_leg}_calf demag sweep  "
                 f"(seed=42, 20 s, single-env)",
                 fontsize=12)
    _common_dual_legend(fig)
    fig.tight_layout(rect=[0, 0, 0.90, 0.96])
    out = PLOT_DIR / f"methoda_{demag_leg}_timeseries.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[OK] {out}")


# ─── Figure 5: 4×3 summary grid ──────────────────────────────────────
def plot_summary_grid(matrix: dict) -> None:
    factors = (0.8, 0.6, 0.4)
    fig, axes = plt.subplots(len(LEGS), len(factors),
                             figsize=(16, 10), sharex=True, squeeze=False)
    for r, leg in enumerate(LEGS):
        healthy = matrix[leg].get(1.0)
        for c, f in enumerate(factors):
            ax = axes[r, c]
            d = matrix[leg].get(f)
            if d is None:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="0.6")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            t = d["time"]
            style = FACTOR_STYLE[f]
            col = style["color"]
            idx = calf_col(d, leg)

            if healthy is not None:
                h_idx = calf_col(healthy, leg)
                ax.plot(healthy["time"],
                        healthy["tau_actual"][:, h_idx],
                        color="0.6", lw=1.0, linestyle=":", alpha=0.6)
            tau_cmd = d["tau_cmd"][:, idx]
            tau_act = d["tau_actual"][:, idx]
            ax.plot(t, tau_cmd, color=col, lw=1.0, linestyle="--", alpha=0.85)
            ax.plot(t, tau_act, color=col, lw=1.4, linestyle="-",  alpha=0.9)

            tc = float(np.mean(np.abs(tau_cmd[-200:])))
            ta = float(np.mean(np.abs(tau_act[-200:])))
            ratio = ta / max(tc, 1e-6)
            ax.set_title(f"{leg} ×{f:.1f}  ratio={ratio:.2f}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if r == len(LEGS) - 1:
                ax.set_xlabel("time [s]")
            if c == 0:
                ax.set_ylabel(f"{leg}\nτ [N·m]")

    handles = [
        Line2D([0], [0], color="0.6", lw=1.0, linestyle=":",
               label="healthy τ_actual"),
        Line2D([0], [0], color="0.3", lw=1.0, linestyle="--",
               label="demag τ_cmd"),
        Line2D([0], [0], color="0.3", lw=1.4, linestyle="-",
               label="demag τ_actual"),
    ]
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.995, 0.985), fontsize=9, framealpha=0.95)
    fig.suptitle("MethodA demag summary grid — "
                 "τ_cmd (dashed) vs τ_actual (solid) on demagnetized calf",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 0.92, 0.96])
    out = PLOT_DIR / "methoda_summary_grid.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[OK] {out}")


# ─── Figure 6: PD vs MethodA baseline ────────────────────────────────
def plot_policy_baseline_compare() -> None:
    pd = load_npz(DATA_DIR / "pd" / "nominal.npz")
    ma = load_npz(DATA_DIR / "methoda" / "healthy.npz")
    if pd is None or ma is None:
        print("[skip] baseline compare — missing pd or methoda healthy")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), squeeze=False)
    for d, col, label in ((pd, "tab:blue", "PD"),
                          (ma, "tab:red", "MethodA")):
        t = d["time"]
        calf_idx = [calf_col(d, l) for l in LEGS]
        mean_tau = np.mean(np.abs(d["tau_actual"][:, calf_idx]), axis=1)
        axes[0, 0].plot(t, mean_tau, color=col, lw=1.5, label=label)
        axes[0, 1].plot(t, d["base_pos"][:, 2], color=col, lw=1.5, label=label)
        axes[0, 2].plot(t, d["base_lin_vel"][:, 0], color=col, lw=1.5, label=label)

    axes[0, 2].plot(ma["time"], ma["cmd_vel"][:, 0],
                    color="k", lw=0.8, linestyle=":", label="cmd")

    axes[0, 0].set_title("mean |τ_actual| over 4 calves  [N·m]")
    axes[0, 1].set_title("base z  [m]")
    axes[0, 2].set_title("base vx  [m/s]")
    for ax in axes[0]:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]")
        ax.legend(fontsize=9)

    fig.suptitle("PD baseline vs MethodA healthy", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = PLOT_DIR / "policy_baseline_compare.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[OK] {out}")


# ─── Validation summary printout ─────────────────────────────────────
def print_validation_summary(matrix: dict) -> None:
    print("\n── Validation summary (last 4 s mean |τ| ratio) ──")
    for leg in LEGS:
        for f in (0.8, 0.6, 0.4):
            d = matrix[leg].get(f)
            if d is None:
                continue
            idx = calf_col(d, leg)
            tc = float(np.mean(np.abs(d["tau_cmd"][-200:, idx])))
            ta = float(np.mean(np.abs(d["tau_actual"][-200:, idx])))
            ratio = ta / max(tc, 1e-6)
            tag = ("[OK]  " if f - 0.05 <= ratio <= f + 0.05
                   else "[WARN]")
            print(f"  {tag} {leg}×{f:.1f}: measured={ratio:.3f} "
                  f"(expected {f:.2f} ±0.05)")


def main():
    matrix = load_methoda_matrix()
    for leg in LEGS:
        plot_leg_timeseries(leg, matrix)
    plot_summary_grid(matrix)
    plot_policy_baseline_compare()
    print_validation_summary(matrix)


if __name__ == "__main__":
    main()
