"""Diagnostic: ratio = tau_actual / tau_cmd  vs  |qd_all|  on the demag'd calf joint.

Expected behaviour if plant Ke·gr is correctly wired into the filterexact dyntype:
    ratio = factor · (1 + (1−factor) · (Ke·gr/R) · ω / I_des)
So |qd_all| 증가 시 ratio 가 factor 위로 벌어지는 양의 slope 가 관측돼야 함.

If slope ≈ 0 over |qd_all| ∈ [0, 20] rad/s:
    plant Ke coupling 미반영 → dyntype 구현 점검 필요.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results/demag_rerun/data/methoda"
OUT = Path(__file__).resolve().parents[1] / "results/demag_rerun/plots"
OUT.mkdir(parents=True, exist_ok=True)

LEGS = ("FL", "FR", "RL", "RR")
FACTORS = (0.8, 0.6, 0.4)

# Motor parameters (from GO2_METHODA_*).
Kt = 0.128
Ke = 0.128
gr = 6.33
R = 0.3
Kt_nom_joint = Kt * gr   # 0.8102

# Thresholds for ratio validity — small tau_cmd → ratio = noise.
TAU_CMD_MIN = 2.0   # N·m; skip steps where |tau_cmd| below this


def case_path(leg: str, factor: float) -> Path:
    return ROOT / f"{leg}_{factor:.1f}.npz"


def theoretical_ratio(omega: np.ndarray, I_des: np.ndarray, factor: float) -> np.ndarray:
    """ratio = factor * (1 + (1−factor)·(Ke·gr/R)·ω / I_des)."""
    coef = (1.0 - factor) * (Ke * gr / R)
    safe_I = np.where(np.abs(I_des) > 1e-3, I_des, np.nan)
    return factor * (1.0 + coef * omega / safe_I)


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(slope), float(intercept)


def main():
    fig, axes = plt.subplots(
        len(FACTORS), len(LEGS),
        figsize=(4 * len(LEGS), 3 * len(FACTORS)),
        sharex=True, sharey=False,
    )

    slope_table: list[tuple[str, float, float, float, int]] = []

    for i, factor in enumerate(FACTORS):
        for j, leg in enumerate(LEGS):
            ax = axes[i, j]
            path = case_path(leg, factor)
            if not path.exists():
                ax.set_title(f"{leg}×{factor} MISSING")
                continue

            d = np.load(path, allow_pickle=True)
            meta = json.loads(str(d["meta"]))
            joint_names = meta["q_all_joint_names"]
            col = joint_names.index(f"{leg}_calf_joint")

            tau_cmd    = d["tau_cmd"][:, col]
            tau_actual = d["tau_actual"][:, col]
            qd         = d["qd_all"][:, col]
            I_des      = d["I_des"][:, col]

            # Drop first 100 steps (transient) and small-load steps.
            mask = (np.abs(tau_cmd) > TAU_CMD_MIN)
            mask[:100] = False

            ratio = np.full_like(tau_cmd, np.nan, dtype=np.float64)
            ratio[mask] = tau_actual[mask] / tau_cmd[mask]
            omega = np.abs(qd)

            xv = omega[mask]
            yv = ratio[mask]
            if xv.size < 20:
                ax.set_title(f"{leg}×{factor} (N={xv.size} too few)")
                continue

            ax.scatter(xv, yv, s=4, alpha=0.35, color="tab:blue",
                       label=f"data N={xv.size}")
            ax.axhline(factor, ls="--", color="tab:red", lw=1.0,
                       label=f"factor = {factor}")

            # Theoretical curve (non-monotone in ω/I_des — draw sorted mean).
            x_sort = np.sort(xv)
            # Use median |I_des| at each omega bin for a clean analytic line.
            I_med = np.median(np.abs(I_des[mask]))
            if I_med > 0.1:
                y_theory = factor * (1.0 + (1.0 - factor) * (Ke * gr / R) *
                                      x_sort / I_med)
                ax.plot(x_sort, y_theory, color="tab:green", lw=1.5,
                        label=f"theory (|I_des|≈{I_med:.1f}A)")

            slope, intercept = linear_fit(xv, yv)
            slope_table.append((f"{leg}×{factor}", slope, intercept, float(np.mean(yv)), int(xv.size)))

            ax.set_title(f"{leg}×{factor} | slope={slope:+.4f}, "
                         f"mean={np.mean(yv):.3f}")
            if i == len(FACTORS) - 1:
                ax.set_xlabel("|qd_all| [rad/s]")
            if j == 0:
                ax.set_ylabel(r"$\tau_{actual}/\tau_{cmd}$")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")

    fig.suptitle(r"Demag diagnostic: $\tau_{actual}/\tau_{cmd}$ vs $|\dot q|$ "
                 "on demag'd calf (slope $>0$ ⇒ plant Ke coupling works)",
                 fontsize=12)
    fig.tight_layout()
    out_path = OUT / "diag_ratio_vs_qd.png"
    fig.savefig(out_path, dpi=130)
    print(f"[OK] saved {out_path}")

    print("\n=== Slope summary ===")
    print(f"{'case':<10}  {'slope':>10}  {'intercept':>10}  "
          f"{'mean':>8}  {'N':>6}")
    for name, s, ic, mn, n in slope_table:
        print(f"{name:<10}  {s:>+10.5f}  {ic:>+10.4f}  {mn:>8.3f}  {n:>6d}")

    # Slope sign check by factor (averaged across 4 legs).
    print("\n=== Mean slope per factor (across 4 legs) ===")
    for factor in FACTORS:
        rows = [s for name, s, *_ in slope_table if name.endswith(f"×{factor}")]
        if rows:
            print(f"  factor {factor}: mean slope = {np.mean(rows):+.5f} "
                  f"rad⁻¹·s  (n={len(rows)})")


if __name__ == "__main__":
    main()
