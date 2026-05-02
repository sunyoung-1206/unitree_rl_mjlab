"""Compare Schur vs non-Schur filterexact on a single demag case.

Two diagnostics:
  (A) ratio = tau_actual/tau_cmd  vs  |qd|      — user's original metric.
       Caveat: sign of (ratio − factor) = sign(ω · I_des), so a walking
       gait mixes both phases and |ω|-axis averages them out. Flat slope
       here is INCONCLUSIVE by itself.

  (B) Δ_I = I_actual − I_des     vs  qd (signed) — cleaner metric.
       Steady-state prediction (if plant Ke is in the ODE):
           Δ_I = (1 − factor) · Ke_nom · gr / R · ω
       For factor=0.6 this is slope ≈ 1.08 A·s/rad, linear, sign-consistent.
       Flat slope here → plant Ke genuinely missing from integration.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "results"
NONSCHUR = ROOT / "demag_rerun_ke_ignored/data/methoda/FL_0.6.npz"
SCHUR    = ROOT / "demag_rerun_schur_test/data/methoda/FL_0.6.npz"
OUT = ROOT / "demag_rerun_schur_test/plots"
OUT.mkdir(parents=True, exist_ok=True)

LEG = "FL"
FACTOR = 0.6
Kt, Ke, gr, R = 0.128, 0.128, 6.33, 0.3
Kt_nom_joint = Kt * gr  # 0.8102
Ke_nom_joint = Ke * gr  # 0.8102

# Theoretical slopes for factor=0.6
#   Δ_I / ω            = (1 − factor)·Ke_nom·gr / R
#   ratio corr coeff   = factor · (1 − factor) · Ke_nom·gr / R / I_des
DI_SLOPE_THEORY = (1.0 - FACTOR) * Ke_nom_joint / R   # A / (rad/s)

TAU_CMD_MIN = 2.0   # N·m; gate for ratio stability


def load(path: Path):
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    col = meta["q_all_joint_names"].index(f"{LEG}_calf_joint")
    return {
        "tau_cmd":    d["tau_cmd"][:, col],
        "tau_actual": d["tau_actual"][:, col],
        "qd":         d["qd_all"][:, col],
        "I_des":      d["I_des"][:, col],
        "I_cmd":      d["I_cmd"][:, col],
        "I_actual":   d["I_actual"][:, col],
    }


def linfit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    s, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(s), float(c)


def panel_ratio(ax, data, title):
    tc, ta, qd = data["tau_cmd"], data["tau_actual"], data["qd"]
    mask = np.abs(tc) > TAU_CMD_MIN
    mask[:100] = False
    ratio = ta[mask] / tc[mask]
    w = np.abs(qd[mask])
    s, c = linfit(w, ratio)
    ax.scatter(w, ratio, s=4, alpha=0.3, color="tab:blue", label=f"N={w.size}")
    ax.axhline(FACTOR, ls="--", color="tab:red", lw=1.0, label=f"factor={FACTOR}")
    xlin = np.linspace(0, w.max(), 50)
    ax.plot(xlin, s * xlin + c, color="tab:orange", lw=1.5,
            label=f"fit slope={s:+.4f}, c={c:.3f}")
    ax.set_xlabel(r"$|\dot q|$ [rad/s]")
    ax.set_ylabel(r"$\tau_{actual}/\tau_{cmd}$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    ax.set_ylim(max(0.0, FACTOR - 0.4), min(1.2, FACTOR + 0.5))
    return s, c


def panel_di(ax, data, title):
    Id, Ia, qd = data["I_des"], data["I_actual"], data["qd"]
    mask = np.ones_like(qd, dtype=bool)
    mask[:100] = False
    dI = Ia[mask] - Id[mask]
    w  = qd[mask]
    s, c = linfit(w, dI)
    ax.scatter(w, dI, s=4, alpha=0.3, color="tab:blue", label=f"N={w.size}")
    ax.axhline(0, ls=":", color="gray", lw=0.8)
    xlin = np.linspace(w.min(), w.max(), 50)
    ax.plot(xlin, s * xlin + c, color="tab:orange", lw=1.5,
            label=f"fit slope={s:+.3f}, c={c:+.3f}")
    ax.plot(xlin, DI_SLOPE_THEORY * xlin, color="tab:green", lw=1.3, ls="--",
            label=f"theory slope={DI_SLOPE_THEORY:+.3f}")
    ax.set_xlabel(r"$\dot q$ [rad/s]")
    ax.set_ylabel(r"$I_{actual} - I_{des}$ [A]")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    return s, c


def verdict(di_slope_schur: float, ratio_intercept_schur: float):
    # Primary decisive test: Δ_I slope. theory = 1.08 A·s/rad for factor=0.6.
    # Threshold: at least 50% of theory = 0.54.
    if di_slope_schur > 0.5 * DI_SLOPE_THEORY:
        return ("A", "Schur mode activates plant Ke coupling — "
                     "non-Schur path is the bug.")
    if di_slope_schur < 0.2 * DI_SLOPE_THEORY:
        return ("B", "Schur mode also shows flat Δ_I — dyntype itself "
                     "is not integrating Ke. Deeper fork investigation needed.")
    return ("?", "Slope intermediate — inconclusive, inspect scatter.")


def main():
    if not SCHUR.exists():
        print(f"[ABORT] Schur test file missing: {SCHUR}")
        return
    if not NONSCHUR.exists():
        print(f"[ABORT] non-Schur baseline missing: {NONSCHUR}")
        return

    d_ns = load(NONSCHUR)
    d_sc = load(SCHUR)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    s_r_ns, c_r_ns = panel_ratio(axes[0, 0], d_ns,
        r"(A1) ratio vs $|\dot q|$ — non-Schur (method='A')")
    s_r_sc, c_r_sc = panel_ratio(axes[0, 1], d_sc,
        r"(A2) ratio vs $|\dot q|$ — Schur (method='A+')")
    s_i_ns, c_i_ns = panel_di(axes[1, 0], d_ns,
        r"(B1) $I_{actual}-I_{des}$ vs $\dot q$ — non-Schur")
    s_i_sc, c_i_sc = panel_di(axes[1, 1], d_sc,
        r"(B2) $I_{actual}-I_{des}$ vs $\dot q$ — Schur")

    fig.suptitle(f"Schur vs non-Schur diagnostic — MethodA {LEG}×{FACTOR} "
                 f"(theory Δ_I slope = {DI_SLOPE_THEORY:.3f} A·s/rad)",
                 fontsize=12)
    fig.tight_layout()
    out = OUT / f"schur_vs_nonschur_{LEG}_{FACTOR}.png"
    fig.savefig(out, dpi=130)
    print(f"[OK] saved {out}")

    print("\n=== slope summary ===")
    print(f"{'metric':<26} {'non-Schur':>14} {'Schur':>14} {'theory':>10}")
    print(f"{'ratio slope (A/A)':<26} {s_r_ns:>+14.5f} {s_r_sc:>+14.5f} "
          f"{'(mixed)':>10}")
    print(f"{'ratio intercept':<26} {c_r_ns:>+14.4f} {c_r_sc:>+14.4f} "
          f"{FACTOR:>10.2f}")
    print(f"{'Δ_I slope (A·s/rad)':<26} {s_i_ns:>+14.4f} {s_i_sc:>+14.4f} "
          f"{DI_SLOPE_THEORY:>+10.4f}")
    print(f"{'Δ_I intercept (A)':<26} {c_i_ns:>+14.4f} {c_i_sc:>+14.4f} "
          f"{'0.000':>10}")

    case, msg = verdict(s_i_sc, c_r_sc)
    print(f"\n=== VERDICT: Case {case} ===")
    print(msg)


if __name__ == "__main__":
    main()
