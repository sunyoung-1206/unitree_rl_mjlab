"""Phase 3 verification — post Ke-fix mjlab rollout.

Primary metric: Δ_I = I_actual - I_des regressed on ω = qd_all[:, calf_col].
Theory slope = (1 - factor) · Ke_nom · gr / R   (e.g., 1.08 A·s/rad @ factor=0.6).

Secondary plots (reporting): time series of ω/I_des/I_actual, ratio(t),
Δ_I vs ω scatter + regression, |ω| histogram.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / "results/demag_rerun_ke_fixed/data/methoda"
OUT  = Path(__file__).resolve().parents[1] / "results/demag_rerun_ke_fixed/plots"
OUT.mkdir(parents=True, exist_ok=True)

Kt, Ke, gr, R = 0.128, 0.128, 6.33, 0.3
Kt_gr, Ke_gr = Kt * gr, Ke * gr   # 0.8102

# Per-calf joint current spec limit (joint-space) — effort_limit / Kt_nom·gr.
I_LIMIT_CALF = 45.0 / Kt_gr    # ≈ 55.5 A


def load_case(path: Path, leg: str) -> dict:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    col = meta["q_all_joint_names"].index(f"{leg}_calf_joint")
    return {
        "meta": meta, "col": col,
        "tau_cmd":    d["tau_cmd"][:, col],
        "tau_actual": d["tau_actual"][:, col],
        "qd":         d["qd_all"][:, col],
        "I_des":      d["I_des"][:, col],
        "I_cmd":      d["I_cmd"][:, col],
        "I_actual":   d["I_actual"][:, col],
        "factor":     float(meta["demag_factor"]),
    }


def linfit(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    s, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(s), float(c)


def verdict_slope(slope: float, factor: float) -> tuple[str, str]:
    theory = (1.0 - factor) * Ke_gr / R
    if factor >= 0.999:   # healthy
        ok = abs(slope) < 0.05
        return ("PASS" if ok else "FAIL",
                f"healthy | slope={slope:+.4f} (expect ≈ 0, |·|<0.05) | theory={theory:.3f}")
    # demag
    if slope < 0:
        return ("FAIL", f"slope NEGATIVE, sign wrong (factor={factor}, theory={theory:+.3f})")
    err_pct = abs(slope - theory) / theory * 100.0
    if 0.8 * theory <= slope <= 1.2 * theory:
        return ("PASS", f"slope={slope:+.4f} within ±20% of theory={theory:+.4f} ({err_pct:.1f}%)")
    if 0.5 * theory <= slope <= 2.0 * theory:
        return ("PARTIAL", f"slope={slope:+.4f} within ±50% of theory={theory:+.4f} ({err_pct:.1f}%)")
    return ("FAIL", f"slope={slope:+.4f} vs theory={theory:+.4f} ({err_pct:.1f}% off)")


def plot_case(case: dict, leg: str, ax_row):
    meta = case["meta"]
    N = meta["num_steps"]
    dt = meta["dt"]
    t = np.arange(N) * dt
    factor = case["factor"]

    qd = case["qd"]
    I_des = case["I_des"]
    I_act = case["I_actual"]
    tau_cmd = case["tau_cmd"]
    tau_act = case["tau_actual"]

    # Panel 1: ω, I_des, I_actual time series
    ax = ax_row[0]
    ax.plot(t, qd, color="tab:green", lw=0.9, label="ω [rad/s]", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(t, I_des, color="tab:orange", lw=0.9, ls="--", label="I_des [A]")
    ax2.plot(t, I_act, color="tab:red",    lw=0.9, label="I_actual [A]")
    ax.set_xlabel("time [s]"); ax.set_ylabel("ω [rad/s]", color="tab:green")
    ax2.set_ylabel("current [A]")
    ax.set_title(f"{leg}×{factor} | time series")
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=7)
    ax2.legend(loc="upper right", fontsize=7)

    # Panel 2: ratio(t)
    ax = ax_row[1]
    mask = np.abs(tau_cmd) > 2.0
    ratio = np.full_like(tau_cmd, np.nan, dtype=np.float64)
    ratio[mask] = tau_act[mask] / tau_cmd[mask]
    ax.plot(t, ratio, color="tab:blue", lw=0.6)
    ax.axhline(factor, ls="--", color="tab:red", lw=1.0, label=f"factor = {factor}")
    ax.axhline(1.0, ls=":", color="gray", lw=0.8)
    ax.set_xlabel("time [s]"); ax.set_ylabel(r"$\tau_{actual}/\tau_{cmd}$")
    ax.set_title(f"{leg}×{factor} | ratio(t)")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Panel 3: Δ_I vs ω scatter + regression
    ax = ax_row[2]
    # skip first 100 steps (transient)
    dI = I_act[100:] - I_des[100:]
    w  = qd[100:]
    slope, intercept = linfit(w, dI)
    theory_slope = (1.0 - factor) * Ke_gr / R
    ax.scatter(w, dI, s=3, alpha=0.3, color="tab:blue", label=f"N={w.size}")
    xlin = np.linspace(w.min(), w.max(), 50)
    ax.plot(xlin, slope * xlin + intercept, color="tab:orange", lw=1.5,
            label=f"fit slope={slope:+.3f}, c={intercept:+.3f}")
    ax.plot(xlin, theory_slope * xlin, color="tab:green", lw=1.3, ls="--",
            label=f"theory slope={theory_slope:+.3f}")
    ax.axhline(0, ls=":", color="gray", lw=0.8)
    ax.set_xlabel(r"$\dot q$ [rad/s]"); ax.set_ylabel(r"$I_{actual} - I_{des}$ [A]")
    ax.set_title(f"{leg}×{factor} | Δ_I vs ω")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    # Panel 4: |ω| histogram (trajectory coverage)
    ax = ax_row[3]
    absw = np.abs(qd[100:])
    frac_above5 = float(np.mean(absw > 5.0))
    ax.hist(absw, bins=40, color="tab:purple", alpha=0.7)
    ax.axvline(5.0, ls="--", color="tab:red", lw=1.0,
               label=f"|ω|>5 frac = {frac_above5*100:.1f}%")
    ax.set_xlabel(r"$|\dot q|$ [rad/s]"); ax.set_ylabel("count")
    ax.set_title(f"{leg}×{factor} | ω coverage")
    ax.grid(alpha=0.3); ax.legend(fontsize=7)

    return slope, intercept, float(np.mean(ratio[mask])), np.abs(I_act).max(), frac_above5


def main():
    # Two cases: healthy (factor=1.0) and FL_0.6 (demag).
    cases = [
        ("FL", ROOT / "healthy_vx1.5.npz"),
        ("FL", ROOT / "FL_0.6_vx1.5.npz"),
    ]

    fig, axes = plt.subplots(len(cases), 4, figsize=(18, 4 * len(cases)))
    if len(cases) == 1:
        axes = axes.reshape(1, -1)

    rows = []
    for i, (leg, path) in enumerate(cases):
        if not path.exists():
            print(f"[ABORT] missing {path}")
            return
        case = load_case(path, leg)
        slope, intercept, ratio_mean, I_max, frac_above5 = plot_case(case, leg, axes[i])
        status, msg = verdict_slope(slope, case["factor"])
        theory = (1.0 - case["factor"]) * Ke_gr / R
        rows.append({
            "leg": leg, "factor": case["factor"],
            "slope": slope, "theory_slope": theory, "intercept": intercept,
            "ratio_mean": ratio_mean, "I_max": I_max, "frac_above5": frac_above5,
            "status": status, "msg": msg,
        })

    fig.suptitle("Phase 3 Ke-fix verification — MethodA post-patch",
                 fontsize=12)
    fig.tight_layout()
    out = OUT / "phase3_ke_fix_verify.png"
    fig.savefig(out, dpi=120)
    print(f"[OK] saved {out}")

    # Table
    print("\n" + "=" * 90)
    print(f"{'case':<12} {'slope':>10} {'theory':>10} {'err%':>8} {'ratio':>8} "
          f"{'Imax':>8} {'Ilim%':>8} {'|ω|>5':>8} {'status':>8}")
    print("=" * 90)
    for r in rows:
        err_pct = (abs(r["slope"] - r["theory_slope"]) /
                   max(abs(r["theory_slope"]), 1e-6) * 100.0)
        print(f"{r['leg']+'×'+str(r['factor']):<12} "
              f"{r['slope']:>+10.4f} {r['theory_slope']:>+10.4f} "
              f"{err_pct:>7.1f}% {r['ratio_mean']:>8.3f} "
              f"{r['I_max']:>8.2f} {(r['I_max']/I_LIMIT_CALF*100):>7.1f}% "
              f"{r['frac_above5']*100:>7.1f}% {r['status']:>8}")
    print("=" * 90)

    print(f"\nGo2 calf I limit (joint-space) = {I_LIMIT_CALF:.1f} A "
          f"(= effort_limit 45 N·m / Kt·gr 0.8102)")

    overall = "PASS" if all(r["status"] == "PASS" for r in rows) else (
        "PARTIAL" if any(r["status"] == "PARTIAL" for r in rows) and
        not any(r["status"] == "FAIL" for r in rows) else "FAIL")
    print(f"\n=== OVERALL: {overall} ===")
    for r in rows:
        print(f"  {r['leg']}×{r['factor']}: {r['msg']}")


if __name__ == "__main__":
    main()
