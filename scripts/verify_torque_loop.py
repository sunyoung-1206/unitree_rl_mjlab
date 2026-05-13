"""Verify torque-tracking integral loop: alpha=1.0 vs alpha=0.5.

Standalone simulation that mirrors the algorithm implemented in
NativeElectricActuator.compute() (recompute_pd branch). Avoids full mjlab
boot-up while still exercising the closed-loop dynamics:

  Driver (5 ms): PD -> tau_des -> integral correction -> I_des
  Motor  (0.1 ms): filterexact ZOH on I (time const tau_e = L/R)
  Plant  (1-DOF): rotational inertia + damping + constant external torque

A constant external torque (gravity-like) is applied so the steady state
demands a non-zero tau_actual. With the integral loop ON the integrator
must converge to a value that exactly cancels Kt_real != Kt_nom.

Compared cases:
  off / alpha=1.0   (baseline)
  off / alpha=0.5   (regression: shows tau_actual = alpha * tau_cmd)
  ON  / alpha=1.0   (regression: integral stays near 0)
  ON  / alpha=0.5   (the demag fix: integral grows, tau_actual tracks tau_cmd)

Reports:
  - settling time: first t with |tau_actual - tau_cmd| < 5% of |tau_cmd|.max
  - steady-state integral value vs theoretical
        integral_theo = tau_cmd_ss * (1/Kt_real_gr - 1/Kt_nom_gr) / Ki

Usage:
  python scripts/verify_torque_loop.py --Ki 50 --integral-max 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Go2 hip parameters (match go2_constants.py: GO2_COUPLED_ELECTRIC_HIP)
# ---------------------------------------------------------------------------
KT_NOM = 0.128
KE_NOM = 0.128
GR = 6.33
R_M = 0.3
L_M = 1e-4
EFFORT_LIMIT = 23.5  # N.m, joint side

# Driver / physics timing (matches Coupled cfg: substeps=200, pd_substeps=50, dt=0.1ms)
DRIVER_DT = 5e-3
PHYSICS_DT = 1e-4
SUBSTEPS = int(round(DRIVER_DT / PHYSICS_DT))  # 50

# 1-DOF mechanical plant (joint side)
J_INERTIA = 0.06
B_DAMPING = 0.05

# PD gains (match GO2_COUPLED_ELECTRIC_HIP)
KP = 20.0
KD = 1.0


def simulate(
    alpha: float,
    use_torque_loop: bool,
    Ki: float,
    integral_max: float,
    q_des: float,
    tau_ext: float,
    T_total: float,
):
    """Run one closed-loop scenario and return time-series logs + metrics."""
    Kt_nom_gr = KT_NOM * GR
    Kt_real_gr = alpha * Kt_nom_gr
    tau_e = L_M / R_M
    alpha_e = float(np.exp(-PHYSICS_DT / tau_e))  # filterexact ZOH coefficient
    I_max = EFFORT_LIMIT / Kt_nom_gr

    # Plant state
    q = 0.0
    qd = 0.0
    I = 0.0
    integral = 0.0

    log_keys = [
        "t", "tau_cmd", "tau_actual", "I_cmd", "I_actual",
        "integral", "q_err", "q_des", "q",
    ]
    log = {k: [] for k in log_keys}

    n_steps = int(round(T_total / DRIVER_DT))
    for k in range(n_steps):
        t_k = k * DRIVER_DT

        # ---- driver (5 ms) ------------------------------------------
        # PD -> tau_des, then DC motor saturation effectively bounded by EFFORT_LIMIT
        tau_cmd = KP * (q_des - q) - KD * qd
        tau_cmd = float(np.clip(tau_cmd, -EFFORT_LIMIT, EFFORT_LIMIT))

        # tau_actual_prev = post-gain torque from previous 5 ms window's last step.
        # In the actual code this is data.actuator_force; here we just read the
        # current motor torque (last physics step result).
        tau_actual_prev = Kt_real_gr * I

        if use_torque_loop:
            error = tau_cmd - tau_actual_prev
            integral = float(
                np.clip(integral + error * DRIVER_DT, -integral_max, integral_max)
            )
            I_cmd = tau_cmd / Kt_nom_gr + Ki * integral
        else:
            I_cmd = tau_cmd / Kt_nom_gr
        I_cmd = float(np.clip(I_cmd, -I_max, I_max))

        # log driver-rate quantities (taken at the start of this 5 ms window)
        log["t"].append(t_k)
        log["tau_cmd"].append(tau_cmd)
        log["I_cmd"].append(I_cmd)
        log["integral"].append(integral)
        log["q_des"].append(q_des)
        log["q"].append(q)
        log["q_err"].append(q_des - q)
        log["I_actual"].append(I)
        log["tau_actual"].append(Kt_real_gr * I)

        # ---- physics: SUBSTEPS x 0.1ms ------------------------------
        # filterexact ZOH on I (back-EMF compensated by virtual voltage controller)
        # Plant: J*qdd + b*qd + tau_ext = tau_actual  (tau_ext = constant external load)
        for _ in range(SUBSTEPS):
            I = I * alpha_e + I_cmd * (1.0 - alpha_e)
            tau_actual = Kt_real_gr * I
            qdd = (tau_actual - B_DAMPING * qd - tau_ext) / J_INERTIA
            qd += qdd * PHYSICS_DT
            q += qd * PHYSICS_DT

    log = {k: np.asarray(v, dtype=np.float64) for k, v in log.items()}

    # ---- metrics --------------------------------------------------------
    # Settling time: first t with |tau_actual - tau_cmd| < 5% of peak |tau_cmd|
    tau_cmd_peak = float(np.max(np.abs(log["tau_cmd"])))
    if tau_cmd_peak < 1e-9:
        settle_t = float("nan")
    else:
        thresh = 0.05 * tau_cmd_peak
        tracked = np.abs(log["tau_actual"] - log["tau_cmd"]) < thresh
        # require sustained tracking for >=10 driver steps (50ms)
        sustained_idx = -1
        run = 0
        for i, ok in enumerate(tracked):
            run = run + 1 if ok else 0
            if run >= 10:
                sustained_idx = i - 9
                break
        settle_t = log["t"][sustained_idx] if sustained_idx >= 0 else float("nan")

    # Steady-state integral
    # tail 100 ms average (or last 20 samples if T < 100 ms)
    n_tail = min(20, len(log["t"]))
    integral_obs = float(np.mean(log["integral"][-n_tail:]))
    tau_cmd_ss = float(np.mean(log["tau_cmd"][-n_tail:]))

    # Theoretical: tau_actual = tau_cmd in SS requires
    #   Kt_real_gr * I_cmd = tau_cmd
    #   I_cmd = tau_cmd / Kt_real_gr
    # And I_cmd = tau_cmd/Kt_nom_gr + Ki * integral
    # =>  integral_ss = tau_cmd_ss * (1/Kt_real_gr - 1/Kt_nom_gr) / Ki
    if Kt_real_gr > 0 and Ki > 0:
        integral_theo = tau_cmd_ss * (1.0 / Kt_real_gr - 1.0 / Kt_nom_gr) / Ki
    else:
        integral_theo = float("nan")

    return log, {
        "settle_t_ms": settle_t * 1e3,
        "integral_obs": integral_obs,
        "integral_theo": integral_theo,
        "tau_cmd_ss": tau_cmd_ss,
        "tau_actual_ss": float(np.mean(log["tau_actual"][-n_tail:])),
        "q_err_ss": float(np.mean(log["q_err"][-n_tail:])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ki", type=float, default=50.0)
    ap.add_argument("--integral-max", type=float, default=0.5,
                    help="anti-windup clamp [A.s]; explicit value (not the cfg fallback)")
    ap.add_argument("--q-des", type=float, default=0.5, help="step target [rad]")
    ap.add_argument("--tau-ext", type=float, default=4.0,
                    help="constant external torque (gravity-like) [N.m]")
    ap.add_argument("--T", type=float, default=2.0, help="sim duration [s]")
    ap.add_argument("--out", type=str,
                    default="results/torque_loop_verify/verify.png")
    args = ap.parse_args()

    cases = [
        ("off  alpha=1.0", 1.0, False),
        ("off  alpha=0.5", 0.5, False),
        ("ON   alpha=1.0", 1.0, True),
        ("ON   alpha=0.5", 0.5, True),
    ]
    colors = {
        "off  alpha=1.0": "tab:blue",
        "off  alpha=0.5": "tab:orange",
        "ON   alpha=1.0": "tab:green",
        "ON   alpha=0.5": "tab:red",
    }

    print(f"\n=== Torque-loop verification (Ki={args.Ki}, integral_max={args.integral_max}) ===")
    print(f"q_des={args.q_des} rad, tau_ext={args.tau_ext} N.m, T={args.T} s\n")
    print(f"{'case':18s} {'settle [ms]':>11s}  {'q_err_ss':>9s}  "
          f"{'tau_cmd_ss':>10s}  {'tau_act_ss':>10s}  "
          f"{'int_obs':>9s}  {'int_theo':>9s}  {'|err|':>8s}")
    print("-" * 105)

    results = []
    for label, alpha, on in cases:
        log, m = simulate(
            alpha=alpha,
            use_torque_loop=on,
            Ki=args.Ki,
            integral_max=args.integral_max,
            q_des=args.q_des,
            tau_ext=args.tau_ext,
            T_total=args.T,
        )
        err = abs(m["integral_obs"] - m["integral_theo"])
        print(f"{label:18s} {m['settle_t_ms']:11.1f}  "
              f"{m['q_err_ss']:9.4f}  "
              f"{m['tau_cmd_ss']:10.4f}  {m['tau_actual_ss']:10.4f}  "
              f"{m['integral_obs']:9.5f}  {m['integral_theo']:9.5f}  {err:8.5f}")
        results.append((label, log, m))

    # ---------------- plot ---------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    panels = [
        ("tau_cmd",   "tau_cmd [N.m]"),
        ("tau_actual","tau_actual [N.m]"),
        ("I_cmd",     "I_cmd [A]"),
        ("I_actual",  "I_actual [A]"),
        ("integral",  "integral [A.s]"),
        ("q_err",     "q_des - q [rad]"),
    ]
    for ax, (key, ylabel) in zip(axes.flat, panels):
        for label, log, _ in results:
            ax.plot(log["t"], log[key], label=label, lw=1.4,
                    color=colors[label])
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 1].set_xlabel("time [s]")
    axes[0, 0].legend(fontsize=8, loc="best")
    fig.suptitle(
        f"Torque-tracking integral loop verification | "
        f"Ki={args.Ki}, integral_max={args.integral_max}, "
        f"tau_ext={args.tau_ext} N.m",
        fontsize=11,
    )
    fig.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
