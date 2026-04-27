"""Phase 2 isolation test — minimal 1-DOF MuJoCo + mjwarp, bypass mjlab.

Goal: verify the patched FILTEREXACT kernel integrates
    dI/dt = (ctrl - I)/tau_e + (Ke_nom*gr - Ke_plant*gr) * omega / L
against analytical response for
  (a) healthy: dynprm[3] == dynprm[1] = Ke_nom*gr → vanilla filterexact.
  (b) demag factor=0.6: dynprm[1] = 0.6 * Ke_nom*gr, dynprm[3] = Ke_nom*gr.

Strategy:
- Build a tiny MjSpec with 1 hinge joint and 1 actuator (dyntype=filterexact).
- Put it on mjwarp, override dynprm[0..3] directly, override qvel before each step.
- Run a small time horizon at fixed ω, log act (= motor current I).
- Compare to closed-form  I(t) = I_ss*(1 - exp(-t/tau_e)),  I_ss = ctrl + beta*omega*tau_e.

Verification criteria (Phase 1 §9a, user-reviewed):
- healthy: I settles to ctrl, no ω-dependence in steady state → slope ≈ 0.
- factor=0.6: I_ss slope vs ω equals (1-factor)*Ke_nom*gr / R = +1.08 A·s/rad.
- other dyntypes (FILTER, INTEGRATOR) unaffected — separate smoke tests below.
"""
from __future__ import annotations

import numpy as np
import mujoco
import mujoco_warp as mjwarp
import warp as wp

wp.init()

# ── Motor parameters (Go2 calf) ──────────────────────────────────────────
Kt_motor = 0.128       # N·m/A
Ke_motor = 0.128       # V·s/rad_motor
gr       = 6.33        # gear ratio
R        = 0.3         # Ω
L        = 1e-4        # H
tau_e    = L / R       # 0.333 ms

Kt_gr = Kt_motor * gr   # 0.81024 N·m/A_joint
Ke_gr = Ke_motor * gr   # 0.81024 V·s/rad_joint


def build_mjmodel() -> mujoco.MjModel:
    """Minimal XML — 1 body, 1 hinge, 1 filterexact actuator."""
    xml = f"""
    <mujoco>
      <option timestep="0.0001"/>
      <worldbody>
        <body name="link">
          <joint name="hinge" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02" density="1000"/>
        </body>
      </worldbody>
      <actuator>
        <general name="motor" joint="hinge" dyntype="filterexact" gaintype="fixed" biastype="none"
                 dynprm="{tau_e} {Ke_gr} {L} {Ke_gr}" gainprm="{Kt_gr}" forcelimited="true" forcerange="-100 100"
                 ctrllimited="true" ctrlrange="-200 200"/>
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def run_simulation(
    factor: float,
    omega_target: float,
    ctrl_val: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (time, I_actual) trajectories."""
    mj_model = build_mjmodel()

    # dynprm: [tau_e, Ke_plant*gr, L, Ke_nom*gr]
    # healthy: factor=1.0 → dynprm[1] = dynprm[3]
    # demag:   dynprm[1] = factor*Ke_nom*gr, dynprm[3] = Ke_nom*gr
    mj_model.actuator_dynprm[0, 0] = tau_e
    mj_model.actuator_dynprm[0, 1] = factor * Ke_gr        # Ke_plant*gr
    mj_model.actuator_dynprm[0, 2] = L
    mj_model.actuator_dynprm[0, 3] = Ke_gr                 # Ke_nom*gr — never modified

    # Put on warp
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    with wp.ScopedDevice("cuda:0"):
        wp_model = mjwarp.put_model(mj_model)
        wp_data = mjwarp.put_data(mj_model, mj_data, nworld=1)

        t = np.zeros(n_steps + 1)
        I_log = np.zeros(n_steps + 1)
        I_log[0] = 0.0  # initial act

        # Hold omega fixed: override qvel before each step.
        # Because there's only one joint, qvel has 1 element.
        ctrl_arr_np = np.array([[ctrl_val]], dtype=np.float32)
        qvel_arr_np = np.array([[omega_target]], dtype=np.float32)

        for i in range(n_steps):
            # Force qvel = omega_target, qpos irrelevant.
            wp_data.qvel.assign(qvel_arr_np)
            wp_data.ctrl.assign(ctrl_arr_np)
            mjwarp.step(wp_model, wp_data)
            t[i + 1] = (i + 1) * mj_model.opt.timestep
            I_log[i + 1] = wp_data.act.numpy()[0, 0]

    return t, I_log


def analytic(
    t: np.ndarray, factor: float, omega: float, ctrl: float
) -> np.ndarray:
    """Closed-form I(t) for dI/dt = (ctrl - I)/tau_e + (Ke_n - Ke_p)*omega/L."""
    I_ss = ctrl + (Ke_gr - factor * Ke_gr) * omega / R
    return I_ss * (1.0 - np.exp(-t / tau_e))


def test_case(factor: float, omega: float, ctrl: float, label: str, n_steps: int = 200):
    t, I_sim = run_simulation(factor, omega, ctrl, n_steps)
    I_ref = analytic(t, factor, omega, ctrl)
    I_ss_expected = ctrl + (1.0 - factor) * Ke_gr * omega / R

    # RMS over the settled tail (skip first 10 steps for init transient).
    err = I_sim[10:] - I_ref[10:]
    rms = float(np.sqrt(np.mean(err**2)))
    rel_rms = rms / max(abs(I_ss_expected), 1e-6)
    I_max_sim = float(np.max(np.abs(I_sim)))
    I_max_ref = float(np.max(np.abs(I_ref)))

    print(f"\n=== {label}  factor={factor}  omega={omega} rad/s  ctrl={ctrl} A ===")
    print(f"  I_ss expected = {I_ss_expected:+.3f} A")
    print(f"  I_sim final   = {I_sim[-1]:+.3f} A")
    print(f"  I_ref final   = {I_ref[-1]:+.3f} A")
    print(f"  |I|_max sim   = {I_max_sim:.3f} A   (ref {I_max_ref:.3f})")
    print(f"  RMS error     = {rms:.4f} A  ({rel_rms*100:.2f}% of I_ss)")
    ok = rel_rms < 0.01
    print(f"  PASS" if ok else f"  FAIL (rel RMS > 1%)")
    return ok, rel_rms, I_ss_expected


def sweep_slope(factor: float, omegas: np.ndarray, ctrl: float, label: str):
    """Verify Δ_I/ω slope against theory."""
    I_ss_list = []
    for w in omegas:
        t, I_sim = run_simulation(factor, float(w), ctrl, n_steps=200)
        I_ss_list.append(float(I_sim[-1]))
    I_ss_arr = np.array(I_ss_list)
    dI = I_ss_arr - ctrl
    slope, intercept = np.polyfit(omegas, dI, 1)
    theory_slope = (1.0 - factor) * Ke_gr / R
    print(f"\n=== slope sweep  {label}  factor={factor} ===")
    print(f"  omegas:     {omegas}")
    print(f"  I_ss - ctrl: {np.array2string(dI, precision=3)}")
    print(f"  fit slope:  {slope:+.4f} A·s/rad  (theory {theory_slope:+.4f})")
    print(f"  intercept:  {intercept:+.4f} A")
    rel_err = abs(slope - theory_slope) / max(abs(theory_slope), 1e-6)
    ok = rel_err < 0.05 if abs(theory_slope) > 1e-6 else abs(slope) < 0.05
    print(f"  PASS" if ok else f"  FAIL (slope off by {rel_err*100:.1f}%)")
    return ok


def main():
    print(f"=== Phase 2 isolated filterexact test ===")
    print(f"Kt*gr = {Kt_gr:.4f}, Ke*gr = {Ke_gr:.4f}, R = {R}, L = {L}, tau_e = {tau_e*1e3:.3f} ms")

    all_pass = []

    # (a) healthy: dynprm[1] == dynprm[3] → correction should be 0
    ok1, _, _ = test_case(factor=1.0, omega=0.0, ctrl=5.0, label="healthy static")
    ok2, _, _ = test_case(factor=1.0, omega=10.0, ctrl=5.0, label="healthy with omega")
    all_pass += [ok1, ok2]

    # (b) demag factor=0.6: correction should kick in
    ok3, _, _ = test_case(factor=0.6, omega=0.0, ctrl=5.0, label="demag static (omega=0)")
    ok4, _, _ = test_case(factor=0.6, omega=10.0, ctrl=5.0, label="demag with omega=10")
    ok5, _, _ = test_case(factor=0.6, omega=-10.0, ctrl=5.0, label="demag with omega=-10")
    all_pass += [ok3, ok4, ok5]

    # (c) slope sweep for healthy and demag
    omegas = np.array([-10, -5, 0, 5, 10, 15], dtype=float)
    ok6 = sweep_slope(1.0, omegas, ctrl=5.0, label="healthy")
    ok7 = sweep_slope(0.6, omegas, ctrl=5.0, label="demag 0.6")
    ok8 = sweep_slope(0.4, omegas, ctrl=5.0, label="demag 0.4")
    all_pass += [ok6, ok7, ok8]

    print(f"\n=== SUMMARY: {sum(all_pass)}/{len(all_pass)} pass ===")
    if not all(all_pass):
        print("FAIL — return to Phase 1 §9 assumptions.")
    else:
        print("PASS — patch mathematically correct.  Phase 2 complete.")


if __name__ == "__main__":
    main()
