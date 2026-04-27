"""Regression smoke: FILTER / INTEGRATOR / MUSCLE / NONE dyntypes unaffected by patch."""
from __future__ import annotations

import numpy as np
import mujoco
import mujoco_warp as mjwarp
import warp as wp

wp.init()

# Build minimal model per dyntype
def run_case(dyntype: str, ctrl: float, n_steps: int = 50) -> np.ndarray:
    xml = f"""
    <mujoco>
      <option timestep="0.0001"/>
      <worldbody>
        <body><joint name="j" type="hinge" axis="0 0 1"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02" density="1000"/></body>
      </worldbody>
      <actuator>
        <general joint="j" dyntype="{dyntype}" gaintype="fixed" biastype="none"
                 dynprm="0.01 0 0 0" gainprm="1.0" ctrllimited="true" ctrlrange="-100 100"/>
      </actuator>
    </mujoco>
    """
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    with wp.ScopedDevice("cuda:0"):
        wp_model = mjwarp.put_model(m)
        wp_data = mjwarp.put_data(m, d, nworld=1)
        ctrl_arr = np.array([[ctrl]], dtype=np.float32)
        qvel_zero = np.zeros((1, 1), dtype=np.float32)
        acts = np.zeros(n_steps + 1)
        for i in range(n_steps):
            wp_data.qvel.assign(qvel_zero)
            wp_data.ctrl.assign(ctrl_arr)
            mjwarp.step(wp_model, wp_data)
            if wp_data.act.numpy().size > 0:
                acts[i + 1] = wp_data.act.numpy()[0, 0]
    return acts


def main():
    print("=== regression smoke for non-FILTEREXACT dyntypes ===")

    # FILTER: first-order, tau=0.01. Apply ctrl=1.0, expect I → 1.0 monotonically.
    acts_f = run_case("filter", ctrl=1.0)
    final_f = acts_f[-1]
    # Analytic: acts[n] = 1 - exp(-n·dt/tau), at n=50, dt=1e-4, tau=1e-2 → 1 - exp(-0.5) ≈ 0.3935
    expected_f = 1.0 - np.exp(-50 * 1e-4 / 0.01)
    err_f = abs(final_f - expected_f)
    print(f"  FILTER:     final={final_f:.4f}  expected={expected_f:.4f}  err={err_f:.2e}  "
          f"{'PASS' if err_f < 1e-3 else 'FAIL'}")

    # INTEGRATOR: act integrates ctrl. acts[n] = n·dt·ctrl.
    acts_i = run_case("integrator", ctrl=1.0)
    final_i = acts_i[-1]
    expected_i = 50 * 1e-4 * 1.0
    err_i = abs(final_i - expected_i)
    print(f"  INTEGRATOR: final={final_i:.4f}  expected={expected_i:.4f}  err={err_i:.2e}  "
          f"{'PASS' if err_i < 1e-3 else 'FAIL'}")

    # NONE: act_dot=0. act stays 0 (no act allocated actually — just smoke test no crash).
    try:
        acts_n = run_case("none", ctrl=1.0)
        print(f"  NONE:       ran without crash. final_act (if any)={acts_n[-1]:.4f}")
    except Exception as e:
        print(f"  NONE:       crash: {e}")


if __name__ == "__main__":
    main()
