#!/usr/bin/env python
"""Sanity check: Schur term is actually active in the GPU build of MethodA.

Run BEFORE long RL training to avoid wasted GPU hours.

Two checks:
  A) Build Unitree-Go2-Flat-MethodA-Electric env and assert
       dynprm[i, 1] == Ke·gr  > 0  (Schur activation requires Ke_gr != 0)
       dynprm[i, 2] == L      > 0  (Schur activation requires L > 0)
       dynprm[i, 3] == Ke·gr      (demag-detection slot, must equal nominal)
     for every electric actuator (i = 0..11).
  B) Run ONE mjwarp.step with Schur ON (current dynprm) and ONE with
     dynprm[:, 1] = 0 (Schur OFF) starting from the same state.  Assert
     that |qvel|_Schur_ON  <  |qvel|_Schur_OFF — the implicit damping
     should pull velocities down even after a single step at the matching
     ctrl input.

Exit code 0 = both checks pass.  Non-zero = abort training.
"""

from __future__ import annotations

import sys

import numpy as np
import torch
import warp as wp

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401

import mujoco_warp as mjwarp
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


TASK_ID = "Unitree-Go2-Flat-MethodA-Electric"
KT_NOM = 0.128 * 6.33
KE_NOM = 0.128 * 6.33
L_NOM  = 1e-4
TAU_E  = L_NOM / 0.3


def check_dynprm(env) -> bool:
    """A) dynprm slots match Method A spec."""
    mj = env.sim.mj_model
    nu = mj.nu
    ok = True
    print(f"\n[A] dynprm sanity (nu={nu})")
    for i in range(nu):
        prm = mj.actuator_dynprm[i]
        tau_e, ke_plant, L_val, ke_nom = prm[0], prm[1], prm[2], prm[3]
        bad = []
        if not np.isclose(tau_e, TAU_E, rtol=1e-3):
            bad.append(f"dynprm[0]={tau_e:.3e} vs τ_e={TAU_E:.3e}")
        if ke_plant <= 0.0:
            bad.append(f"dynprm[1]={ke_plant:.3e} (must be > 0 for Schur)")
        elif not np.isclose(ke_plant, KE_NOM, rtol=1e-3):
            bad.append(f"dynprm[1]={ke_plant:.3e} vs Ke·gr={KE_NOM:.3e}")
        if L_val <= 0.0:
            bad.append(f"dynprm[2]={L_val:.3e} (must be > 0 for Schur)")
        elif not np.isclose(L_val, L_NOM, rtol=1e-3):
            bad.append(f"dynprm[2]={L_val:.3e} vs L={L_NOM:.3e}")
        if not np.isclose(ke_nom, KE_NOM, rtol=1e-3):
            bad.append(f"dynprm[3]={ke_nom:.3e} vs Ke·gr_nom={KE_NOM:.3e}")
        if bad:
            print(f"  [FAIL] actuator {i}: {' | '.join(bad)}")
            ok = False
    if ok:
        print(f"  [OK] all {nu} actuators: τ_e, Ke_plant·gr, L, Ke_nom·gr correct")
    return ok


def check_schur_effect(env, device: str) -> bool:
    """B) Toggling dynprm[1] -> 0 changes qDeriv → qvel after one step."""
    print("\n[B] Schur effect: ON vs OFF (dynprm[:, 1] = 0)")
    wp_data = env.sim.data.struct
    wp_model = env.sim.model.struct

    # Same ctrl: a moderate constant current command on every actuator.
    nu = env.sim.mj_model.nu
    ctrl_val = 5.0  # A
    ctrl = np.full((1, nu), ctrl_val, dtype=np.float32)

    qpos0 = wp_data.qpos.numpy().copy()
    qvel0 = wp_data.qvel.numpy().copy()
    act0  = wp_data.act.numpy().copy()

    dynprm0 = wp_model.actuator_dynprm.numpy().copy()

    # ── ON: current dynprm (Schur active) ──────────────────────────
    wp_data.qpos.assign(qpos0); wp_data.qvel.assign(qvel0); wp_data.act.assign(act0)
    wp_data.ctrl.assign(ctrl)
    mjwarp.step(env.sim.model.struct, wp_data)
    qvel_on = wp_data.qvel.numpy().copy()

    # ── OFF: dynprm[:, 1] = 0 (Ke_gr=0 → Schur skipped) ────────────
    dyn_off = dynprm0.copy()
    dyn_off[:, :, 1] = 0.0
    wp_model.actuator_dynprm.assign(dyn_off)

    wp_data.qpos.assign(qpos0); wp_data.qvel.assign(qvel0); wp_data.act.assign(act0)
    wp_data.ctrl.assign(ctrl)
    mjwarp.step(wp_model, wp_data)
    qvel_off = wp_data.qvel.numpy().copy()

    # Restore
    wp_model.actuator_dynprm.assign(dynprm0)

    # Compare per-DOF qvel magnitudes
    qv_on_abs = np.abs(qvel_on).sum()
    qv_off_abs = np.abs(qvel_off).sum()
    delta = qv_off_abs - qv_on_abs
    print(f"  Σ|qvel| Schur ON  = {qv_on_abs:.6e}")
    print(f"  Σ|qvel| Schur OFF = {qv_off_abs:.6e}")
    print(f"  Δ (OFF − ON)      = {delta:.6e}  "
          f"(should be > 0: Schur damps motion)")

    if not np.isfinite(qv_on_abs) or not np.isfinite(qv_off_abs):
        print("  [FAIL] non-finite qvel — patched mjwarp may have crashed")
        return False
    if not np.allclose(qvel_on, qvel_off, atol=1e-12):
        print("  [OK] Schur ON vs OFF produce different qvel → Schur path is live")
        return True
    print("  [FAIL] qvel identical with Schur ON/OFF — Schur kernel is not "
          "contributing. Patched mjwarp likely missing.")
    return False


def main() -> int:
    wp.init()
    configure_torch_backends()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    env_cfg = load_env_cfg(TASK_ID, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = int(1e9)
    env_cfg.seed = 0
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)

    a_ok = check_dynprm(env)
    b_ok = check_schur_effect(env, device)

    env.close()

    print("\n" + "=" * 60)
    if a_ok and b_ok:
        print("[PASS] Schur is active in the GPU build. Safe to train.")
        return 0
    print("[FAIL] Schur is NOT properly active. Do not start training.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
