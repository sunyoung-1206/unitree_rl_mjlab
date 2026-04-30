#!/usr/bin/env python
"""Demagnetization re-run (v3: all-joint recording + IMU quat/RPY).

Runs ONE condition per invocation.  Launch many in parallel for the matrix.

Matrix:
    pd      × leg=none factor=1.0                             → 1 run
    methoda × leg=none factor=1.0                             → 1 run (healthy)
    methoda × leg∈{FL,FR,RL,RR} factor∈{0.8,0.6,0.4}          → 12 runs

Only ONE calf is demagnetized per run; the other three stay at nominal.

Controller-vs-plant K_t separation (MethodA):
  Controller side: NativeElectricActuator._Ktgr / _Kegr cached in Python,
    used in I_des = tau_des / Kt_nominal + back-EMF FF.  NEVER modified.
  Plant side:      mj_model.actuator_gainprm[i, 0]  (force = gain × act)
                   mj_model.actuator_dynprm[i, 1]  (Schur coupling).
    Demag factor is applied ONLY here, on the single calf listed.
  Reason: controller is unaware of demagnetization — uses nominal K_t.

PD baseline: builtin PD actuator has no K_t / current concept, so demag
modeling is not meaningful.  PD runs only in the nominal case.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import imageio
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

# ─── Experiment registry ──────────────────────────────────────────────
POLICIES = {
    "pd": {
        "task_id": "Unitree-Go2-Flat",
        "checkpoint": "logs/rsl_rl/pd_policy20ms_physics5ms/"
                      "2026-04-17_00-13-37_seed42/model_1999.pt",
        "policy_dt": 0.020,
    },
    "methoda": {
        "task_id": "Unitree-Go2-Flat-MethodA-Electric",
        "checkpoint": "logs/rsl_rl/methodA_policy20ms_physics0.1ms/"
                      "2026-04-17_00-13-42_seed42/model_1999.pt",
        "policy_dt": 0.020,
    },
}

KT_NOMINAL_JOINT = 0.128 * 6.33
KE_NOMINAL_JOINT = 0.128 * 6.33

LEGS = ("FL", "FR", "RL", "RR")
LEG_TO_CALF_IDX = {"FL": 8, "FR": 9, "RL": 10, "RR": 11}


# ─── Demag injection (plant-side only, single calf) ───────────────────
def inject_demagnetization(env, leg: str, factor: float) -> None:
    """Demagnetize ONLY the specified calf.
    Other calves remain at nominal K_t.

    NOTE: controller is unaware of demagnetization — uses nominal K_t.
    Only plant-side K_t (actuator_gainprm) is scaled.
    MethodA's _Ktgr (controller) and _Kegr (back-EMF FF) must NEVER be touched.
    """
    if leg == "none" or factor >= 1.0:
        return
    idx = LEG_TO_CALF_IDX[leg]

    wp = env.sim.model.struct
    gain = wp.actuator_gainprm.numpy()
    gain[0, idx, 0] = KT_NOMINAL_JOINT * factor
    wp.actuator_gainprm.assign(gain)
    dyn = wp.actuator_dynprm.numpy()
    dyn[0, idx, 1] = KE_NOMINAL_JOINT * factor
    wp.actuator_dynprm.assign(dyn)

    mj = env.sim.mj_model
    mj.actuator_gainprm[idx, 0] = KT_NOMINAL_JOINT * factor
    mj.actuator_dynprm[idx, 1] = KE_NOMINAL_JOINT * factor


# ─── Controller-aware demag (scales controller-side Kt) ───────────────
def scale_controller_kt(env, leg: str, factor: float, device: str) -> None:
    """Controller-aware variant: scale _Ktgr for the demagnetized calf ONLY.

    Default experiment (inject_demagnetization only): controller uses Kt_nom →
      I_des = tau_des / Kt_nom                   (controller unaware of demag)
    This variant: controller knows about demag and compensates →
      I_des = tau_des / (Kt_nom · factor)        (inflated current command)

    The calf actuator group contains all 4 calves (FL/FR/RL/RR) and originally
    stores a SCALAR _Ktgr.  We replace it with a per-joint tensor so only the
    target calf column is scaled; other calves keep Kt_nom.  Broadcasting in
    compute(): tau_des [1, 4] / _Ktgr [4] → I_des [1, 4].

    _Kegr (back-EMF FF) is NOT scaled — per user's literal formula, only Kt is
    compensated.  Physical Ke drop on the plant side remains modeled via
    actuator_dynprm[:, 1] (set by inject_demagnetization).
    """
    if leg == "none" or factor >= 1.0:
        return
    target_calf_name = f"{leg}_calf_joint"
    strip = lambda n: n.replace("robot/", "")

    for a in env.scene["robot"].actuators:
        if not hasattr(a, "_Ktgr"):
            continue
        names_clean = [strip(n) for n in a.target_names]
        if target_calf_name not in names_clean:
            continue

        Kt_nom = (float(a._Ktgr) if not torch.is_tensor(a._Ktgr)
                  else float(a._Ktgr.flatten()[0]))
        scale = torch.ones(len(names_clean), device=device, dtype=torch.float32)
        col = names_clean.index(target_calf_name)
        scale[col] = factor
        a._Ktgr = Kt_nom * scale  # [num_joints_in_group]

        print(f"[INFO] Controller-aware: {target_calf_name} _Ktgr × {factor} "
              f"(group columns {names_clean} → "
              f"{[round(v, 4) for v in (Kt_nom*scale).tolist()]})")
        return
    print(f"[WARN] No calf actuator group found containing '{target_calf_name}'")


# ─── Orientation utility ──────────────────────────────────────────────
def quat_wxyz_to_rpy_zyx_intrinsic(q: np.ndarray) -> np.ndarray:
    """ZYX intrinsic Euler angles (standard robotics roll-pitch-yaw).
    q shape (..., 4) = [qw, qx, qy, qz].  Returns (..., 3) = [roll, pitch, yaw].
    """
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    roll  = np.arctan2(2 * (qw * qx + qy * qz),
                       1 - 2 * (qx * qx + qy * qy))
    pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
    yaw   = np.arctan2(2 * (qw * qz + qx * qy),
                       1 - 2 * (qy * qy + qz * qz))
    return np.stack([roll, pitch, yaw], axis=-1)


# ─── Helpers ──────────────────────────────────────────────────────────
def apply_fixed_velocity(env, device, vx=0.5, vy=0.0, wz=0.0) -> None:
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
    term = env.command_manager.get_term("twist")
    if isinstance(term, UniformVelocityCommand):
        term.vel_command_b[:] = torch.tensor([[vx, vy, wz]], device=device)
        term._resample_command = lambda env_ids: None


def load_policy(task_id, checkpoint, env, device):
    agent_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True,
                map_location=device)
    return runner.get_inference_policy(device=device)


def build_column_map(mj_model):
    """For each of the 12 actuators, find its driven joint.
    Returns (joint_names, qpos_adrs, qvel_adrs) with column order = actuator order.
    All per-joint arrays (tau, I, q_all, qd_all) share this ordering.
    """
    joint_names, qpos_adrs, qvel_adrs = [], [], []
    for i in range(mj_model.nu):
        jid = int(mj_model.actuator_trnid[i, 0])
        name = mj_model.joint(jid).name.replace("robot/", "")
        joint_names.append(name)
        qpos_adrs.append(int(mj_model.jnt_qposadr[jid]))
        qvel_adrs.append(int(mj_model.jnt_dofadr[jid]))
    return joint_names, np.asarray(qpos_adrs), np.asarray(qvel_adrs)


def output_paths(policy: str, leg: str, factor: float,
                 out_root: Path) -> tuple[Path, Path, str]:
    if policy == "pd":
        tag = "nominal"
    elif leg == "none" or factor >= 1.0:
        tag = "healthy"
    else:
        tag = f"{leg}_{factor:.1f}"
    npz = out_root / "data" / policy / f"{tag}.npz"
    mp4 = out_root / "videos" / f"{policy}_{tag}.mp4"
    return npz, mp4, tag


def write_video(path: Path, frames: list, fps: int) -> None:
    writer = imageio.get_writer(
        str(path), fps=fps, codec="libx264",
        macro_block_size=1, pixelformat="yuv420p", quality=8,
        ffmpeg_params=["-profile:v", "baseline", "-level", "3.0",
                       "-bf", "0", "-movflags", "+faststart"],
    )
    try:
        for f in frames:
            writer.append_data(f)
    finally:
        writer.close()
    size = path.stat().st_size
    if size < 10_000:
        print(f"[WARN] Video file suspiciously small ({size} B): {path}")
    else:
        print(f"[OK] Video saved: {path} ({len(frames)} frames @ {fps}fps, "
              f"{size / 1024:.0f} KB)")


def clean_outputs(out_root: Path) -> None:
    for sub in ("data", "videos", "plots"):
        d = out_root / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
    print(f"[CLEAN] wiped {out_root}/{{data,videos,plots}}/")


# ─── Main rollout ─────────────────────────────────────────────────────
def run_one(args) -> None:
    device = args.device
    info = POLICIES[args.policy]

    if args.policy == "pd" and (args.leg != "none" or args.demag_factor < 1.0):
        print("[ERROR] PD baseline must be --leg none --demag-factor 1.0")
        sys.exit(2)

    env_cfg = load_env_cfg(info["task_id"], play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = int(1e9)
    env_cfg.seed = args.seed
    env_cfg.viewer.width = args.video_width
    env_cfg.viewer.height = args.video_height

    if args.schur_test and args.policy == "methoda":
        changed = 0
        for act_cfg in env_cfg.scene.entities["robot"].articulation.actuators:
            if hasattr(act_cfg, "use_filterexact_schur"):
                act_cfg.use_filterexact_schur = True
                changed += 1
        print(f"[SCHUR TEST] flipped use_filterexact_schur=True on "
              f"{changed} actuator cfgs (dynprm[3]=1.0, Schur complement).")

    render_mode = None if args.no_video else "rgb_array"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    # Snapshot dynprm[:, 3] at build (boost A — detect silent overwrite).
    dynprm3_at_build = env.sim.mj_model.actuator_dynprm[:, 3].copy()
    print(f"[DYNPRM3 @ build]     {np.round(dynprm3_at_build, 4).tolist()}")

    if args.policy == "methoda":
        if args.leg == "none":
            print("[INFO] MethodA healthy — no demag applied.")
        else:
            inject_demagnetization(env, args.leg, args.demag_factor)
            print(f"[INFO] MethodA {args.leg}_calf plant Kt·gr / Ke·gr × "
                  f"{args.demag_factor} (idx {LEG_TO_CALF_IDX[args.leg]}).")
            if args.ctrl_aware:
                scale_controller_kt(env, args.leg, args.demag_factor, device)
            else:
                print(f"[INFO] Controller _Ktgr / _Kegr unchanged "
                      f"(controller unaware of demag).")
        # After demag injection: dynprm[3] must still match build snapshot.
        post_inject = env.sim.mj_model.actuator_dynprm[:, 3]
        assert np.allclose(post_inject, dynprm3_at_build), (
            f"dynprm[:, 3] (Ke_nom·gr) modified by demag injection! "
            f"before={dynprm3_at_build} after={post_inject} — silent failure risk")
    else:
        print("[INFO] PD baseline — no demag applied.")

    apply_fixed_velocity(env, device, vx=args.vx, vy=args.vy, wz=args.wz)

    agent_cfg = load_rl_cfg(info["task_id"])
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = load_policy(info["task_id"], info["checkpoint"], wrapped, device)

    joint_names, qpos_adrs, qvel_adrs = build_column_map(env.sim.mj_model)
    assert len(joint_names) == 12, f"expected 12 joints, got {len(joint_names)}"

    # Map policy-commanded q_des (in action-term order) → actuator column order.
    act_term = env.action_manager.get_term("joint_pos")
    target_ids = act_term._target_ids.detach().cpu().numpy().tolist()
    entity_joint_names = list(env.scene["robot"].joint_names)
    action_joint_names = [entity_joint_names[i] for i in target_ids]
    strip = lambda n: n.replace("robot/", "")
    name_to_action_idx = {strip(n): i for i, n in enumerate(action_joint_names)}
    q_des_perm = np.array(
        [name_to_action_idx[strip(n)] for n in joint_names], dtype=np.int64)

    # Foot contact sensor (columns = primary geom order the sensor resolved).
    feet_sensor = env.scene["feet_ground_contact"]
    foot_contact_names = [
        s.primary_name for s in feet_sensor._slots if s.field_name == "found"]

    # Electric actuator handles for tau_des extraction (pre V_bus clamp).
    name_to_col = {n: i for i, n in enumerate(joint_names)}
    electric_actuators: list[tuple[object, list[int]]] = []
    if args.policy == "methoda":
        for a in env.scene["robot"].actuators:
            if hasattr(a, "_tau_des_hold"):
                cols = [name_to_col[strip(n)] for n in a.target_names]
                electric_actuators.append((a, cols))

    out_root = Path(args.output_dir)
    npz_path, mp4_path, case_tag = output_paths(
        args.policy, args.leg, args.demag_factor, out_root)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_video:
        mp4_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Buffers ────────────────────────────────────────────────────
    N = args.num_steps
    tau_cmd    = np.zeros((N, 12), dtype=np.float32)
    tau_des    = np.zeros((N, 12), dtype=np.float32)
    tau_actual = np.zeros((N, 12), dtype=np.float32)
    I_cmd      = np.zeros((N, 12), dtype=np.float32)
    I_des      = np.zeros((N, 12), dtype=np.float32)
    I_actual   = np.zeros((N, 12), dtype=np.float32)
    q_all      = np.zeros((N, 12), dtype=np.float32)
    qd_all     = np.zeros((N, 12), dtype=np.float32)
    q_des      = np.zeros((N, 12), dtype=np.float32)
    foot_contact = np.zeros((N, len(foot_contact_names)), dtype=np.uint8)
    base_pos   = np.zeros((N, 3),  dtype=np.float32)
    base_quat  = np.zeros((N, 4),  dtype=np.float32)  # [w,x,y,z]
    base_lin_vel = np.zeros((N, 3), dtype=np.float32)
    base_ang_vel = np.zeros((N, 3), dtype=np.float32)
    cmd_vel    = np.zeros((N, 3),  dtype=np.float32)
    frames: list[np.ndarray] = []

    obs, _ = wrapped.reset()
    dynprm3_at_rollout_start = env.sim.mj_model.actuator_dynprm[:, 3].copy()
    assert np.allclose(dynprm3_at_rollout_start, dynprm3_at_build), (
        f"dynprm[:, 3] modified between build and rollout start! "
        f"build={dynprm3_at_build} start={dynprm3_at_rollout_start}")
    print(f"[DYNPRM3 @ rollout]    {np.round(dynprm3_at_rollout_start, 4).tolist()}")

    for step in range(N):
        actions = policy(obs)
        obs, rew, dones, extras = wrapped.step(actions)

        wp_data = env.sim.data.struct
        ctrl = wp_data.ctrl.numpy()
        act_raw = wp_data.act.numpy()
        qfrc = wp_data.qfrc_actuator.numpy()[0]
        qpos = wp_data.qpos.numpy()[0]
        qvel = wp_data.qvel.numpy()[0]

        ctrl0 = ctrl[0] if ctrl.shape[1] > 0 else np.zeros(12)
        act0  = act_raw[0] if act_raw.shape[1] > 0 else np.zeros(12)

        # Per-joint arrays (column order = actuator order).
        tau_actual[step] = qfrc[qvel_adrs]
        q_all[step]      = qpos[qpos_adrs]
        qd_all[step]     = qvel[qvel_adrs]
        q_des_full = (
            act_term._processed_actions.detach().cpu().numpy()[0])
        q_des[step] = q_des_full[q_des_perm]
        found = feet_sensor.data.found
        foot_contact[step] = (
            (found[0] > 0).detach().cpu().numpy().astype(np.uint8))
        if args.policy == "methoda":
            I_cmd[step]    = ctrl0
            I_actual[step] = act0 if act0.size == 12 else 0.0
            tau_cmd[step]  = ctrl0 * KT_NOMINAL_JOINT
            for a, cols in electric_actuators:
                hold = a._tau_des_hold
                if hold is None:
                    continue
                td = hold[0].detach().cpu().numpy()
                for k, c in enumerate(cols):
                    tau_des[step, c] = td[k]
            I_des[step] = tau_des[step] / KT_NOMINAL_JOINT
        else:
            # PD: no V_bus saturation stage; tau_des = tau_applied, I derived.
            tau_cmd[step]  = tau_actual[step]
            tau_des[step]  = tau_actual[step]
            I_cmd[step]    = tau_cmd[step] / KT_NOMINAL_JOINT
            I_des[step]    = I_cmd[step]
            I_actual[step] = I_cmd[step]

        base_pos[step]  = qpos[0:3]
        base_quat[step] = qpos[3:7]   # MuJoCo free joint: [w, x, y, z]

        robot = env.scene["robot"]
        base_lin_vel[step] = (
            robot.data.root_link_lin_vel_b.detach().cpu().numpy()[0])
        base_ang_vel[step] = (
            robot.data.root_link_ang_vel_b.detach().cpu().numpy()[0])

        try:
            cmd = env.command_manager.get_command(
                "twist").detach().cpu().numpy()[0]
        except Exception:
            cmd = np.zeros(3, dtype=np.float32)
        cmd_vel[step] = cmd

        if not args.no_video:
            frame = env.render()
            if frame is not None:
                frames.append(frame)

    base_rpy = quat_wxyz_to_rpy_zyx_intrinsic(base_quat).astype(np.float32)

    # Post-rollout dynprm[3] sanity (boost A).
    dynprm3_at_rollout_end = env.sim.mj_model.actuator_dynprm[:, 3].copy()
    if not np.allclose(dynprm3_at_rollout_end, dynprm3_at_build):
        print(f"[WARN] dynprm[:, 3] CHANGED during rollout! "
              f"build={dynprm3_at_build} end={dynprm3_at_rollout_end}")
    print(f"[DYNPRM3 @ end]        {np.round(dynprm3_at_rollout_end, 4).tolist()}")

    # |I_actual| statistics + Go2 motor current limit comparison (boost 2).
    I_abs = np.abs(I_actual)
    # I_max_spec: per-joint actuator effort_limit / Kt_nom·gr.
    # Hip/thigh: effort=23.5, calf: effort=45.0; Kt·gr = 0.8102.
    I_max_per_joint = np.zeros(12, dtype=np.float32)
    for i, nm in enumerate(joint_names):
        eff = 45.0 if "calf" in nm else 23.5
        I_max_per_joint[i] = eff / KT_NOMINAL_JOINT
    I_over_pct = (I_abs.max(axis=0) / I_max_per_joint) * 100.0
    print(f"[I_actual] max per col (A): {np.round(I_abs.max(axis=0), 2).tolist()}")
    print(f"[I_actual] Go2 spec limit:  {np.round(I_max_per_joint, 2).tolist()}")
    print(f"[I_actual] %%-of-limit max: {np.round(I_over_pct, 1).tolist()}")
    if np.any(I_over_pct > 100.0):
        print(f"[WARN] I_actual exceeds Go2 motor current limit — physical plausibility issue (report only, not a fail).")

    # ── Save data ──────────────────────────────────────────────────
    meta = {
        "policy_name": args.policy,
        "task_id": info["task_id"],
        "checkpoint": info["checkpoint"],
        "demag_leg": args.leg,
        "demag_factor": float(args.demag_factor),
        "controller_aware": bool(args.ctrl_aware),
        "controller_aware_note": (
            "if True, controller _Ktgr for the demagnetized calf was scaled by "
            "demag_factor, i.e. I_des = tau_des / (Kt_nom * factor). _Kegr "
            "(back-EMF FF) was left at nominal. If False, controller uses "
            "nominal Kt (unaware of demag)."),
        "case_tag": case_tag,
        "num_steps": N,
        "dt": info["policy_dt"],
        "fs": 1.0 / info["policy_dt"],
        "total_seconds": N * info["policy_dt"],
        "seed": args.seed,
        "cmd": {"vx": args.vx, "vy": args.vy, "wz": args.wz},
        "leg_to_calf_idx": LEG_TO_CALF_IDX,
        "kt_nominal_joint": KT_NOMINAL_JOINT,
        "ke_nominal_joint": KE_NOMINAL_JOINT,
        "q_all_joint_names": joint_names,
        "q_des_note": (
            "policy-commanded joint position target = "
            "raw_action * scale + default_joint_pos (pre encoder_bias); "
            "column order matches q_all_joint_names"),
        "tau_cmd_note": (
            "I_cmd * Kt_nominal; post V_bus clamp. equals tau_des "
            "when voltage is not saturated. for PD baseline, mirrors "
            "tau_actual (no saturation stage)."),
        "tau_des_note": (
            "PD+DcMotorActuator direct output (after effort_limit / DC "
            "torque-speed envelope, before V_bus clamp); read from "
            "NativeElectricActuator._tau_des_hold, held per PD period. "
            "for PD baseline, mirrors tau_actual."),
        "I_des_note": "tau_des / Kt_nominal; pre V_bus clamp target current.",
        "foot_contact_names": foot_contact_names,
        "foot_contact_note": (
            "1 = foot geom in contact with terrain this step, 0 = no contact; "
            "column order = foot_contact_names"),
        "quat_convention": "wxyz (MuJoCo scalar-first)",
        "rpy_convention": "ZYX intrinsic",
    }
    np.savez_compressed(
        npz_path,
        tau_cmd=tau_cmd, tau_des=tau_des, tau_actual=tau_actual,
        I_cmd=I_cmd, I_des=I_des, I_actual=I_actual,
        q_all=q_all, qd_all=qd_all, q_des=q_des,
        foot_contact=foot_contact,
        base_pos=base_pos, base_quat=base_quat, base_rpy=base_rpy,
        base_lin_vel=base_lin_vel, base_ang_vel=base_ang_vel,
        cmd_vel=cmd_vel,
        meta=json.dumps(meta),
    )
    print(f"[OK] Data saved: {npz_path}")

    if not args.no_video and frames:
        fps = int(round(1.0 / info["policy_dt"]))
        write_video(mp4_path, frames, fps)

    # ── Validation ratio (MethodA demag only) ──────────────────────
    if args.policy == "methoda" and args.leg != "none" and args.demag_factor < 1.0:
        calf_name = f"{args.leg}_calf_joint"
        try:
            col = joint_names.index(calf_name)
        except ValueError:
            print(f"[WARN] column '{calf_name}' not in joint_names")
            return
        tc = float(np.mean(np.abs(tau_cmd[-200:, col])))
        ta = float(np.mean(np.abs(tau_actual[-200:, col])))
        measured = ta / max(tc, 1e-6)
        expected = float(args.demag_factor)
        if expected - 0.05 <= measured <= expected + 0.05:
            print(f"[OK] {args.leg}×{expected}: measured ratio "
                  f"{measured:.3f} (within ±0.05 of spec)")
        else:
            print(f"[WARN] {args.leg}×{expected}: measured ratio "
                  f"{measured:.3f} diverges from spec {expected:.3f} "
                  f"(possible current saturation)")

    wrapped.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=list(POLICIES.keys()), required=True)
    parser.add_argument("--leg", choices=[*LEGS, "none"], default="none")
    parser.add_argument("--demag-factor", type=float, default=1.0)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--wz", type=float, default=0.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="results/demag_rerun")
    parser.add_argument("--video-width", type=int, default=640)
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--clean", action="store_true",
                        help="wipe data/videos/plots before starting "
                             "(use only on the first run of a batch)")
    parser.add_argument("--no-video", action="store_true",
                        help="skip mp4 rendering (data npz only)")
    parser.add_argument("--schur-test", action="store_true",
                        help="flip use_filterexact_schur=True at eval time "
                             "(plant dynprm[3]=1.0, Schur complement path)")
    parser.add_argument("--ctrl-aware", action="store_true",
                        help="controller-aware demag: scale controller-side "
                             "_Ktgr for the demag'd calf so I_des = tau_des / "
                             "(Kt_nom * factor). _Kegr (back-EMF FF) stays at "
                             "nominal. No effect with --policy pd.")
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    configure_torch_backends()

    if args.clean:
        clean_outputs(Path(args.output_dir))

    run_one(args)


if __name__ == "__main__":
    main()
