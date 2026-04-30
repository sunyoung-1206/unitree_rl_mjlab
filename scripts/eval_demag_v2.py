"""Demagnetization v2: RR_calf single-motor fault, 4 methods (PD/Native/A/A+).

Usage:
  conda activate mjlab
  python scripts/eval_demag_v2.py [--num-envs 64] [--num-steps 1000]
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

# ---------------------------------------------------------------------------
DEMAG_FACTORS = [1.0, 0.8, 0.6, 0.4]
NOMINAL_KT_GR = 0.81024
NOMINAL_KE_GR = 0.81024
FAULT_ACTUATOR = 8   # RR_calf
COMPARE_ACTUATOR = 11  # RL_calf (symmetric normal joint)

TASKS = {
    "PD": {
        "task_id": "Unitree-Go2-Flat",
        "checkpoint": "logs/rsl_rl/demag_v2_PD/2026-04-15_01-06-35_seed42_dt5ms/model_2899.pt",
    },
    "Native": {
        "task_id": "Unitree-Go2-Flat-Native-Electric",
        "checkpoint": "logs/rsl_rl/demag_v2_Native/2026-04-14_22-41-27_seed42/model_2899.pt",
    },
    "MethodA": {
        "task_id": "Unitree-Go2-Flat-MethodA-Electric",
        "checkpoint": "logs/rsl_rl/demag_v2_MethodA/2026-04-14_22-41-25_seed42/model_2899.pt",
    },
    "Aplus": {
        "task_id": "Unitree-Go2-Flat-Coupled-Electric",
        "checkpoint": "logs/rsl_rl/demag_v2_Aplus/2026-04-14_22-41-26_seed42/model_2899.pt",
    },
}

OUTPUT_DIR = Path("results/demagnetization_v2")


# ---------------------------------------------------------------------------
def inject_demagnetization(env, task_name, demag):
    if task_name == "PD":
        return

    idx = FAULT_ACTUATOR
    wp_model = env.sim.model.struct

    # Reset all to nominal first
    gainprm = wp_model.actuator_gainprm.numpy()
    gainprm[0, :, 0] = NOMINAL_KT_GR
    gainprm[0, idx, 0] = NOMINAL_KT_GR * demag
    wp_model.actuator_gainprm.assign(gainprm)

    if task_name in ("MethodA", "Aplus"):
        dynprm = wp_model.actuator_dynprm.numpy()
        dynprm[0, :, 1] = NOMINAL_KE_GR  # reset all to nominal
        dynprm[0, idx, 1] = NOMINAL_KE_GR * demag  # fault on RR_calf only
        wp_model.actuator_dynprm.assign(dynprm)

    # mj_model too
    mj_model = env.sim.mj_model
    for i in range(mj_model.nu):
        mj_model.actuator_gainprm[i, 0] = NOMINAL_KT_GR
        if task_name in ("MethodA", "Aplus"):
            mj_model.actuator_dynprm[i, 1] = NOMINAL_KE_GR
    mj_model.actuator_gainprm[idx, 0] = NOMINAL_KT_GR * demag
    if task_name in ("MethodA", "Aplus"):
        mj_model.actuator_dynprm[idx, 1] = NOMINAL_KE_GR * demag


def load_policy(task_id, checkpoint, env, device):
    agent_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


def apply_fixed_velocity(env, device, vx=0.5, vy=0.0, wz=0.0):
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
    try:
        term = env.command_manager.get_term("twist")
    except Exception:
        return
    if isinstance(term, UniformVelocityCommand):
        term.vel_command_b[:] = torch.tensor([[vx, vy, wz]], device=device)
        term._resample_command = lambda env_ids: None


# ---------------------------------------------------------------------------
def run_eval(task_name, demag, args):
    task_info = TASKS[task_name]
    device = args.device

    env_cfg = load_env_cfg(task_info["task_id"], play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = int(1e9)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    inject_demagnetization(env, task_name, demag)
    apply_fixed_velocity(env, device)

    agent_cfg = load_rl_cfg(task_info["task_id"])
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = load_policy(task_info["task_id"], task_info["checkpoint"], wrapped, device)

    obs, _ = wrapped.reset()
    ne = args.num_envs

    # Collect signed data for RR_calf (fault) and RL_calf (normal)
    fi = FAULT_ACTUATOR      # act index
    ci = COMPARE_ACTUATOR    # act index
    fd = 6 + fi              # DOF index (skip floating base)
    cd = 6 + ci

    rec = {k: [] for k in [
        "rr_calf_current", "rr_calf_torque", "rr_calf_velocity", "rr_calf_ctrl",
        "rl_calf_current", "rl_calf_torque", "rl_calf_velocity",
        "reward", "base_lin_vel", "base_ang_vel", "command_vel", "base_height",
    ]}

    alive = torch.ones(ne, dtype=torch.bool, device=device)
    survival = torch.zeros(ne, device=device)

    for step in range(args.num_steps):
        actions = policy(obs)
        obs, rew, dones, extras = wrapped.step(actions)

        just_died = dones.bool() & alive
        survival[alive] += 1
        alive[just_died] = False

        wp = env.sim.data.struct
        act = wp.act.numpy()[0]
        qfrc = wp.qfrc_actuator.numpy()[0]
        qvel = wp.qvel.numpy()[0]
        ctrl = wp.ctrl.numpy()[0]

        has_act = len(act) > 0
        rec["rr_calf_current"].append(float(act[fi]) if has_act else 0.0)
        rec["rr_calf_torque"].append(float(qfrc[fd]))
        rec["rr_calf_velocity"].append(float(qvel[fd]))
        rec["rr_calf_ctrl"].append(float(ctrl[fi]) if len(ctrl) > fi else 0.0)
        rec["rl_calf_current"].append(float(act[ci]) if has_act else 0.0)
        rec["rl_calf_torque"].append(float(qfrc[cd]))
        rec["rl_calf_velocity"].append(float(qvel[cd]))

        rec["reward"].append(rew.detach().cpu().numpy().copy())

        robot = env.scene["robot"]
        rec["base_lin_vel"].append(robot.data.root_link_lin_vel_b.detach().cpu().numpy()[0].copy())
        rec["base_ang_vel"].append(robot.data.root_link_ang_vel_b.detach().cpu().numpy()[0].copy())
        rec["base_height"].append(float(wp.qpos.numpy()[0, 2]))

        try:
            cmd = env.command_manager.get_command("twist").detach().cpu().numpy()[0].copy()
        except Exception:
            cmd = np.zeros(3, dtype=np.float32)
        rec["command_vel"].append(cmd)

    # Stack
    for k in rec:
        rec[k] = np.array(rec[k])

    rewards = np.stack([r for r in rec["reward"]])  # (steps, envs) or (steps,) depending
    ep_return = float(np.sum(rewards.mean(axis=-1)) if rewards.ndim > 1 else np.sum(rewards))

    summary = {
        "task": task_name, "demag_factor": demag,
        "episode_return": ep_return,
        "mean_survival_steps": float(survival.mean().cpu()),
        "rr_calf_mean_abs_I": float(np.mean(np.abs(rec["rr_calf_current"]))),
        "rr_calf_mean_abs_tau": float(np.mean(np.abs(rec["rr_calf_torque"]))),
        "rl_calf_mean_abs_I": float(np.mean(np.abs(rec["rl_calf_current"]))),
        "rl_calf_mean_abs_tau": float(np.mean(np.abs(rec["rl_calf_torque"]))),
    }

    out = OUTPUT_DIR / f"{task_name}_demag_{demag}.npz"
    np.savez_compressed(out, **rec)
    print(f"  Saved: {out}")

    wrapped.close()
    return summary


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.keys()),
                        choices=list(TASKS.keys()))
    parser.add_argument("--demag-factors", nargs="+", type=float, default=DEMAG_FACTORS)
    args = parser.parse_args()
    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    configure_torch_backends()

    for tn in args.tasks:
        ckpt = Path(TASKS[tn]["checkpoint"])
        if not ckpt.exists():
            print(f"[ERROR] Checkpoint not found: {tn}: {ckpt}"); sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries = []
    for tn in args.tasks:
        for demag in args.demag_factors:
            print(f"\n{'='*60}\n  {tn} | demag={demag}\n{'='*60}")
            s = run_eval(tn, demag, args)
            summaries.append(s)
            print(f"  Return={s['episode_return']:.1f}  Surv={s['mean_survival_steps']:.0f}  "
                  f"RR|I|={s['rr_calf_mean_abs_I']:.3f}  RR|τ|={s['rr_calf_mean_abs_tau']:.3f}  "
                  f"RL|I|={s['rl_calf_mean_abs_I']:.3f}")

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSummary saved: {OUTPUT_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
