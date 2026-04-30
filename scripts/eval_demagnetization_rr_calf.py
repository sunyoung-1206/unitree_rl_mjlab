"""Demagnetization fault injection — RR_calf single motor only.

Usage:
  conda activate mjlab
  python scripts/eval_demagnetization_rr_calf.py [--num-envs 64] [--num-steps 1000]
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
import mjlab.tasks  # noqa: F401 – populate registry
import src.tasks  # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEMAG_FACTORS = [1.0, 0.8, 0.6, 0.4]

NOMINAL_KT_GR = 0.81024  # Kt * gr
NOMINAL_KE_GR = 0.81024  # Ke * gr

TASKS = {
    "PD": {
        "task_id": "Unitree-Go2-Flat",
        "checkpoint": "logs/rsl_rl/go2_velocity/2026-04-01_19-47-51/model_900.pt",
    },
    "Native": {
        "task_id": "Unitree-Go2-Flat-Native-Electric",
        "checkpoint": "logs/rsl_rl/phase4_native/2026-04-07_11-15-28_seed7/model_1999.pt",
    },
    "Coupled": {
        "task_id": "Unitree-Go2-Flat-Coupled-Electric",
        "checkpoint": "logs/rsl_rl/phase5_Aplus_v2/2026-04-14_15-08-42_seed42/model_2900.pt",
    },
}

FAULT_ACTUATOR = 8  # RR_calf (index 8)
OUTPUT_DIR = Path("results/demagnetization_rr_calf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def inject_demagnetization(env: ManagerBasedRlEnv, task_name: str, demag: float):
    """Modify MuJoCo model parameters to simulate demagnetization.

    Demagnetization reduces Ke and Kt by the same factor.
    - PD task: no modification (baseline, no electric model)
    - Native:  gainprm[0] = Kt*gr*demag
    - Coupled: gainprm[0] = Kt*gr*demag, dynprm[1] = Ke*gr*demag
    """
    if task_name == "PD":
        return  # PD doesn't use gainprm/dynprm for torque

    idx = FAULT_ACTUATOR  # RR_calf only

    # Warp model: actuator_gainprm shape (1, nu, 10), shared across envs
    wp_model = env.sim.model.struct
    gainprm = wp_model.actuator_gainprm.numpy()
    gainprm[0, idx, 0] = NOMINAL_KT_GR * demag
    wp_model.actuator_gainprm.assign(gainprm)

    if task_name == "Coupled":
        dynprm = wp_model.actuator_dynprm.numpy()
        dynprm[0, idx, 1] = NOMINAL_KE_GR * demag
        wp_model.actuator_dynprm.assign(dynprm)

    # Also update the mj_model (used by some code paths)
    mj_model = env.sim.mj_model
    mj_model.actuator_gainprm[idx, 0] = NOMINAL_KT_GR * demag
    if task_name == "Coupled":
        mj_model.actuator_dynprm[idx, 1] = NOMINAL_KE_GR * demag


def load_policy(task_id: str, checkpoint: str, env, device: str):
    """Load trained policy from checkpoint."""
    agent_cfg = load_rl_cfg(task_id)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
    return runner.get_inference_policy(device=device)


def collect_mj_data(env: ManagerBasedRlEnv, task_name: str):
    """Read per-step quantities from Warp simulation data (env 0)."""
    wp_data = env.sim.data.struct

    # Warp arrays: shape (num_envs, ...), read env 0
    result = {
        "act": wp_data.act.numpy()[0].astype(np.float32).copy(),
        "qfrc_actuator": wp_data.qfrc_actuator.numpy()[0].astype(np.float32).copy(),
        "qvel": wp_data.qvel.numpy()[0].astype(np.float32).copy(),
        "ctrl": wp_data.ctrl.numpy()[0].astype(np.float32).copy(),
    }
    return result


def apply_fixed_velocity(env: ManagerBasedRlEnv, device: str,
                         vx: float = 0.5, vy: float = 0.0, wz: float = 0.0):
    """Fix velocity command to ensure fair comparison across tasks."""
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

    try:
        term = env.command_manager.get_term("twist")
    except Exception:
        return
    if not isinstance(term, UniformVelocityCommand):
        return

    fixed = torch.tensor([[vx, vy, wz]], device=device)
    term.vel_command_b[:] = fixed
    term._resample_command = lambda env_ids: None


# ---------------------------------------------------------------------------
# Single evaluation run
# ---------------------------------------------------------------------------
def run_eval(task_name: str, demag: float, args) -> dict:
    """Run one evaluation: task × demag_factor → metrics dict."""
    task_info = TASKS[task_name]
    task_id = task_info["task_id"]
    checkpoint = task_info["checkpoint"]

    device = args.device

    # Create environment
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.episode_length_s = int(1e9)  # no truncation

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)

    # Inject demagnetization
    inject_demagnetization(env, task_name, demag)

    # Fix velocity command
    apply_fixed_velocity(env, device, vx=0.5, vy=0.0, wz=0.0)

    # Wrap and load policy
    agent_cfg = load_rl_cfg(task_id)
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = load_policy(task_id, checkpoint, wrapped_env, device)

    # Rollout
    obs, _ = wrapped_env.reset()
    num_envs = env_cfg.scene.num_envs

    all_data = {
        "act": [], "qfrc_actuator": [], "qvel": [], "ctrl": [],
        "reward": [], "base_lin_vel": [], "command_vel": [],
    }

    alive = torch.ones(num_envs, dtype=torch.bool, device=device)
    survival_steps = torch.zeros(num_envs, device=device)

    for step_i in range(args.num_steps):
        actions = policy(obs)
        obs, rew, dones, extras = wrapped_env.step(actions)

        # Track survival
        just_died = (dones.bool()) & alive
        survival_steps[alive] += 1
        alive[just_died] = False

        # Collect MuJoCo data (from env 0 for time series)
        mj_step_data = collect_mj_data(env, task_name)
        for k, v in mj_step_data.items():
            all_data[k].append(v)

        # Collect RL data
        all_data["reward"].append(rew.detach().cpu().numpy().copy())

        # Base velocity and command
        robot = env.scene["robot"]
        base_vel = robot.data.root_link_lin_vel_b.detach().cpu().numpy().copy()
        all_data["base_lin_vel"].append(base_vel)

        try:
            cmd = env.command_manager.get_command("twist").detach().cpu().numpy().copy()
        except Exception:
            cmd = np.zeros((num_envs, 3), dtype=np.float32)
        all_data["command_vel"].append(cmd)

    # Stack arrays
    stacked = {}
    for k, v in all_data.items():
        stacked[k] = np.stack(v, axis=0)  # (num_steps, ...)

    # Compute summary statistics
    rewards = stacked["reward"]  # (num_steps, num_envs)
    episode_return = float(rewards.sum(axis=0).mean())
    mean_reward = float(rewards.mean())

    base_vel = stacked["base_lin_vel"]  # (num_steps, num_envs, 3)
    cmd_vel = stacked["command_vel"]    # (num_steps, num_envs, 3)
    vel_error_rms = float(np.sqrt(np.mean((base_vel[:, :, :2] - cmd_vel[:, :, :2]) ** 2)))

    mean_survival = float(survival_steps.mean().cpu())

    # Motor-specific stats (from env 0 time series)
    # RR_calf: act index 8, DOF index 6+8=14 (6 for floating base)
    fault_act_idx = FAULT_ACTUATOR
    fault_dof_idx = 6 + FAULT_ACTUATOR  # floating base offset

    if task_name != "PD":
        mean_abs_current = float(np.mean(np.abs(stacked["act"][:, fault_act_idx])))
        mean_abs_current_all = float(np.mean(np.abs(stacked["act"])))
    else:
        mean_abs_current = 0.0
        mean_abs_current_all = 0.0
    mean_abs_torque = float(np.mean(np.abs(stacked["qfrc_actuator"][:, fault_dof_idx])))
    mean_abs_torque_all = float(np.mean(np.abs(stacked["qfrc_actuator"][:, 6:])))

    summary = {
        "task": task_name,
        "demag_factor": demag,
        "episode_return": episode_return,
        "mean_reward": mean_reward,
        "vel_error_rms": vel_error_rms,
        "mean_survival_steps": mean_survival,
        "mean_abs_current": mean_abs_current,          # RR_calf only
        "mean_abs_current_all": mean_abs_current_all,  # all 12 motors avg
        "mean_abs_torque": mean_abs_torque,            # RR_calf only
        "mean_abs_torque_all": mean_abs_torque_all,    # all 12 motors avg
    }

    # Save raw data
    out_path = OUTPUT_DIR / f"{task_name}_demag_{demag}.npz"
    np.savez_compressed(out_path, **stacked)
    print(f"  Saved: {out_path}")

    wrapped_env.close()
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Demagnetization fault evaluation")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--tasks", nargs="+", default=["PD", "Native", "Coupled"],
                        choices=["PD", "Native", "Coupled"])
    parser.add_argument("--demag-factors", nargs="+", type=float, default=DEMAG_FACTORS)
    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    configure_torch_backends()

    # Verify checkpoints exist
    for task_name in args.tasks:
        ckpt = Path(TASKS[task_name]["checkpoint"])
        if not ckpt.exists():
            print(f"[ERROR] Checkpoint not found for {task_name}: {ckpt}")
            print(f"  Train first or update TASKS dict in this script.")
            sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    for task_name in args.tasks:
        for demag in args.demag_factors:
            print(f"\n{'='*60}")
            print(f"  {task_name} | demag={demag}")
            print(f"{'='*60}")
            summary = run_eval(task_name, demag, args)
            all_summaries.append(summary)
            print(f"  Return={summary['episode_return']:.1f}  "
                  f"VelErr={summary['vel_error_rms']:.4f}  "
                  f"|I_rr|={summary['mean_abs_current']:.3f}  "
                  f"|τ_rr|={summary['mean_abs_torque']:.3f}  "
                  f"|I_all|={summary['mean_abs_current_all']:.3f}  "
                  f"Survival={summary['mean_survival_steps']:.0f}")

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
