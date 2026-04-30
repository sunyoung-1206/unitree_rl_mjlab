"""PD torque motor response plots (no electric dynamics).

Collects per-physics-step data and generates position/torque/velocity plots
for each joint. Style matches plot_electric_motor.py.

Usage:
  python scripts/plot_pd_motor.py --checkpoint-file <path> --num-steps 2000
"""

from __future__ import annotations
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro

matplotlib.use("Agg")


@dataclass
class PlotConfig:
    checkpoint_file: str
    num_steps: int = 2000
    joints: str | None = None
    vx: float | None = None
    vy: float | None = None
    wz: float | None = None
    tag: str = "pd_"
    out: str = "motor_tracking"
    timestamp: bool = False
    """True면 파일명 끝에 _YYYYMMDD_HHMMSS 시간 태그 추가."""
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def collect_pd_data(cfg: PlotConfig):
    import mjlab.tasks; import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    task_id = "Unitree-Go2-Flat"

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.terminations = {}

    physics_dt = env_cfg.sim.mujoco.timestep
    decimation = env_cfg.decimation
    print(f"[INFO] physics_dt={physics_dt*1e3:.3f}ms  decimation={decimation}  policy_dt={physics_dt*decimation*1e3:.1f}ms")

    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=None)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
    runner.load(cfg.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=cfg.device)
    policy = runner.get_inference_policy(device=cfg.device)

    # Fixed velocity
    if any(v is not None for v in (cfg.vx, cfg.vy, cfg.wz)):
        from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
        try:
            term = env.command_manager.get_term("twist")
            if isinstance(term, UniformVelocityCommand):
                vx = cfg.vx or 0.0; vy = cfg.vy or 0.0; wz = cfg.wz or 0.0
                term.vel_command_b[:] = torch.tensor([[vx, vy, wz]], device=cfg.device)
                term._resample_command = lambda env_ids: None
        except Exception:
            pass

    obs, _ = wrapped.reset()

    # Get joint info from warp model
    import mujoco
    mj_model = env.sim.mj_model
    joint_names = []
    qpos_addrs = []
    dof_addrs = []
    for i in range(mj_model.nu):
        jid = mj_model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        joint_names.append(jname)
        qpos_addrs.append(mj_model.jnt_qposadr[jid])
        dof_addrs.append(mj_model.jnt_dofadr[jid])

    nu = mj_model.nu

    # Collect data
    records = {"pos": [], "vel": [], "qfrc_actuator": [], "step": []}
    step_count = 0

    for ps in range(cfg.num_steps):
        with torch.no_grad():
            action = policy(obs)
        obs, _, _, _ = wrapped.step(action)

        # Read from warp data (env 0)
        wp = env.sim.data.struct
        qpos = wp.qpos.numpy()[0]
        qvel = wp.qvel.numpy()[0]
        qfrc = wp.qfrc_actuator.numpy()[0]

        pos = np.array([qpos[a] for a in qpos_addrs])
        vel = np.array([qvel[a] for a in dof_addrs])
        tau = np.array([qfrc[a] for a in dof_addrs])

        records["pos"].append(pos)
        records["vel"].append(vel)
        records["qfrc_actuator"].append(tau)
        records["step"].append(step_count)
        step_count += 1

    wrapped.close()

    data = {k: np.array(v) for k, v in records.items()}
    return data, joint_names, physics_dt * 1e3, decimation


def plot_pd(data, joint_names, cfg, physics_dt_ms, decimation):
    t_ms = data["step"] * physics_dt_ms * decimation  # policy step times
    policy_dt_ms = physics_dt_ms * decimation

    target_joints = cfg.joints.split(",") if cfg.joints else joint_names
    # Match by suffix (joint_names may have "robot/" prefix)
    def find_idx(name):
        for i, jn in enumerate(joint_names):
            if jn == name or jn.endswith("/" + name):
                return i
        return None
    indices = [find_idx(j) for j in target_joints]
    indices = [i for i in indices if i is not None]

    out_root = Path(cfg.out)

    # Timestamp suffix
    if cfg.timestamp:
        from datetime import datetime
        ts = "_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        ts = ""

    for ji in indices:
        jname = joint_names[ji].split("/")[-1]  # strip "robot/" prefix
        jdir = out_root / jname
        jdir.mkdir(parents=True, exist_ok=True)
        p = cfg.tag

        base_title = f"policy={policy_dt_ms:.1f}ms  physics={physics_dt_ms:.2f}ms  (PD torque)"

        # 1. Torque (= qfrc_actuator for PD)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t_ms, data["qfrc_actuator"][:, ji], lw=1.0, color="tab:red", label="tau_applied")
        ax.set_xlabel("time (ms)"); ax.set_ylabel("N·m")
        ax.set_title(f"{jname}  torque  |  {base_title}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(jdir / f"{p}1_torque{ts}.png", dpi=150); plt.close(fig)

        # 2. Velocity
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t_ms, data["vel"][:, ji], lw=1.0, color="tab:blue", label="omega")
        ax.set_xlabel("time (ms)"); ax.set_ylabel("rad/s")
        ax.set_title(f"{jname}  velocity  |  {base_title}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(jdir / f"{p}2_velocity{ts}.png", dpi=150); plt.close(fig)

        # 3. Position
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(t_ms, data["pos"][:, ji], lw=1.0, color="tab:green", label="pos")
        ax.set_xlabel("time (ms)"); ax.set_ylabel("rad")
        ax.set_title(f"{jname}  position  |  {base_title}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); fig.savefig(jdir / f"{p}3_position{ts}.png", dpi=150); plt.close(fig)

        print(f"[INFO] {jname}: PD plots saved → {jdir}")


if __name__ == "__main__":
    import mjlab; import mjlab.tasks; import src.tasks  # noqa: F401
    cfg = tyro.cli(PlotConfig, config=mjlab.TYRO_FLAGS)
    data, joint_names, pdt, dec = collect_pd_data(cfg)
    plot_pd(data, joint_names, cfg, pdt, dec)
