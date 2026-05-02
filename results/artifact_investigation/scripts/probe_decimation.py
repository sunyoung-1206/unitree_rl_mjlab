"""Probe decimation sensitivity of the healthy baseline slope (+0.1766).

Hypothesis H4: PD ZOH × log decimation interaction.
Test: sweep decimation ∈ {1, 2, 10, 20, 50, 200} (and actuator substeps matched),
measure ω vs ΔI slope over a short healthy rollout.

Design choices:
- Keep physics_dt × decimation = 20ms (same policy rate) so gait behavior comparable.
- Set actuator.substeps = decimation so compute() called every substep with PD
  recompute every substep (trivially = at start of each policy step).
- Run 500 policy steps = 10 s simulated.
- healthy only (no demag injection).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends
from dataclasses import asdict

CHECKPOINT = "logs/rsl_rl/methodA_policy20ms_physics0.1ms/2026-04-17_00-13-42_seed42/model_1999.pt"
TASK = "Unitree-Go2-Flat-MethodA-Electric"
Kt_gr = 0.128 * 6.33  # 0.8102


def run(decimation: int, physics_dt: float, num_steps: int = 500, seed: int = 42):
    configure_torch_backends()
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = int(1e9)
    env_cfg.seed = seed
    env_cfg.decimation = decimation
    env_cfg.sim.mujoco.timestep = physics_dt

    # Match actuator.substeps to decimation so PD recomputes at every substep.
    # (pd_substeps = max(1, decimation // 4) — or just 1 for simplest case)
    for act_cfg in env_cfg.scene.entities["robot"].articulation.actuators:
        if hasattr(act_cfg, "substeps"):
            act_cfg.substeps = max(decimation, 1)
            act_cfg.pd_substeps = max(decimation, 1)  # PD every substep = every policy step

    device = "cuda:0"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)

    # Fixed velocity command
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
    term = env.command_manager.get_term("twist")
    if isinstance(term, UniformVelocityCommand):
        term.vel_command_b[:] = torch.tensor([[0.5, 0.0, 0.0]], device=device)
        term._resample_command = lambda env_ids: None

    # Load policy
    agent_cfg = load_rl_cfg(TASK)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(TASK) or MjlabOnPolicyRunner
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(CHECKPOINT, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Find FL_calf column
    mj = env.sim.mj_model
    joint_names = []
    qvel_adrs = []
    for i in range(mj.nu):
        jid = int(mj.actuator_trnid[i, 0])
        name = mj.joint(jid).name.replace("robot/", "")
        joint_names.append(name)
        qvel_adrs.append(int(mj.jnt_dofadr[jid]))
    col = joint_names.index("FL_calf_joint")

    # Collect data
    I_cmd = np.zeros((num_steps, 12), dtype=np.float32)
    I_act = np.zeros((num_steps, 12), dtype=np.float32)
    qd    = np.zeros((num_steps, 12), dtype=np.float32)

    obs, _ = wrapped.reset()
    for step in range(num_steps):
        actions = policy(obs)
        obs, _, _, _ = wrapped.step(actions)
        ctrl = env.sim.data.struct.ctrl.numpy()[0]
        act  = env.sim.data.struct.act.numpy()[0]
        qv   = env.sim.data.struct.qvel.numpy()[0]
        I_cmd[step] = ctrl
        I_act[step] = act
        qd[step]    = qv[qvel_adrs]

    # Regression
    sk = slice(100, None)
    dI = I_act[sk, col] - I_cmd[sk, col]
    w = qd[sk, col]
    A = np.vstack([w, np.ones_like(w)]).T
    slope, intercept = np.linalg.lstsq(A, dI, rcond=None)[0]
    wrapped.close()

    return {
        "decimation": decimation, "physics_dt": physics_dt,
        "policy_dt_ms": physics_dt * decimation * 1000,
        "slope": float(slope), "intercept": float(intercept),
        "std_dI": float(np.std(dI)), "max_abs_dI": float(np.abs(dI).max()),
        "qd_range": (float(w.min()), float(w.max())),
        "frac_omega_gt3": float(np.mean(np.abs(w) > 3.0)),
    }


def main():
    cases = [
        # (decimation, physics_dt_ms) -- keep policy_dt = physics_dt * decimation
        # Try 5 decimations all with 20ms policy dt
        (1,   0.020),      # 1 substep of 20 ms
        (2,   0.010),      # 2 substeps of 10 ms
        (10,  0.002),      # 10 substeps of 2 ms
        (20,  0.001),      # 20 substeps of 1 ms
        (50,  0.0004),     # 50 substeps of 0.4 ms
        (200, 0.0001),     # 200 substeps of 0.1 ms (baseline — matches production)
    ]
    results = []
    for dec, pdt in cases:
        try:
            r = run(dec, pdt, num_steps=500)
            results.append(r)
            print(f"decimation={dec:>3}  physics_dt={pdt*1000:>6.2f}ms  "
                  f"policy={r['policy_dt_ms']:>5.1f}ms  slope={r['slope']:+.4f}  "
                  f"std={r['std_dI']:.3f}  ω range [{r['qd_range'][0]:.2f}, {r['qd_range'][1]:.2f}]  "
                  f"|ω|>3: {r['frac_omega_gt3']*100:.1f}%")
        except Exception as e:
            print(f"decimation={dec} FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    # Summary
    out = Path("/home/rbdo/unitree_rl_mjlab/results/artifact_investigation/logs/decimation_sweep.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write("decimation physics_dt policy_dt slope intercept std_dI |ω|>3\n")
        for r in results:
            f.write(f"{r['decimation']} {r['physics_dt']*1000:.3f} {r['policy_dt_ms']:.2f} "
                    f"{r['slope']:+.4f} {r['intercept']:+.4f} {r['std_dI']:.3f} "
                    f"{r['frac_omega_gt3']*100:.1f}%\n")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
