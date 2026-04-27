"""Native vs Coupled 비교 플롯: 전류 잔차, 20초 시계열.

Usage:
  python scripts/plot_native_vs_coupled.py \
    --native-ckpt logs/rsl_rl/phase4_native/.../model_900.pt \
    --coupled-ckpt logs/rsl_rl/phase4_coupled/.../model_900.pt
"""
import os, sys
os.environ["MUJOCO_GL"] = "egl"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

def collect(task_id, ckpt_path, num_steps=4000, vx=0.5, device="cuda:0"):
    """정책 rollout 후 physics step 데이터 수집."""
    import mjlab.tasks, src.tasks
    from dataclasses import asdict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    from src.assets.robots.unitree_go2.electric_actuator import ElectricMotorActuator
    from src.assets.robots.unitree_go2.mj_native_electric_actuator import NativeElectricActuator
    def get_electric_actuators(env):
        entity = env.unwrapped.scene["robot"]
        return [a for a in entity._custom_actuators
                if isinstance(a, (ElectricMotorActuator, NativeElectricActuator))]

    configure_torch_backends()
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.terminations = {}

    physics_dt = env_cfg.sim.mujoco.timestep
    decimation = env_cfg.decimation

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env_w = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env_w, asdict(agent_cfg), device=device)
    runner.load(ckpt_path, load_cfg={"actor": True}, strict=True, map_location=device)
    policy = runner.get_inference_policy(device=device)

    actuators = get_electric_actuators(env_w)
    for act in actuators:
        act.start_logging()

    obs, _ = env_w.reset()

    # 고정 속도
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
    try:
        term = env_w.unwrapped.command_manager.get_term("twist")
        if isinstance(term, UniformVelocityCommand):
            term.vel_command_b[:] = torch.tensor([[vx, 0.0, 0.0]], device=device)
            term._resample_command = lambda env_ids: None
    except: pass

    for _ in range(num_steps):
        with torch.no_grad():
            action = policy(obs)
        obs, _, _, _ = env_w.step(action)

    # 수집
    all_joint_names = []
    for act in actuators:
        all_joint_names.extend(act.target_names)

    combined = {}
    steps_arr = None
    for act in actuators:
        log = act.get_log()
        if not log: continue
        for k, v in log.items():
            if k == "physics_step":
                if steps_arr is None: steps_arr = np.array(v)
            else:
                combined.setdefault(k, []).append(np.stack(v))

    arrays = {k: np.concatenate(v, axis=1) for k, v in combined.items()}

    # I_des 유도
    if "I_des" not in arrays and "tau_des" in arrays:
        from src.assets.robots.unitree_go2.mj_native_electric_actuator import NativeElectricActuator
        for act in actuators:
            if isinstance(act, NativeElectricActuator):
                arrays["I_des"] = arrays["tau_des"] / act._Ktgr
                break
    if "I" in arrays and "I_after" not in arrays:
        arrays["I_after"] = arrays["I"]

    n = min(len(v) for v in arrays.values())
    if steps_arr is not None: n = min(n, len(steps_arr))
    t_ms = (steps_arr[:n] if steps_arr is not None else np.arange(n)) * physics_dt * 1000

    result = {k: v[:n] for k, v in arrays.items()}
    result["t_ms"] = t_ms

    env.close()
    return result, all_joint_names, decimation


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-ckpt", required=True)
    parser.add_argument("--coupled-ckpt", required=True)
    parser.add_argument("--num-steps", type=int, default=4000)
    parser.add_argument("--vx", type=float, default=0.5)
    parser.add_argument("--joints", default="FR_thigh_joint,FR_calf_joint,RL_thigh_joint")
    parser.add_argument("--out", default="solver_comparison/phase4_results/comparison_plots")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    target_joints = args.joints.split(",")

    print("Collecting native data...")
    d_nat, names, dec = collect("Unitree-Go2-Flat-Native-Electric", args.native_ckpt,
                                 args.num_steps, args.vx, args.device)
    print("Collecting coupled data...")
    d_cpl, _, _ = collect("Unitree-Go2-Flat-Coupled-Electric", args.coupled_ckpt,
                           args.num_steps, args.vx, args.device)

    for jname in target_joints:
        if jname not in names:
            print(f"  Skip {jname} (not found)")
            continue
        ji = names.index(jname)
        t_nat = d_nat["t_ms"]
        t_cpl = d_cpl["t_ms"]

        # ── 1. 전류 잔차 비교 (I_des - I_after) ──
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"{jname}  —  Native vs Coupled current residual (vx={args.vx})", fontsize=12)

        # 패널 1: I_des vs I_after (native)
        ax = axes[0]
        ax.plot(t_nat / 1000, d_nat["I_des"][:, ji], 'k--', lw=0.8, label="I_des")
        ax.plot(t_nat / 1000, d_nat["I_after"][:, ji], 'r-', lw=0.6, alpha=0.7, label="I_after (native)")
        ax.set_ylabel("Current [A]")
        ax.legend(fontsize=8)
        ax.set_title("Native (filterexact)", fontsize=10)
        ax.grid(True, alpha=0.2)

        # 패널 2: I_des vs I_after (coupled)
        ax = axes[1]
        ax.plot(t_cpl / 1000, d_cpl["I_des"][:, ji], 'k--', lw=0.8, label="I_des")
        ax.plot(t_cpl / 1000, d_cpl["I_after"][:, ji], 'b-', lw=0.6, alpha=0.7, label="I_after (coupled)")
        ax.set_ylabel("Current [A]")
        ax.legend(fontsize=8)
        ax.set_title("Coupled (Schur complement)", fontsize=10)
        ax.grid(True, alpha=0.2)

        # 패널 3: 잔차 비교 overlay
        ax = axes[2]
        res_nat = d_nat["I_des"][:, ji] - d_nat["I_after"][:, ji]
        res_cpl = d_cpl["I_des"][:, ji] - d_cpl["I_after"][:, ji]
        ax.plot(t_nat / 1000, res_nat, 'r-', lw=0.5, alpha=0.6, label=f"native |res| rms={np.sqrt(np.mean(res_nat**2)):.4f}")
        ax.plot(t_cpl / 1000, res_cpl, 'b-', lw=0.5, alpha=0.6, label=f"coupled |res| rms={np.sqrt(np.mean(res_cpl**2)):.4f}")
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel("I_des - I_after [A]")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8)
        ax.set_title("Current residual comparison", fontsize=10)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        path = os.path.join(args.out, f"{jname}_current_residual_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

        # ── 2. 20초 ω, back-EMF, I 시계열 (native vs coupled overlay) ──
        has_bemf = "back_emf" in d_nat and "back_emf" in d_cpl
        nrows = 3 if has_bemf else 2

        fig, axes = plt.subplots(nrows, 1, figsize=(14, 3 * nrows), sharex=True)
        fig.suptitle(f"{jname}  —  20s time series: Native (red) vs Coupled (blue)", fontsize=12)

        # ω
        ax = axes[0]
        ax.plot(t_nat / 1000, d_nat["vel"][:, ji], 'r-', lw=0.5, alpha=0.7, label="native")
        ax.plot(t_cpl / 1000, d_cpl["vel"][:, ji], 'b-', lw=0.5, alpha=0.7, label="coupled")
        ax.set_ylabel("omega [rad/s]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # back-EMF
        if has_bemf:
            ax = axes[1]
            ax.plot(t_nat / 1000, d_nat["back_emf"][:, ji], 'r-', lw=0.5, alpha=0.7, label="native")
            ax.plot(t_cpl / 1000, d_cpl["back_emf"][:, ji], 'b-', lw=0.5, alpha=0.7, label="coupled")
            ax.set_ylabel("back-EMF [V]")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)

        # I
        ax = axes[-1]
        ax.plot(t_nat / 1000, d_nat["I_after"][:, ji], 'r-', lw=0.5, alpha=0.7, label="native")
        ax.plot(t_cpl / 1000, d_cpl["I_after"][:, ji], 'b-', lw=0.5, alpha=0.7, label="coupled")
        ax.set_ylabel("Current I [A]")
        ax.set_xlabel("Time [s]")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        plt.tight_layout()
        path = os.path.join(args.out, f"{jname}_20s_timeseries.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved: {path}")

    print("Done!")


if __name__ == "__main__":
    main()
