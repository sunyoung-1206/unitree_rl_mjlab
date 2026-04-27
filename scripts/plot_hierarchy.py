"""Multi-panel hierarchy plots with ω_des=0 reference line."""
import sys, os, glob
os.chdir("/home/rbdo/unitree_rl_mjlab")
sys.path.insert(0, ".")

import mjlab; import mjlab.tasks; import src.tasks
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tyro, torch, mujoco
from dataclasses import asdict
from datetime import datetime
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

configure_torch_backends()
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = "motor_tracking/FL_calf_joint"
os.makedirs(outdir, exist_ok=True)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

T_WINDOW = (0, 120)


def grid_lines(ax, t0, t1, policy_ms=20, pd_ms=5, show_pd=True):
    for t in range(t0, t1+1, policy_ms):
        ax.axvline(t, color='tab:red', lw=1.2, alpha=0.5, ls='--')
    if show_pd:
        for t in range(t0, t1+1, pd_ms):
            if t % policy_ms != 0:
                ax.axvline(t, color='gray', lw=0.5, alpha=0.3, ls=':')


def find_ji(name, names):
    for i, n in enumerate(names):
        if n == name or n.endswith("/" + name):
            return i
    return None


# ── PD collection ─────────────────────────────────────────────
print("=== PD: collecting at 5ms resolution ===")
task_pd = "Unitree-Go2-Flat"
ckpt_pd = glob.glob("logs/rsl_rl/pd_policy20ms_physics5ms/*/model_1999.pt")[0]

env_cfg = load_env_cfg(task_pd, play=True)
agent_cfg = load_rl_cfg(task_pd)
env_cfg.scene.num_envs = 1
env_cfg.terminations = {}
env_cfg.decimation = 1
pdt_pd = env_cfg.sim.mujoco.timestep * 1e3

env_pd = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
wrapped_pd = RslRlVecEnvWrapper(env_pd, clip_actions=agent_cfg.clip_actions)
runner_cls = load_runner_cls(task_pd) or MjlabOnPolicyRunner
runner = runner_cls(wrapped_pd, asdict(agent_cfg), device=device)
runner.load(ckpt_pd, load_cfg={"actor": True}, strict=True, map_location=device)
policy_pd = runner.get_inference_policy(device=device)

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
try:
    term = env_pd.command_manager.get_term("twist")
    if isinstance(term, UniformVelocityCommand):
        term.vel_command_b[:] = torch.tensor([[0.5, 0.0, 0.0]], device=device)
        term._resample_command = lambda env_ids: None
except: pass

obs_pd, _ = wrapped_pd.reset()

mj_model = env_pd.sim.mj_model
jnames_pd, qpos_addrs, dof_addrs = [], [], []
for i in range(mj_model.nu):
    jid = mj_model.actuator_trnid[i, 0]
    jnames_pd.append(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jid))
    qpos_addrs.append(mj_model.jnt_qposadr[jid])
    dof_addrs.append(mj_model.jnt_dofadr[jid])

ji_pd = find_ji("FL_calf_joint", jnames_pd)

rec_pd = {"t": [], "q": [], "omega": [], "tau": [], "q_des": []}
for ps in range(30):
    with torch.no_grad():
        current_action = policy_pd(obs_pd)
    for sub in range(4):
        obs_pd, _, _, _ = wrapped_pd.step(current_action)
        step_idx = ps * 4 + sub
        t_ms = step_idx * pdt_pd
        wp = env_pd.sim.data.struct
        qp, qv, qf = wp.qpos.numpy()[0], wp.qvel.numpy()[0], wp.qfrc_actuator.numpy()[0]
        rec_pd["t"].append(t_ms)
        rec_pd["q"].append(qp[qpos_addrs[ji_pd]])
        rec_pd["omega"].append(qv[dof_addrs[ji_pd]])
        rec_pd["tau"].append(qf[dof_addrs[ji_pd]])
        robot = env_pd.scene["robot"]
        rec_pd["q_des"].append(float(robot.data.joint_pos_target[0, ji_pd].cpu()))

wrapped_pd.close()
for k in rec_pd:
    rec_pd[k] = np.array(rec_pd[k])

# ── Method A collection ────────────────────────────────────────
print("=== Method A: collecting at 0.1ms resolution ===")
from scripts.plot_electric_motor import PlotConfig, collect_data

ckpt_a = glob.glob("logs/rsl_rl/methodA_policy20ms_physics0.1ms/*/model_1999.pt")[0]
sys.argv = ['x', '--checkpoint-file', ckpt_a,
  '--num-steps', '30', '--joints', 'FL_calf_joint',
  '--vx', '0.5', '--vy', '0.0', '--wz', '0.0', '--tag', 'x_']
cfg = tyro.cli(PlotConfig, config=mjlab.TYRO_FLAGS)
data_a, jn_a, pdt_a, dec_a = collect_data('Unitree-Go2-Flat-MethodA-Electric', cfg)

ji_a = jn_a.index('FL_calf_joint')
t_ms_a = data_a['physics_step'] * pdt_a


# ── Plot PD ────────────────────────────────────────────────────
print("=== Plotting PD ===")
t0, t1 = T_WINDOW
m_pd = (rec_pd['t'] >= t0) & (rec_pd['t'] <= t1)

fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)

ax = axes[0]
ax.step(rec_pd['t'][m_pd], rec_pd['q_des'][m_pd], where='post', lw=1.8,
        color='black', ls='--', label='q_des (policy, 20ms update)')
ax.plot(rec_pd['t'][m_pd], rec_pd['q'][m_pd], lw=1.2,
        color='tab:blue', label='q (actual, 5ms)')
grid_lines(ax, t0, t1, show_pd=False)
ax.set_ylabel('rad'); ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)
ax.set_title('FL_calf_joint — PD Torque Control | policy=20ms, physics=5ms')

ax = axes[1]
ax.axhline(0, color='black', ls='--', lw=1.5, label='ω_des = 0 (always)', alpha=0.7)
ax.plot(rec_pd['t'][m_pd], rec_pd['omega'][m_pd], lw=1.2,
        color='tab:blue', label='ω (actual, 5ms)')
grid_lines(ax, t0, t1, show_pd=False)
ax.set_ylabel('rad/s'); ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)

ax = axes[2]
ax.step(rec_pd['t'][m_pd], rec_pd['tau'][m_pd], where='post', lw=1.5,
        color='tab:red', label='τ = kp(q_des−q) − kd·ω  (5ms update)')
grid_lines(ax, t0, t1, show_pd=False)
ax.set_xlabel('time (ms)'); ax.set_ylabel('N·m')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)

fig.tight_layout()
f_pd = f"{outdir}/FL_calf_hierarchy_PD_{ts}.png"
fig.savefig(f_pd, dpi=150); plt.close(fig)
print(f"Saved: {f_pd}")


# ── Plot Method A ──────────────────────────────────────────────
print("=== Plotting Method A ===")
m_a = (t_ms_a >= t0) & (t_ms_a <= t1)

fig, axes = plt.subplots(5, 1, figsize=(15, 15), sharex=True)

ax = axes[0]
ax.step(t_ms_a[m_a], data_a['pos_target'][m_a, ji_a], where='post', lw=1.8,
        color='black', ls='--', label='q_des (policy, 20ms update)')
ax.plot(t_ms_a[m_a], data_a['pos'][m_a, ji_a], lw=1.0,
        color='tab:blue', label='q (actual, 0.1ms)')
grid_lines(ax, t0, t1)
ax.set_ylabel('rad'); ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)
ax.set_title('FL_calf_joint — Method A (coupled Schur) | policy=20ms, PD=5ms, physics=0.1ms')

ax = axes[1]
ax.axhline(0, color='black', ls='--', lw=1.5, label='ω_des = 0 (always)', alpha=0.7)
ax.plot(t_ms_a[m_a], data_a['vel'][m_a, ji_a], lw=1.0,
        color='tab:blue', label='ω (coupled implicit solve, 0.1ms)')
grid_lines(ax, t0, t1)
ax.set_ylabel('rad/s'); ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)

ax = axes[2]
ax.step(t_ms_a[m_a], data_a['tau_des'][m_a, ji_a], where='post', lw=1.5,
        color='black', ls='--', label='τ_des (PD, 5ms recompute)')
ax.plot(t_ms_a[m_a], data_a['tau_applied'][m_a, ji_a], lw=1.0,
        color='tab:red', alpha=0.8, label='τ_applied = Kt·gr·I (0.1ms)')
grid_lines(ax, t0, t1)
ax.set_ylabel('N·m'); ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)

ax = axes[3]
ax.step(t_ms_a[m_a], data_a['I_des'][m_a, ji_a], where='post', lw=1.5,
        color='black', ls='--', label='I_des = τ_des/(Kt·gr) (5ms update)')
ax.plot(t_ms_a[m_a], data_a['I_after'][m_a, ji_a], lw=1.0,
        color='tab:orange', alpha=0.8, label='I (filterexact, 0.1ms)')
grid_lines(ax, t0, t1)
ax.set_ylabel('A'); ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)

ax = axes[4]
if 'V' in data_a:
    ax.plot(t_ms_a[m_a], data_a['V'][m_a, ji_a], lw=1.0,
            color='tab:purple', label='V = R·I_des + Ke·gr·ω')
if 'back_emf' in data_a:
    ax.plot(t_ms_a[m_a], data_a['back_emf'][m_a, ji_a], lw=0.8,
            color='tab:cyan', alpha=0.7, label='back-EMF = Ke·gr·ω')
grid_lines(ax, t0, t1)
ax.set_xlabel('time (ms)'); ax.set_ylabel('V')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)

fig.tight_layout()
f_a = f"{outdir}/FL_calf_hierarchy_methodA_{ts}.png"
fig.savefig(f_a, dpi=150); plt.close(fig)
print(f"Saved: {f_a}")

print("Done.")
