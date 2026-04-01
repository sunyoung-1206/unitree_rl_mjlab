"""Electric motor 전류·토크 추종 시각화.

정책 명령 주기(20ms)로 구분된 physics step 데이터를 수집하여
tau_des vs tau_applied, I_des vs I 추종 그래프를 출력한다.

사용법:
  python scripts/plot_electric_motor.py Unitree-Go2-Flat-Electric \
      --checkpoint-file logs/rsl_rl/.../model_XXXX.pt \
      --num-steps 50 \
      --joints FR_hip_joint,FR_thigh_joint,FR_calf_joint
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

matplotlib.use("Agg")  # headless


@dataclass
class PlotConfig:
    checkpoint_file: str
    """학습된 체크포인트 경로."""
    num_steps: int = 50
    """수집할 policy step 수 (× 4 = physics step 수, × 20ms = 실제 시간)."""
    joints: str | None = None
    """시각화할 관절 이름 (쉼표 구분). 예: FR_hip_joint,FR_thigh_joint,FR_calf_joint. 미지정 시 처음 3개."""
    physics_dt_ms: float = 5.0
    """physics step 간격 (ms). 기본 5ms."""
    decimation: int = 4
    """1 policy step당 physics step 수."""
    current_window_steps: int = 12
    """전류 subplot에 표시할 physics step 수. 기본 12 (= 60ms @ 5ms/step, ≈ 3 policy steps)."""
    vx: float | None = None
    """전진/후진 선속도 명령 [m/s]. 미지정 시 env 설정 범위에서 랜덤 샘플링."""
    vy: float | None = None
    """좌우 선속도 명령 [m/s]. 미지정 시 env 설정 범위에서 랜덤 샘플링."""
    wz: float | None = None
    """요(yaw) 각속도 명령 [rad/s]. 미지정 시 env 설정 범위에서 랜덤 샘플링."""
    tag: str = ""
    """파일명 앞에 붙는 태그. 비워두면 vx/vy/wz 값으로 자동 생성 (예: vx+0.5_vy+0.0_)."""
    out: str = "motor_tracking"
    """출력 루트 디렉토리. {out}/{joint_name}/{tag}1_position.png 형태로 저장."""
    plots: str = "1,2,3,4,5,6,7"
    """출력할 그래프 번호 (쉼표 구분). 예: 1,3,6 → position·torque substep·current substep만 저장."""
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_electric_actuators(env):
    """env에서 ElectricMotorActuator 인스턴스 목록 반환."""
    from src.assets.robots.unitree_go2.electric_actuator import ElectricMotorActuator

    entity = env.unwrapped.scene["robot"]
    result = []
    for act in entity._custom_actuators:
        if isinstance(act, ElectricMotorActuator):
            result.append(act)
    return result


def collect_data(task_id: str, cfg: PlotConfig) -> tuple[dict, list[str]]:
    """정책을 실행하며 physics step 데이터 수집."""
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    env_cfg.scene.num_envs = 1
    env_cfg.terminations = {}  # play 중 종료 없이 수집

    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
    runner.load(
        cfg.checkpoint_file,
        load_cfg={"actor": True},
        strict=True,
        map_location=cfg.device,
    )
    policy = runner.get_inference_policy(device=cfg.device)

    # 액추에이터 찾아서 로깅 시작
    actuators = get_electric_actuators(env)
    if not actuators:
        raise RuntimeError("ElectricMotorActuator를 env에서 찾을 수 없습니다.")
    for act in actuators:
        act.start_logging()

    # 관절 이름 수집 (첫 번째 actuator 기준)
    all_joint_names: list[str] = []
    for act in actuators:
        all_joint_names.extend(act.target_names)

    # 정책 실행
    obs, _ = env.reset()

    # 고정 속도 명령 설정 (reset 이후에 적용해야 덮어쓰이지 않음)
    if any(v is not None for v in (cfg.vx, cfg.vy, cfg.wz)):
        from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
        try:
            term = env.unwrapped.command_manager.get_term("twist")
            if isinstance(term, UniformVelocityCommand):
                vx = cfg.vx if cfg.vx is not None else 0.0
                vy = cfg.vy if cfg.vy is not None else 0.0
                wz = cfg.wz if cfg.wz is not None else 0.0
                term.vel_command_b[:] = torch.tensor([[vx, vy, wz]], device=cfg.device)
                term._resample_command = lambda env_ids: None
                print(f"[INFO] 고정 속도 명령: vx={vx:+.2f}  vy={vy:+.2f}  wz={wz:+.2f}")
        except Exception as e:
            print(f"[WARN] 속도 명령 고정 실패: {e}")

    for _ in range(cfg.num_steps):
        with torch.no_grad():
            action = policy(obs)
        obs, _, _, _ = env.step(action)

    # 로그 수집: 액추에이터별로 합산
    combined: dict[str, list] = {}
    physics_step_arr: np.ndarray | None = None

    for act in actuators:
        log = act.get_log()
        if not log or "tau_des" not in log:
            continue
        for k, v in log.items():
            if k == "physics_step":
                if physics_step_arr is None:
                    physics_step_arr = np.array(v)
            else:
                combined.setdefault(k, []).append(np.stack(v))  # (n, nj)

    # 각 key를 (n, total_joints) 배열로 병합 (액추에이터별 배열을 axis=1로 이어붙임)
    all_arrays: dict[str, np.ndarray] = {}
    for k in ("pos_target", "pos", "vel", "tau_des", "tau_applied", "I_des", "I_before", "I_after"):
        if k in combined:
            all_arrays[k] = np.concatenate(combined[k], axis=1)  # (n, total_joints)

    # I_substep / tau_substep: (n, n_sub, total_joints) — axis=2로 이어붙임
    for k in ("I_substep", "tau_substep"):
        if k in combined:
            all_arrays[k] = np.concatenate(combined[k], axis=2)

    # 모든 키가 compute()에서 동시에 기록되므로 길이 불일치는 없지만 안전 처리 유지
    n_min = min(len(v) for v in all_arrays.values())
    if physics_step_arr is not None:
        n_min = min(n_min, len(physics_step_arr))

    merged: dict[str, np.ndarray] = {
        "physics_step": (physics_step_arr[:n_min] if physics_step_arr is not None else np.arange(n_min))
    }
    for k, v in all_arrays.items():
        merged[k] = v[:n_min]

    env.close()
    return merged, all_joint_names


def _vlines(ax, t_ms, decimation, full=True):
    """physics step (silver) and policy step (gray) vertical lines."""
    if full:
        for t in t_ms:
            ax.axvline(t, color="silver", lw=0.5, alpha=0.5)
    for t in t_ms[::decimation]:
        ax.axvline(t, color="gray", lw=0.9, alpha=0.7)


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[INFO] saved: {path.resolve()}")


def plot(data: dict, joint_names: list[str], cfg: PlotConfig) -> None:
    """관절별 폴더에 개별 그래프 저장."""
    target_joints = cfg.joints.split(",") if cfg.joints else joint_names
    indices = [joint_names.index(j) for j in target_joints if j in joint_names]
    if not indices:
        print(f"[WARN] 요청한 관절 없음. 전체 관절: {joint_names}")
        indices = list(range(len(joint_names)))

    physics_dt = cfg.physics_dt_ms
    decimation  = cfg.decimation
    t_ms = data["physics_step"] * physics_dt

    has_substep = "I_substep" in data and "tau_substep" in data
    if has_substep:
        n_steps, n_sub, _ = data["I_substep"].shape
        dt_sub_ms = physics_dt / n_sub
        t_substep = np.concatenate([
            t_ms[i] + np.arange(1, n_sub + 1) * dt_sub_ms
            for i in range(n_steps)
        ])

    w = min(cfg.current_window_steps, len(t_ms))
    t_ms_w = t_ms[:w]
    if has_substep:
        t_sub_w        = t_substep[: w * n_sub]
        I_substep_w    = data["I_substep"][:w]
        tau_substep_w  = data["tau_substep"][:w]

    policy_step_ms = decimation * physics_dt
    sub_str = f"  sub-step={dt_sub_ms:.2f}ms" if has_substep else ""
    base_title = (
        f"policy={policy_step_ms:.0f}ms  physics={physics_dt:.0f}ms{sub_str}"
    )

    # 파일명 prefix 결정
    if cfg.tag:
        prefix = cfg.tag.rstrip("_") + "_"
    elif any(v is not None for v in (cfg.vx, cfg.vy, cfg.wz)):
        parts = []
        if cfg.vx is not None: parts.append(f"vx{cfg.vx:+.2f}")
        if cfg.vy is not None: parts.append(f"vy{cfg.vy:+.2f}")
        if cfg.wz is not None: parts.append(f"wz{cfg.wz:+.2f}")
        prefix = "_".join(parts) + "_"
    else:
        prefix = ""

    enabled = {int(x.strip()) for x in cfg.plots.split(",") if x.strip()}
    out_root = Path(cfg.out)

    for ji in indices:
        jname = joint_names[ji]
        jdir  = out_root / jname
        jdir.mkdir(parents=True, exist_ok=True)

        p = prefix  # 파일명 앞에 붙는 태그

        # 공통 스타일
        C_DES    = "black"
        C_TORQUE = "tab:red"
        C_CURR   = "tab:orange"
        C_RESID  = "tab:blue"
        A_ACT    = 0.55   # applied/actual 투명도

        # ── 1. position ──────────────────────────────────────────────────────
        if 1 in enabled:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.grid(False)
            ax.plot(t_ms, data["pos"][:, ji],        label="pos",        lw=1.2, color="tab:blue", zorder=2)
            ax.step(t_ms, data["pos_target"][:, ji], label="pos_target", lw=1.2, color=C_DES, where="post", zorder=3, ls="--")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("rad")
            ax.set_title(f"{jname}  position  |  {base_title}")
            ax.legend(fontsize=8)
            _vlines(ax, t_ms, decimation, full=False)
            _save(fig, jdir / f"{p}1_position.png")

        # ── 2. torque (sub-step resolution, full duration) ────────────────────
        if 2 in enabled:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.grid(False)
            if has_substep:
                tau_sub_full  = data["tau_substep"][:, :, ji].reshape(-1)
                tau_des_full  = np.repeat(data["tau_des"][:, ji], n_sub)
                ax.fill_between(t_substep, tau_des_full, tau_sub_full, alpha=0.15, color=C_TORQUE, label="_nolegend_")
                ax.plot(t_substep, tau_sub_full, label="tau_applied (Kt·I·gr)", lw=1.0, color=C_TORQUE, zorder=2)
                ax.plot(t_substep, tau_des_full, label="tau_des (PD)",          lw=1.0, color=C_DES,    zorder=3, ls="--")
            else:
                tau_d = data["tau_des"][:, ji]
                tau_a = data["tau_applied"][:, ji]
                ax.fill_between(t_ms, tau_d, tau_a, alpha=0.15, color=C_TORQUE, label="_nolegend_")
                ax.plot(t_ms, tau_a, label="tau_applied (Kt·I·gr)", lw=1.2, color=C_TORQUE, zorder=2)
                ax.plot(t_ms, tau_d, label="tau_des (PD)",          lw=1.2, color=C_DES,    zorder=3, ls="--")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("N·m")
            ax.set_title(f"{jname}  torque  |  {base_title}")
            ax.legend(fontsize=8)
            _vlines(ax, t_ms, decimation, full=False)
            _save(fig, jdir / f"{p}2_torque.png")

        # ── 3. torque sub-step (zoomed) ──────────────────────────────────────
        if 3 in enabled:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.grid(False)
            if has_substep:
                tau_sub_flat = tau_substep_w[:, :, ji].reshape(-1)
                tau_des_expanded = np.repeat(data["tau_des"][:w, ji], n_sub)
                ax.fill_between(t_sub_w, tau_des_expanded, tau_sub_flat,
                                alpha=0.15, color=C_TORQUE, label="_nolegend_")
                ax.plot(t_sub_w, tau_sub_flat,
                        label=f"tau sub-step (dt={dt_sub_ms:.2f}ms)", lw=1.2, color=C_TORQUE, zorder=2)
                ax.plot(t_sub_w, tau_des_expanded,
                        label="tau_des", lw=1.2, color=C_DES, zorder=3, ls="--")
            else:
                ax.step(t_ms_w, data["tau_des"][:w, ji],     label="tau_des",     lw=1.2, color=C_DES,    where="post", zorder=2)
                ax.plot(t_ms_w, data["tau_applied"][:w, ji], label="tau_applied", lw=1.2, color=C_TORQUE, zorder=3, alpha=A_ACT)
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("N·m")
            ax.set_title(f"{jname}  torque sub-step  |  {base_title}")
            ax.legend(fontsize=8)
            _vlines(ax, t_ms_w, decimation, full=True)
            _save(fig, jdir / f"{p}3_torque_substep.png")

        # ── 4. torque residual (full duration) ───────────────────────────────
        if 4 in enabled:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.grid(False)
            tau_res = data["tau_des"][:, ji] - data["tau_applied"][:, ji]
            ax.fill_between(t_ms, tau_res, 0, alpha=0.2, color=C_RESID, label="_nolegend_")
            ax.plot(t_ms, tau_res, lw=1.0, color=C_RESID, label="tau_des - tau_applied")
            ax.axhline(0, color="black", lw=0.6, alpha=0.4)
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("N·m")
            ax.set_title(f"{jname}  torque residual  |  {base_title}")
            ax.legend(fontsize=8)
            _vlines(ax, t_ms, decimation, full=False)
            _save(fig, jdir / f"{p}4_torque_residual.png")

        # ── 5. current (sub-step resolution, full duration) ───────────────────
        if 5 in enabled:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.grid(False)
            if has_substep:
                I_sub_full  = data["I_substep"][:, :, ji].reshape(-1)
                I_des_full  = np.repeat(data["I_des"][:, ji], n_sub)
                ax.fill_between(t_substep, I_des_full, I_sub_full, alpha=0.15, color=C_CURR, label="_nolegend_")
                ax.plot(t_substep, I_sub_full, label="I_after", lw=1.0, color=C_CURR, zorder=2)
                ax.plot(t_substep, I_des_full, label="I_des",   lw=1.0, color=C_DES,  zorder=3, ls="--")
            else:
                I_d = data["I_des"][:, ji]
                I_a = data["I_after"][:, ji]
                ax.fill_between(t_ms, I_d, I_a, alpha=0.15, color=C_CURR, label="_nolegend_")
                ax.plot(t_ms, I_a, label="I_after", lw=1.2, color=C_CURR, zorder=2)
                ax.plot(t_ms, I_d, label="I_des",   lw=1.2, color=C_DES,  zorder=3, ls="--")
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("A")
            ax.set_title(f"{jname}  current  |  {base_title}")
            ax.legend(fontsize=8)
            _vlines(ax, t_ms, decimation, full=False)
            _save(fig, jdir / f"{p}5_current.png")

        # ── 6. current sub-step (zoomed) ─────────────────────────────────────
        if 6 in enabled:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.grid(False)
            if has_substep:
                I_sub_flat = I_substep_w[:, :, ji].reshape(-1)
                I_des_expanded = np.repeat(data["I_des"][:w, ji], n_sub)
                ax.fill_between(t_sub_w, I_des_expanded, I_sub_flat,
                                alpha=0.15, color=C_CURR, label="_nolegend_")
                ax.plot(t_sub_w, I_sub_flat,
                        label=f"I sub-step (dt={dt_sub_ms:.2f}ms)", lw=1.2, color=C_CURR, zorder=2)
                ax.plot(t_sub_w, I_des_expanded,
                        label="I_des", lw=1.2, color=C_DES, zorder=3, ls="--")
            else:
                ax.step(t_ms_w, data["I_des"][:w, ji],   label="I_des",   lw=1.2, color=C_DES,  where="post", zorder=2)
                ax.plot(t_ms_w, data["I_after"][:w, ji], label="I_after", lw=1.2, color=C_CURR, zorder=3, alpha=A_ACT)
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("A")
            ax.set_title(f"{jname}  current sub-step  |  {base_title}")
            ax.legend(fontsize=8)
            _vlines(ax, t_ms_w, decimation, full=True)
            _save(fig, jdir / f"{p}6_current_substep.png")

        # ── 7. current residual (full duration) ──────────────────────────────
        if 7 in enabled:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.grid(False)
            I_res = data["I_des"][:, ji] - data["I_after"][:, ji]
            ax.fill_between(t_ms, I_res, 0, alpha=0.2, color=C_RESID, label="_nolegend_")
            ax.plot(t_ms, I_res, lw=1.0, color=C_RESID, label="I_des - I_after")
            ax.axhline(0, color="black", lw=0.6, alpha=0.4)
            ax.set_xlabel("time (ms)")
            ax.set_ylabel("A")
            ax.set_title(f"{jname}  current residual  |  {base_title}")
            ax.legend(fontsize=8)
            _vlines(ax, t_ms, decimation, full=False)
            _save(fig, jdir / f"{p}7_current_residual.png")

        saved = sorted(enabled)
        print(f"[INFO] {jname}: saved plots {saved} → {jdir.resolve()}")


def main():
    import mjlab  # noqa: F401

    all_tasks_raw, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(["Unitree-Go2-Flat-Electric"]),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    task_id = all_tasks_raw

    cfg = tyro.cli(PlotConfig, args=remaining)

    data, joint_names = collect_data(task_id, cfg)
    plot(data, joint_names, cfg)


if __name__ == "__main__":
    # 직접 task_id를 넘겨 쓸 수 있도록 간단한 인터페이스 제공
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401

    from mjlab.tasks.registry import list_tasks

    if len(sys.argv) < 2 or sys.argv[1] not in list_tasks():
        print("사용법: python scripts/plot_electric_motor.py Unitree-Go2-Flat-Electric --checkpoint-file <path>")
        sys.exit(1)

    task_id = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    import mjlab

    cfg = tyro.cli(PlotConfig, config=mjlab.TYRO_FLAGS)
    data, joint_names = collect_data(task_id, cfg)
    plot(data, joint_names, cfg)
