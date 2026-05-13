"""Shared heading-lock helper for play / eval scripts.

`UniformVelocityCommand.heading_command=True` 인 학습 분포에 대응되는 추론용
래퍼. 매 step 다음을 수행한 뒤 원본 `_update_command` 를 호출해 라이브러리가
`wz = K_h * wrap_to_pi(target - heading_w)` 를 직접 계산하도록 위임한다:

  term.vel_command_b[:, 0] = vx
  term.vel_command_b[:, 1] = vy
  term.is_heading_env[:]   = True
  term.heading_target[:]   = stored_target
  term.is_standing_env[:]  = False

`scripts/play.py` 와 `results/.../run_demag_experiment.py` 가 이 헬퍼를 공유한다.
"""

from __future__ import annotations

import torch


def apply_heading_lock_velocity(
    env,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    target_heading: float | None = None,
    heading_threshold: float = 0.1,
    no_heading_control: bool = False,
    debug_cmd: bool = False,
    device: str | None = None,
) -> bool:
    """Twist 커맨드에 vx/vy/wz 를 박고 wz 크기에 따라 heading-lock / manual 자동 전환.

    Args:
        env: ManagerBasedRlEnv (`env.command_manager.get_term("twist")` 가
            `UniformVelocityCommand` 여야 함).
        vx, vy: 선속도 [m/s].
        wz: 요 각속도 [rad/s]. `|wz| < heading_threshold` 면 heading-lock 모드.
        target_heading: 명시적 target [rad]. None 이면 첫 step 의 base yaw 자동 캡처.
        heading_threshold: heading-lock ↔ manual 전환 임계.
        no_heading_control: True 면 mode 분기 없이 wz 를 그대로 박음 (라이브러리 P 제어 우회).
        debug_cmd: 50 step 마다 mode / cmd / yaw 상태 stdout.
        device: torch device. None 이면 robot.data.heading_w 와 동일 device 사용.

    Returns:
        True 면 패치 성공. twist term 이 없거나 타입이 안 맞으면 False.
    """
    from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

    try:
        term = env.command_manager.get_term("twist")
    except Exception:
        print("[WARN] heading_lock: 'twist' term 을 찾을 수 없음 → no-op")
        return False

    if not isinstance(term, UniformVelocityCommand):
        print("[WARN] heading_lock: 'twist' term 이 UniformVelocityCommand 아님 → no-op")
        return False

    if device is None:
        device = term.robot.data.heading_w.device

    vx_user = float(vx)
    vy_user = float(vy)
    wz_user = float(wz)
    threshold = float(heading_threshold)

    # 랜덤 재샘플링 차단 — is_heading_env / heading_target 이 의도치 않게 바뀌지 않도록.
    term._resample_command = lambda env_ids: None

    # --- 분기 A: no-heading-control (기존 동작 보존, 비교 실험용) ---
    if no_heading_control:
        fixed = torch.tensor([[vx_user, vy_user, wz_user]], device=device)
        term.vel_command_b[:] = fixed
        term.is_heading_env[:] = False
        term.is_standing_env[:] = False

        state_nh: dict = {"step": 0}

        def patched_update_no_heading() -> None:
            term.vel_command_b[:] = fixed
            term.is_heading_env[:] = False
            term.is_standing_env[:] = False
            if debug_cmd:
                state_nh["step"] += 1
                if state_nh["step"] % 50 == 1:
                    cmd = term.vel_command_b[0].detach().cpu().tolist()
                    yaw_cur = float(term.robot.data.heading_w[0].item())
                    print(
                        f"[DBG step={state_nh['step']:5d}] mode=no_heading      "
                        f"cmd=[{cmd[0]:+.3f} {cmd[1]:+.3f} {cmd[2]:+.3f}] "
                        f"is_heading=False target_yaw=  N/A  cur_yaw={yaw_cur:+.3f}"
                    )

        term._update_command = patched_update_no_heading
        print(
            f"[INFO] heading_lock(no_heading_control): vx={vx_user:+.2f} "
            f"vy={vy_user:+.2f} wz={wz_user:+.2f} | debug={debug_cmd}"
        )
        return True

    # --- 분기 B: heading-lock 모드 ---
    num_envs = term.num_envs
    original_update = term._update_command

    if target_heading is not None:
        stored_target_init = torch.full(
            (num_envs,), float(target_heading), device=device
        )
        target_strategy = f"explicit ({target_heading:+.3f} rad)"
    else:
        stored_target_init = None
        target_strategy = "auto (첫 step base yaw)"

    state: dict = {
        "prev_mode": None,
        "stored_target": stored_target_init,
        "step": 0,
    }

    def patched_update() -> None:
        if state["stored_target"] is None:
            state["stored_target"] = term.robot.data.heading_w.detach().clone()

        is_manual = abs(wz_user) >= threshold
        new_mode = "manual" if is_manual else "heading_control"

        # Manual → heading-control 전환 시 현재 yaw 를 새 target 으로 갱신.
        if state["prev_mode"] == "manual" and new_mode == "heading_control":
            state["stored_target"] = term.robot.data.heading_w.detach().clone()
        state["prev_mode"] = new_mode

        term.vel_command_b[:, 0] = vx_user
        term.vel_command_b[:, 1] = vy_user
        term.is_standing_env[:] = False

        if is_manual:
            term.is_heading_env[:] = False
            term.vel_command_b[:, 2] = wz_user
        else:
            term.is_heading_env[:] = True
            term.heading_target[:] = state["stored_target"]

        # 원본 _update_command — is_heading_env=True 인 env 의 wz 를
        # K_h * wrap_to_pi(target - heading_w) 로 라이브러리가 덮어쓴다.
        original_update()

        if debug_cmd:
            state["step"] += 1
            if state["step"] % 50 == 1:
                cmd = term.vel_command_b[0].detach().cpu().tolist()
                yaw_cur = float(term.robot.data.heading_w[0].item())
                yaw_tgt = float(state["stored_target"][0].item())
                is_h = bool(term.is_heading_env[0].item())
                print(
                    f"[DBG step={state['step']:5d}] mode={new_mode:14s} "
                    f"cmd=[{cmd[0]:+.3f} {cmd[1]:+.3f} {cmd[2]:+.3f}] "
                    f"is_heading={is_h!s:5s} target_yaw={yaw_tgt:+.3f} "
                    f"cur_yaw={yaw_cur:+.3f}"
                )

    term._update_command = patched_update

    print(
        f"[INFO] heading_lock: vx={vx_user:+.2f} vy={vy_user:+.2f} wz={wz_user:+.2f}"
    )
    print(
        f"       threshold={threshold:.3f} rad/s | target heading: "
        f"{target_strategy} | debug={debug_cmd}"
    )
    return True
