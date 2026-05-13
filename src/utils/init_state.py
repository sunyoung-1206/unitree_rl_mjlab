"""Zero-initial-state helper for play / eval scripts.

추론 시 모든 환경을 결정론적 초기조건(좌표 0, 높이는 init_state.pos[2],
quaternion identity, 모든 속도 0, joint=default_joint_pos)에서 출발시킨다.

mjlab `reset_event` term 이 base yaw 를 무작위 샘플링 (-π, π) 하기 때문에,
seed 가 같아도 reset 마다 시작 yaw 가 다른 문제가 발생한다. 본 헬퍼는
`ManagerBasedRlEnv._reset_idx` 를 래핑해 모든 reset 직후 결정론적 상태를
강제하고, 호출 시점에도 즉시 1회 적용한다.

사용 예:
    from src.utils.init_state import apply_zero_initial_state
    env = ManagerBasedRlEnv(cfg=env_cfg, ...)
    apply_zero_initial_state(env)          # ← 한 번만 호출
    # 이후 env.reset() 가 어디서 일어나든 자동 적용.
"""

from __future__ import annotations

import numpy as np
import torch


def _unwrap_env(env):
    """RslRlVecEnvWrapper 등 wrapper 를 벗긴 ManagerBasedRlEnv 반환."""
    inner = env
    while hasattr(inner, "unwrapped") and inner.unwrapped is not inner:
        inner = inner.unwrapped
    return inner


def _write_zero_state(inner_env, env_ids=None, verbose: bool = False) -> None:
    """단일 호출로 robot base / joint 상태를 결정론적 0 상태로 write.

    Args:
        inner_env: ManagerBasedRlEnv (unwrapped).
        env_ids: None 이면 전체. tensor / slice 가능.
        verbose: True 면 env0 의 적용 상태 print.
    """
    robot = inner_env.scene["robot"]
    rs = robot.data.default_root_state.clone()
    rs[:, 0] = 0.0          # x
    rs[:, 1] = 0.0          # y
    # rs[:, 2] = standing height (init_state.pos[2]) 는 그대로 보존
    rs[:, 3] = 1.0          # qw
    rs[:, 4:7] = 0.0        # qx, qy, qz → quat = (1, 0, 0, 0)
    rs[:, 7:13] = 0.0       # lin_vel + ang_vel
    robot.write_root_state_to_sim(rs, env_ids)

    jp = robot.data.default_joint_pos.clone()
    jv = torch.zeros_like(jp)
    robot.write_joint_state_to_sim(jp, jv, env_ids=env_ids)

    if verbose:
        r0 = rs[0].detach().cpu().numpy()
        j0 = jp[0].detach().cpu().numpy()
        yaw = float(np.arctan2(2.0 * (r0[3] * r0[6] + r0[4] * r0[5]),
                                1.0 - 2.0 * (r0[5] * r0[5] + r0[6] * r0[6])))
        print(f"[ZERO-INIT] env0  pos=({r0[0]:+.3f}, {r0[1]:+.3f}, {r0[2]:+.3f})"
              f"  quat=({r0[3]:+.3f}, {r0[4]:+.3f}, {r0[5]:+.3f}, {r0[6]:+.3f})"
              f"  yaw={yaw:+.4f} rad")
        print(f"[ZERO-INIT] env0  lin_vel=({r0[7]:+.2f},{r0[8]:+.2f},{r0[9]:+.2f})"
              f"  ang_vel=({r0[10]:+.2f},{r0[11]:+.2f},{r0[12]:+.2f})")
        print(f"[ZERO-INIT] env0  joint_pos[:4]={np.round(j0[:4], 3).tolist()}"
              f"  joint_vel=0 (all)")


def apply_zero_initial_state(env, verbose: bool = True) -> bool:
    """모든 env 의 초기조건을 결정론적 0 상태로 강제.

    동작:
        1) 즉시 1회 적용 — env 가 이미 reset 된 상태라면 그 결과를 덮어쓴다.
        2) `ManagerBasedRlEnv._reset_idx` 래핑 — 이후 모든 reset 직후 자동 적용.
           viewer.run() 안에서 reset 이 일어나든, wrapped.reset() 이 명시 호출되든
           동일하게 결정론 상태 유지.

    중복 호출 시 패치는 1회만 적용 (멱등).

    Returns:
        True 면 패치 성공 (또는 이미 적용됨). 환경 구조가 예상과 다르면 False.
    """
    inner = _unwrap_env(env)
    if not hasattr(inner, "_reset_idx"):
        print("[WARN] zero_init: ManagerBasedRlEnv._reset_idx 를 찾지 못함 → no-op")
        return False
    try:
        _ = inner.scene["robot"]
    except Exception:
        print("[WARN] zero_init: scene['robot'] 접근 실패 → no-op")
        return False

    # 1) 즉시 1회 적용 (env 가 이미 reset 된 경우 시작 상태를 덮어쓰기 위해)
    _write_zero_state(inner, env_ids=None, verbose=verbose)

    # 2) 멱등 패치 — 이후 reset 마다 자동 적용
    if getattr(inner, "_zero_init_patched", False):
        return True
    original_reset_idx = inner._reset_idx

    def patched_reset_idx(env_ids=None):
        original_reset_idx(env_ids)
        # reset 후 자동 zero 상태 (verbose 는 첫 1회만 — 즉시 적용 시 이미 출력함)
        _write_zero_state(inner, env_ids=env_ids, verbose=False)

    inner._reset_idx = patched_reset_idx
    inner._zero_init_patched = True
    return True
