# Unitree-Go2-Flat: Actor / Critic MLP 입력 구성

`Unitree-Go2-Flat` (및 `Unitree-Go2-Flat-NoClip`) 정책의 actor / critic 신경망
입력 벡터 구성입니다. 코드로 직접 검증한 값입니다 (2026-05-20, `num_envs=1`,
`play=True`로 `ManagerBasedRlEnv` 인스턴스 생성 후 `observation_manager` 덤프).

- **Actor (policy) 입력: 47차원**
- **Critic (value) 입력: 74차원**

네트워크 구조는 양쪽 모두 MLP `(512, 256, 128)`, activation `elu`,
`obs_normalization=True` (`rl_cfg.py:16-30`).

정의 위치: `src/tasks/velocity/velocity_env_cfg.py:58-136` (actor/critic term + group),
flat 변형에서 `height_scan` 제거: `src/tasks/velocity/config/go2/env_cfgs.py:411-416`.

---

## 1. Actor 입력 (47차원)

그룹 설정: `enable_corruption=True`, `history_length=1` (velocity_env_cfg.py:124-129).
→ 매 step 1 프레임만, **노이즈 적용됨** (sim2real 강건성).

| # | Term | 차원 | 누적 | 함수 | 노이즈 (Uniform) | 설명 |
|---|---|---|---|---|---|---|
| 1 | `base_ang_vel` | 3 | 0:3 | `mdp.builtin_sensor` (`robot/imu_ang_vel`) | ±0.2 rad/s | IMU 자이로 각속도 (body frame) |
| 2 | `projected_gravity` | 3 | 3:6 | `mdp.projected_gravity` | ±0.05 | 중력벡터를 base frame에 투영 (자세 정보) |
| 3 | `command` | 3 | 6:9 | `mdp.generated_commands` (`twist`) | — | 속도 명령 [v_x, v_y, ω_z] |
| 4 | `phase` | 2 | 9:11 | `mdp.phase` (period=0.6) | — | 보행 위상 [sin(2πφ), cos(2πφ)], 정지 시 0 |
| 5 | `joint_pos` | 12 | 11:23 | `mdp.joint_pos_rel` | ±0.01 rad | 관절각 − default (4 leg × 3 joint) |
| 6 | `joint_vel` | 12 | 23:35 | `mdp.joint_vel_rel` | ±1.5 rad/s | 관절 각속도 |
| 7 | `actions` | 12 | 35:47 | `mdp.last_action` | — | 직전 step의 정책 출력 |

**합계: 3+3+3+2+12+12+12 = 47**

### 항목별 비고

- **`phase` (2)**: `(episode_length × step_dt) mod 0.6 / 0.6` 의 위상을 sin/cos로
  인코딩. 명령 norm < 0.1 (정지)이면 `[0, 0]` 으로 마스킹 (observations.py:47-54).
  보행 리듬 가이드 신호 — `foot_gait` reward와 같은 period=0.6 사용.
- **`command` (3)**: heading_command=True 여도 관측은 3차원
  ([v_x, v_y, ω_z]). heading은 내부에서 ω_z로 변환되어 명령에 반영됨.
- **`joint_pos` (12)**: default_joint_pos 기준 상대값 (`*_rel`).
- 노이즈는 `enable_corruption=True` 라서 actor에만 적용. critic은 비적용 (아래).

---

## 2. Critic 입력 (74차원)

그룹 설정: `enable_corruption=False`, `history_length=1` (velocity_env_cfg.py:130-135).
→ **노이즈 미적용** (privileged value 추정용 깨끗한 관측).

Actor의 7개 term을 그대로 포함(노이즈만 빠짐) + privileged 5개 추가:

| # | Term | 차원 | 누적 | 함수 | 설명 |
|---|---|---|---|---|---|
| 1–7 | (actor와 동일) | 47 | 0:47 | — | base_ang_vel, projected_gravity, command, phase, joint_pos, joint_vel, actions |
| 8 | `base_lin_vel` | 3 | 47:50 | `mdp.builtin_sensor` (`robot/imu_lin_vel`) | base 선속도 (privileged — 실기 측정 어려움) |
| 9 | `foot_height` | 4 | 50:54 | `mdp.foot_height` | 4발 site의 world z 높이 |
| 10 | `foot_air_time` | 4 | 54:58 | `mdp.foot_air_time` (`feet_ground_contact`) | 4발의 현재 공중 체류 시간 [s] |
| 11 | `foot_contact` | 4 | 58:62 | `mdp.foot_contact` (`feet_ground_contact`) | 4발 접촉 여부 (0/1) |
| 12 | `foot_contact_forces` | 12 | 62:74 | `mdp.foot_contact_forces` (`feet_ground_contact`) | 4발 × 3축(xyz) 접촉력, sign·log1p 압축 |

**합계: 47 + 3 + 4 + 4 + 4 + 12 = 74**

### Privileged term 비고

- **`base_lin_vel` (3)**: 실 로봇에서 직접 측정이 어려운 base 선속도. critic만
  사용 (asymmetric actor-critic). actor에는 없음.
- **`foot_contact_forces` (12)**: `force.flatten` → [B, 4×3], `sign(f)·log1p(|f|)`
  로 큰 충격력을 압축 (observations.py:39-44). 발당 3축(x,y,z).
- 발 순서는 `("FR", "FL", "RR", "RL")` (env_cfgs.py:48-50).

---

## 3. Actor vs Critic 차이 요약

| 구분 | Actor | Critic |
|---|---|---|
| 입력 차원 | 47 | 74 |
| 노이즈 (corruption) | 적용 (±값) | 미적용 |
| privileged 관측 | 없음 | base_lin_vel + foot 4종 (27차원) |
| 용도 | 배포(deploy) 정책 | 학습용 value 추정 (asymmetric) |

Asymmetric actor-critic: critic은 학습 중에만 쓰는 privileged 정보(실 속도,
발 접촉력/높이/공중시간)를 추가로 받아 더 정확한 value를 추정하고, actor는
실기에서 얻을 수 있는 47차원만 받아 배포 가능.

---

## 4. 다른 task와의 차이

| Task | actor 차원 | critic 차원 | 차이 원인 |
|---|---|---|---|
| **Unitree-Go2-Flat / -NoClip** | **47** | **74** | 본 문서 |
| Unitree-Go2-Rough | 47 + 187 = 234 | 74 + 187 = 261 | `height_scan` 187차원(terrain_scan 1.6×1.0 / 0.1 grid) 추가 |
| Unitree-Go2-Flat-MethodA-Electric | 47 × 5 = **235** | 74 | actor `history_length=5` + per-term delay 0~4 step (env_cfgs.py:203-208). critic은 history=1 유지 |
| -MethodA-Electric-Sim2Real | 235 | 74 | MA-P1과 동일 obs 구조 |

> ⚠️ MethodA-Electric 계열은 actor만 history=5로 쌓아 235차원
> (47 × 5). critic은 history=1 그대로 74. flat baseline (본 문서)은 actor도
> history=1 → 47.

---

## 5. 재현 방법

```python
import torch
import mjlab.tasks.registry as reg
reg._REGISTRY.clear()
from src.tasks.velocity.config.go2 import __init__  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

env_cfg = load_env_cfg("Unitree-Go2-Flat", play=True)
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)

om = env.observation_manager
for group in ("actor", "critic"):
    print(group, "terms:", om.active_terms[group])
    print("  dims:", om.group_obs_term_dim[group])
```
