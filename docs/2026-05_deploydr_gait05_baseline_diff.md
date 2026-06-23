# `Unitree-Go2-Flat-DeployDR-Gait05-v0` baseline diff 분석

- 작성일: 2026-05-27
- 분석 대상 task: `Unitree-Go2-Flat-DeployDR-Gait05-v0`
- 비교 baseline: `Unitree-Go2-Flat` (= `_go2_flat_pd_cfg` = vanilla DR + 5 ms PD sim)
- 분석 원칙: 코드를 직접 읽고 baseline ↔ DeployDR 차이만 표기. 추정 금지.

---

## 0. Registration & Baseline 식별

**Registration**: `src/tasks/velocity/config/go2/__init__.py:210-216`
```python
register_mjlab_task(
  task_id="Unitree-Go2-Flat-DeployDR-Gait05-v0",
  env_cfg=_go2_flat_deploydr_gait05_cfg(),
  play_env_cfg=_go2_flat_deploydr_gait05_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),         # PPO 공용 cfg (run_name override 없음)
  runner_cls=VelocityOnPolicyRunner,
)
```

**클래스(함수) 계층** (factory 호출 chain):

```
_go2_flat_deploydr_gait05_cfg(play)              ← Gait05 본인 (line 205-208)
  └─ _go2_flat_deploydr_cfg(play)                ← DeployDR-v0 와 100% 동일 (line 61-191)
       └─ _go2_flat_pd_cfg(play)                 ← Unitree-Go2-Flat (line 36-42)
            └─ unitree_go2_flat_env_cfg(play)    ← env_cfgs.py:397
                 └─ unitree_go2_rough_env_cfg    ← env_cfgs.py:31
                      └─ make_velocity_env_cfg() ← velocity_env_cfg.py:36
```

**비교 baseline 정의**: `_go2_flat_pd_cfg` = `unitree_go2_flat_env_cfg` + `timestep=0.005 / decimation=4` 로 잡음. 이게 현재 `Unitree-Go2-Flat` task와 동치이며 "바닐라 DR + 5 ms PD" 상태.

**Gait05 의 직접 차이**: `_go2_flat_deploydr_cfg` 결과에 `cfg.rewards["foot_gait"].weight = 0.50` 한 줄만 추가. 따라서 아래 분석은 사실상 **DeployDR-v0 변경분 + Gait05 한 줄**.

---

## 1. 커리큘럼 학습 (Curriculum)

baseline `make_velocity_env_cfg` 의 `curriculum` (velocity_env_cfg.py:373-388):

| term | 기능 | 근거 |
|---|---|---|
| `terrain_levels` | 거리 기반 terrain difficulty up/down | velocity_env_cfg.py:374-377, mdp/curriculums.py:30 |
| `command_vel` | step 기반 lin_vel range 점진 확장 (0→5000×24 step에서 ±0.5/±0.5 → ±1.0/±1.0 등) | velocity_env_cfg.py:378-387, curriculums.py:67 |

**Flat env에서 적용된 제거** (env_cfgs.py:419): `unitree_go2_flat_env_cfg`가 `terrain_levels`를 pop → flat에서는 **`command_vel` 만** 살아있음.

### DeployDR-Gait05-v0 추가/변경

| term | 변경 | 근거 |
|---|---|---|
| `deploy_dr` | **신규 추가** — `src_mdp.DeployDRCurriculum` 단일 스칼라 level (0~1) 자동 조정 | `__init__.py:136-139` |
| `terrain_levels`, `command_vel` | 변경 없음 (flat에서는 `command_vel`만 남고 그대로) | — |

### `DeployDRCurriculum` 상세 (`mdp/curriculums.py:110-298`)

**제어 대상** (level이 곱해지는 것):
1. 모든 actor obs term의 noise 범위 → `noise_min/max × max_obs_noise_scale × level` (curriculums.py:199-204)
2. `push_robot` event의 `velocity_range` (x, y) → `±(max_push × level)` (curriculums.py:205-207)
3. `DelayedNoisyJointPositionAction.delay_steps` = `round(delay_max × level)` (curriculums.py:213-214)
4. `DelayedNoisyJointPositionAction.noise_std` = `noise_std_max × level` (curriculums.py:215)

**기본 파라미터** (`params={}` 로 등록되므로 `__init__` 디폴트 그대로, curriculums.py:128-149):

| 파라미터 | 값 |
|---|---|
| `ema_alpha` | 0.03 |
| `level_init` | **0.1** (학습 시작 시 DR 거의 꺼진 상태) |
| `level_min / level_max` | 0.0 / 1.0 |
| `up_step / down_step` | +0.01 / −0.03 (비대칭) |
| `up_count / down_count` | 4 / 2 reset 연속 충족 시 변동 |
| `cooldown` | 변경 후 5 reset 동결 |
| `timeout_up` | 0.80 (level-up 조건: timeout_ema ≥ 0.80) |
| `tracking_up` | 0.75 (level-up: tracking_ema ≥ 0.75) |
| `fall_up` | 0.15 (level-up: fall_ema ≤ 0.15) |
| `fall_down` | 0.25 (level-down: fall_ema ≥ 0.25) |
| `max_obs_noise_scale` | 1.0 |
| `max_push` | 0.5 m/s |
| `tracking_terms` | `("track_linear_velocity", "track_angular_velocity")` |
| `push_term` | `"push_robot"` |

**진행 방식** (curriculums.py:231-298):
- 매 reset 시 호출 → tracking ema / timeout ema / fall ema 갱신
- `up_ok = (timeout_ema≥0.80 AND tracking_ema≥0.75 AND fall_ema≤0.15)` 가 4회 연속 → `level += 0.01`
- `down_ok = (fall_ema≥0.25)` 가 2회 연속 → `level -= 0.03`
- 변경 시 5 reset 쿨다운

**critic 노출**: `env._deploy_dr_level`, `env._deploy_delay_steps`, `env._deploy_action_noise_std` 에 기록 (curriculums.py:219-221) → §4 critic obs가 읽음.

---

## 2. Domain Randomization (DR)

### baseline `make_velocity_env_cfg.events` 6종 (velocity_env_cfg.py:187-256)

| term 이름 | 대상 | 모드 | 범위/분포 | 커리큘럼 연동 | 근거 |
|---|---|---|---|---|---|
| `reset_base` | base pose & velocity | reset | pose: x,y∈±0.5, z=0, yaw∈±π / velocity: `{}` (없음) | ✗ | velocity_env_cfg.py:188-200 |
| `reset_robot_joints` | 모든 joint pos/vel | reset | pos offset (0,0), vel offset (0,0) — **외란 0** | ✗ | velocity_env_cfg.py:201-209 |
| `push_robot` | base velocity (6축 push) | interval 1~3 s | x,y∈±0.5, z∈±0.4, roll/pitch∈±0.52, yaw∈±0.78 | ✗ (baseline) | velocity_env_cfg.py:210-224 |
| `foot_friction` | 4 발 geom friction (shared) | startup | abs U(0.3, 1.2) | ✗ | velocity_env_cfg.py:225-234 |
| `encoder_bias` | 모든 joint 인코더 bias | startup | U(−0.015, 0.015) rad | ✗ | velocity_env_cfg.py:235-242 |
| `base_com` | base_link CoM offset | startup | x,y∈±0.025, z∈±0.03 m | ✗ | velocity_env_cfg.py:243-255 |

추가로 `unitree_go2_rough_env_cfg(play=True)` 시 `randomize_terrain` event 추가하지만 flat에서는 무관 (env_cfgs.py:132-136).

### DeployDR-Gait05-v0 의 DR diff (`__init__.py:62-114`)

#### 제거된 term

| term | 처리 | 근거 |
|---|---|---|
| `encoder_bias` | `pop` 으로 제거 | __init__.py:67-78 |
| `base_com` | `pop` 으로 제거 | __init__.py:67-78 |
| `randomize_base_mass`, `randomize_link_mass`, `randomize_actuator_gains`, `randomize_motor_strength`, `randomize_V_bus`, `external_force_torque`, `joint_pos_bias` | `pop(..., None)` — baseline에 없는 항목이라 no-op | __init__.py:67-78 |

→ baseline 기준 **실제 제거된 건 `encoder_bias`, `base_com` 2개**.

#### 변경/오버라이드된 term

| term | baseline | DeployDR-Gait05-v0 | 모드 | 근거 |
|---|---|---|---|---|
| `foot_friction.ranges` | abs U(0.3, 1.2) | **abs U(0.3, 1.25)** | startup | __init__.py:93 |
| `reset_base.pose_range` | x,y∈±0.5, yaw∈±π | **x,y,z,yaw 모두 0 고정** | reset | __init__.py:97-100 |
| `reset_base.velocity_range` | `{}` (없음) | **6축 U(−0.5, 0.5)** (x,y,z,roll,pitch,yaw) | reset | __init__.py:101-104 |
| `reset_robot_joints.func` | `mdp.reset_joints_by_offset` (덧셈) | **`src_mdp.reset_joints_by_scale`** (곱셈) | reset | __init__.py:108-114, events.py:162-207 |
| `reset_robot_joints.position_range` | (0, 0) offset | **(0.5, 1.5) scale** ← `default × U(0.5,1.5)` + soft limit clamp | reset | events.py:182-190 |
| `reset_robot_joints.velocity_range` | (0, 0) | (0, 0) (변경 안 함) | reset | __init__.py:111 |
| `push_robot.func` | `mdp.push_by_setting_velocity` | **`src_mdp.push_by_setting_velocity_logged`** ((x,y) push 기록 → critic 노출) | interval | __init__.py:85, events.py:210-230 |
| `push_robot.interval_range_s` | (1.0, 3.0) | **(5.0, 5.0)** 고정 | — | __init__.py:86 |
| `push_robot.velocity_range` | 6축 | **x,y만 ±0.5** (z / 각속도 제거) ★ curriculum이 `±(0.5×level)` 로 매번 덮어씀 | — | __init__.py:87 + curriculums.py:205-207 |

### 신규 추가된 DR

| term | 대상 | 모드 | 범위 | 커리큘럼 연동 | 근거 |
|---|---|---|---|---|---|
| `deploy_dr` (curriculum) | obs noise scale + push range + action delay/noise | reset 호출 | level∈[0,1] 자동 | **이 자체가 커리큘럼** | __init__.py:136, curriculums.py:110 |
| Action delay (ring buffer) | 정책 raw action | 매 step | delay_steps∈{0,1} (level≥0.5에서 1) | ✓ curriculum이 set | __init__.py:175, actions.py:33 |
| Action gaussian noise | delay 적용 후 raw action | 매 step | std = `0.1 × level` | ✓ curriculum이 set | actions.py:66 |

### 요청한 항목 점검

| 항목 | 결과 |
|---|---|
| 몸체 질량 | **DR 없음** (baseline에도 없고 DeployDR에도 추가 안 함) |
| 몸체 속도 | reset 시 6축 ±0.5 + push로 (x,y) `±0.5×level` |
| 관절 각도 | reset 시 `default × U(0.5,1.5)` (scale, clamp). 학습 중 외란 없음 |
| 발 높이 | **DR 없음** (foot_clearance reward에서 target 0.10 m 페널티만 있음) |
| 정지/운동 마찰계수 | **분리 DR 없음 (확정)**. MuJoCo는 단일 Coulomb μ 모델이라 static/kinetic 분리 자체를 지원 안 함. foot geom의 sliding(axis 0)만 `U(0.3, 1.25)` shared, torsional/rolling은 nominal 고정. 자세한 근거는 마지막 섹션 |
| Push | 5 s 고정 interval, x/y `±0.5×level` |
| CoM offset | **제거됨** |
| PD gain | **DR 없음** (baseline에도 없음. `randomize_actuator_gains`는 pop 대상이었지만 baseline에 미존재) |
| Actuator delay | action delay 1-step max, level로 on/off |

---

## 3. Actor 네트워크

### Actor obs (정책 입력) — `make_velocity_env_cfg.actor_terms` (velocity_env_cfg.py:58-91)

| term | dim 예상 | scale | noise | flat에서 제거? | 근거 |
|---|---|---|---|---|---|
| `base_ang_vel` | 3 | 1.0 | U(−0.2, 0.2) | — | velocity_env_cfg.py:59-63 |
| `projected_gravity` | 3 | 1.0 | U(−0.05, 0.05) | — | :64-67 |
| `command` (twist) | **3** | 1.0 | — | — | :68-71 |
| `phase` | 2 | 1.0 | — | — | :72-75 |
| `joint_pos` | 12 | 1.0 | U(−0.01, 0.01) | — | :76-79 |
| `joint_vel` | 12 | 1.0 | U(−1.5, 1.5) | — | :80-83 |
| `actions` (last_action) | 12 | 1.0 | — | — | :84 |
| `height_scan` | — | 1/5 | U(−0.1, 0.1) | **제거** (env_cfgs.py:415) | :85-90 |

총 actor dim (flat에서) = 3+3+3+2+12+12+12 = **47**. (`__init__.py:149`도 47D라고 언급.)

**주의**: `command` obs는 `UniformVelocityCommand.command` property가 반환하는 `vel_command_b` 즉 `(lin_vel_x, lin_vel_y, ang_vel_z)` **3D**. `Ranges.heading=(−π, π)`는 내부 `heading_target` 샘플링 범위일 뿐 obs에 직접 들어가지 않음 (heading 정보는 `_update_command`가 `ang_vel_z` 자리에 `heading_error × stiffness`로 환산해 넣음). 근거: `mjlab/envs/mdp/observations.py:91-94`, `mdp/velocity_command.py:54-56`.

`ObservationGroupCfg(actor, enable_corruption=True, history_length=1, concatenate_terms=True)` (velocity_env_cfg.py:124-128).

### DeployDR-Gait05-v0 변경

| 항목 | baseline | DeployDR-Gait05-v0 | 근거 |
|---|---|---|---|
| 추가/제거 | — | **변경 없음 (47D 유지)** | __init__.py:149 주석 |
| obs noise scale | 위 표 그대로 | curriculum이 매 변경 시 `(n_min, n_max) × max_obs_noise_scale × level` 로 **전부 재생성** (level=0이면 noise=None) | curriculums.py:194-204 |

### Action (정책 출력)

| 항목 | baseline | DeployDR-Gait05-v0 | 근거 |
|---|---|---|---|
| 클래스 | `mjlab.envs.mdp.actions.JointPositionActionCfg` | **`src_mdp.DelayedNoisyJointPositionActionCfg`** (sub-class) | __init__.py:175-182, actions.py:77-86 |
| `scale` | 0.25 (velocity_env_cfg.py:156) | 0.25 (그대로 전달) | __init__.py:178 |
| `actuator_names` | `(".*",)` | `(".*",)` | — |
| `use_default_offset` | True | True | — |
| `delay_max_steps` | — | **1** (참조 레포 스펙) | __init__.py:180 |
| `noise_std_max` | — | **0.1** (raw action 공간 std) | __init__.py:181 |
| 적용 순서 | scale + offset 만 | **ring-buffer delay → gaussian noise → scale+offset** | actions.py:57-69 |
| level=0 시 동작 | — | delay=0 / noise=0 → 기존 `JointPositionAction` 과 수치적 동일 | actions.py:42-48 + curriculums.py:211-217 |

---

## 4. Critic 네트워크

`unitree_go2_ppo_runner_cfg`는 actor / critic 모두 `hidden_dims=(512, 256, 128), activation=elu, obs_normalization=True` 의 별개 MLP. `enable_corruption=False` 인 별도 critic obs group을 받는 **asymmetric actor-critic** (rl_cfg.py:13-49, velocity_env_cfg.py:130-135).

### baseline critic obs (velocity_env_cfg.py:93-121)

`critic_terms = {**actor_terms, ...}` 로 actor 의 8개 + 5개 추가:

| 추가 term | dim 예상 | noise | 근거 |
|---|---|---|---|
| `base_lin_vel` | 3 | U(−0.5, 0.5) | velocity_env_cfg.py:95-99 |
| `height_scan` (no-noise 버전) | — | 없음 (flat에서 어차피 제거) | :100-104 |
| `foot_height` | 4 (site_z) | — | :105-108 |
| `foot_air_time` | 4 | — | :109-112 |
| `foot_contact` | 4 | — | :113-116 |
| `foot_contact_forces` | 12 (4×3, sign·log1p) | — | :117-120 |

Flat에서 height_scan 제거 → **critic dim = 47 + 3 + 4 + 4 + 4 + 12 = 74** (확정).

각 항목의 shape는 ContactSensor 코드(`mjlab/sensor/contact_sensor.py`)에서 확인:
- `found`: `[B, N·num_slots]` = (B, 4) (line 70 docstring)
- `force`: `[B, N, 3]` = (B, 4, 3) (line 145) → `flatten(start_dim=1)` → 12
- `current_air_time`: `(n_envs, n_primary)` = (B, 4) (line 246)
- `foot_height`: site_pos_w[:, site_ids, 2] → (B, 4) (site_names 4개)
- `base_lin_vel`: builtin_sensor → (B, 3)

### DeployDR-Gait05-v0 critic 변경 (`__init__.py:152-189`)

| 항목 | baseline | DeployDR-Gait05-v0 | 근거 |
|---|---|---|---|
| `base_lin_vel.scale` | 1.0 (기본) | **2.0** | __init__.py:154-155 |
| `foot_friction_coeff` | 없음 | **신규 추가** — 발 마찰 axis 0 평균, [B,1]. DR로 흔들린 값을 critic이 알도록 | __init__.py:156-162, observations.py:58-68 |
| `deploy_curriculum_level` | 없음 | **신규 추가** — `env._deploy_dr_level`, [B,1] | __init__.py:163-165, observations.py:71-77 |
| `push_history_xy` | 없음 | **신규 추가** — `env._last_push_xy`, [B,2] | __init__.py:166-168, observations.py:80-88 |
| `deploy_delay_steps` | 없음 | **신규 추가** — 현재 action delay step, [B,1] | __init__.py:184-186, observations.py:91-94 |
| `deploy_action_noise_std` | 없음 | **신규 추가** — 현재 action noise std, [B,1] | __init__.py:187-189, observations.py:97-100 |

신규 5개 합산 = 1 + 1 + 2 + 1 + 1 = **+6 dim** → **DeployDR-Gait05-v0 critic dim = 74 + 6 = 80 (확정)**.

**주석 "78D → 80D"의 출처** (`__init__.py:183`): baseline(74D)이 아니라 **commit 중간 단계 기준**.
- baseline (flat) = 74D
- Phase 5 commit `ad1fe10` 후 (friction 1 + level 1 + push 2 추가) = **78D** ← 주석의 "78D"
- 작업 3 commit `366e17f` 후 (delay 1 + noise 1 추가) = **80D** ← 주석의 "80D"

이 5개 추가 obs들은 모두 **DR로 흔들린 물리량 또는 DR 상태를 privileged로 critic에만 제공**하는 형태 — typical asymmetric actor-critic / sim-to-real 패턴.

### Critic output

`RslRlModelCfg(distribution_cfg 없음)` → value head (스칼라). PPO 표준. (rl_cfg.py:26-30)

---

## 5. Reward

### baseline reward (velocity_env_cfg.py:262-355)

| term | weight | 함수 | 비고 |
|---|---|---|---|
| `track_linear_velocity` | **+1.0** | `mdp.track_linear_velocity` (std=√0.25) | :263-267 |
| `track_angular_velocity` | **+1.0** | `mdp.track_angular_velocity` (std=√0.5) | :268-272 |
| `body_orientation_l2` | **−1.0** | `mdp.body_orientation_l2` | :273-277 |
| `pose` (variable_posture) | **+1.0** | `mdp.variable_posture` (standing/walking/running std 분리, walk_thr 0.1, run_thr 1.5) | :278-290, env_cfgs.py:97-111 |
| `body_ang_vel` | **−0.05** | `mdp.body_angular_velocity_penalty` | :291-295 |
| `angular_momentum` | **−0.025** | `mdp.angular_momentum_penalty` | :296-300 |
| `is_terminated` | **−200.0** | `mdp.is_terminated` | :301 |
| `joint_acc_l2` | **−2.5e-7** | `mdp.joint_acc_l2` | :302 |
| `joint_pos_limits` | **−10.0** | `mdp.joint_pos_limits` | :303 |
| `action_rate_l2` | **−0.05** | `mdp.action_rate_l2` | :304 |
| `foot_gait` | **+0.5** | `mdp.feet_gait` (period 0.6, offset [0, 0.5, 0.5, 0] — 대각 trot, threshold 0.56, cmd_thr 0.1) | :305-316, env_cfgs.py:113 |
| `foot_clearance` | **−1.0** | `mdp.feet_clearance` (target 0.10 m) | :317-326 |
| `foot_slip` | **−0.25** | `mdp.feet_slip` | :327-336 |
| `soft_landing` | **−1e-3** | `mdp.soft_landing` | :337-345 |
| `stand_still` | **−1.0** | `mdp.stand_still` (cmd<0.1 일 때 default pos 유지) | :346-354 |

### DeployDR-Gait05-v0 변경

| term | baseline weight | DeployDR-Gait05-v0 weight | 변경처 |
|---|---|---|---|
| `foot_gait` | +0.5 | **+0.50** (Gait05 는 0.50, DeployDR-v0 는 0.10) | DeployDR-v0: `__init__.py:146` (0.10), Gait05: `__init__.py:207` (0.50) |
| 그 외 14개 reward | 동일 | **동일** (추가/삭제/weight 변경 없음) | — |

**관찰**: DeployDR-v0는 baseline `foot_gait=0.5` 를 의도적으로 0.10 으로 낮춤(__init__.py:141-146 주석에 "tracking reward 회복"). Gait05 는 다시 0.50으로 복원 — 즉 **Gait05 의 reward는 baseline과 완전 동일**. baseline → DeployDR-Gait05-v0 의 reward diff는 **없음** (foot_gait도 0.5 → 0.5 net).

→ **sim-to-real 정규화/페널티 (action_rate, joint_acc, joint_pos_limits, foot_slip, soft_landing) 변경 없음.** reward 측면에서 Gait05는 baseline 그대로이고, DR/curriculum/critic/action 쪽으로 sim-to-real 적응을 몰아줬다는 게 코드의 명백한 의도.

---

## 6. 기타 변경점

### Commands (velocity_env_cfg.py:165-181 vs __init__.py:119-120)

| 항목 | baseline | DeployDR-Gait05-v0 |
|---|---|---|
| `resampling_time_range` | (3.0, 8.0) | 동일 |
| `rel_standing_envs` | 0.05 | 동일 |
| `rel_heading_envs` | 0.25 | **1.0** (모든 비정지 env가 heading lock) ← __init__.py:120 |
| `heading_command` | True | True |
| `heading_control_stiffness` | 0.5 | 동일 |
| `ranges.lin_vel_x` | (−1.0, 2.0) | 동일 (단 `command_vel` curriculum이 step 0 ~ 5000×24까지 점진 확장) |
| `ranges.lin_vel_y` | (−1.0, 1.0) | 동일 |
| `ranges.ang_vel_z` | (−1.0, 1.0) | 동일 |
| `ranges.heading` | (−π, π) | 동일 |

### Terminations

| term | baseline | DeployDR-Gait05-v0 |
|---|---|---|
| `time_out` | episode_length 기반 | 동일 (velocity_env_cfg.py:362) |
| `fell_over` | `mdp.bad_orientation`, limit=70° (방향 무관 단일) | **`src_mdp.bad_orientation_roll_pitch`, limit_roll=0.8 rad, limit_pitch=1.0 rad** ← __init__.py:128-131, terminations.py:31-51 |
| `illegal_contact` | force_threshold=10.0 N (env_cfgs.py:119-122, rough에서 추가) | **force_threshold=1.0 N** ← __init__.py:124-125 |

### Sim 파라미터

| 항목 | baseline (`make_velocity_env_cfg`) | `unitree_go2_rough` / `_flat` | `_go2_flat_pd_cfg` (= Flat task) | DeployDR-Gait05-v0 |
|---|---|---|---|---|
| `sim.mujoco.timestep` | 0.005/45 ≈ 0.111 ms | 동일 상속 | **0.005 (5 ms)** | 동일 상속 |
| `decimation` | 4×45=180 | 동일 상속 | **4** | 동일 상속 |
| policy dt | 20 ms (180 × 0.111 ms = 20 ms) | — | 20 ms (4 × 5 ms) | 20 ms |
| `episode_length_s` | 20.0 | 동일 | 동일 | 동일 (play 모드 시 1e9) |
| `ccd_iterations` | 미설정 | rough: 500 / flat: 50 | 동일 | 동일 |
| `njmax / nconmax` | 1500 / 35 | flat: njmax=300, nconmax=None, contact_sensor_maxmatch=64 | 동일 | 동일 |
| Scene terrain | `generator` (ROUGH) | flat은 `"plane"` (env_cfgs.py:408) | 동일 | 동일 |
| 발 geom | — | `{FR,FL,RR,RL}_foot_collision` (env_cfgs.py:50) | 동일 | 동일 |

### 그 외 baseline 대비 눈에 띄는 추가

- `env._last_push_xy`, `env._deploy_dr_level`, `env._deploy_delay_steps`, `env._deploy_action_noise_std` 어트리뷰트가 curriculum/event 함수에서 매 step·reset 시 셋팅됨 (curriculums.py:219-221, events.py:228-230). actor 47D를 유지하면서 critic만 이 4개 어트리뷰트를 obs로 읽음.
- `unitree_go2_ppo_runner_cfg` 자체는 baseline과 동일 — 즉 **PPO 하이퍼파라미터는 전혀 변경 없음** (gamma 0.99, lam 0.95, lr 1e-3 adaptive, KL 0.01, max_iter 10001, num_steps_per_env 24).
- play 모드: rough cfg에서 `push_robot` pop + actor `enable_corruption=False` + `randomize_terrain` 추가 + curriculum clear (env_cfgs.py:125-143). DeployDR-Gait05-v0도 이 play 동작 그대로 상속.

---

## baseline 대비 핵심 변경점 요약

1. **단일 스칼라 DR curriculum** (`DeployDRCurriculum`, level∈[0,1])이 obs noise/push velocity/action delay/action noise 4채널을 동시 스케일링 — reward EMA 기반 자동 up/down.
2. **Action을 `DelayedNoisyJointPositionAction`으로 교체**: 1-step ring buffer delay + raw 공간 gaussian noise (std max 0.1). level=0이면 기존 PD action과 수치적 동일.
3. **Reset 동작이 deploy 스타일로 전환**: base pose 고정·6축 ±0.5 초기 속도 외란, joint pos를 default×U(0.5,1.5) **곱셈** scale (mjlab 기본은 offset만 지원이라 `reset_joints_by_scale` 신설).
4. **Push** : interval 1~3 s → 5 s 고정, 6축 → x,y만, 크기는 curriculum이 `±(0.5×level)`로 실시간 조절. push 외란은 critic이 보도록 `_last_push_xy`에 기록.
5. **baseline DR 2종 제거**: `encoder_bias`, `base_com`. foot_friction 폭만 (0.3,1.2)→(0.3,1.25)로 미세 확장.
6. **Termination 강화**: `illegal_contact` 10 N→**1 N** (다리/몸통이 살짝만 닿아도 종료), `fell_over`을 roll/pitch 분리(0.8/1.0 rad)로 교체.
7. **Heading lock 100%**: `rel_heading_envs` 0.25→1.0 — 모든 비정지 env가 yaw 목표 추종.
8. **Asymmetric critic privileged obs +5종**: `foot_friction_coeff`, `deploy_curriculum_level`, `push_history_xy`, `deploy_delay_steps`, `deploy_action_noise_std`. `base_lin_vel`은 critic에서 ×2 스케일.
9. **Reward는 baseline 그대로 (Gait05만)**: DeployDR-v0가 `foot_gait` 0.5→0.10 으로 낮춘 것을 Gait05가 0.50으로 복원해 net 0 변경. 즉 Gait05는 sim2real 적응을 reward가 아니라 DR/curriculum/critic/action 쪽으로 몰아준다.
10. **변경 없는 것**: PPO 하이퍼파라미터, actor 네트워크(47D), sim dt/decimation (5 ms/4), scene/terrain, 몸체 질량·CoM·PD gain·외력 등 **물리 파라미터 DR은 모두 없음** — 마찰 외 sim 물리는 nominal 고정. 사용자가 요청한 "몸체 질량 DR / PD gain DR" 등은 의도적으로 **부재**.

---

## 마찰계수 처리 (2026-05-27 모델 XML + DR 함수 + MuJoCo 모델 모두 검증)

- **MuJoCo 모델 차원에서 정지/운동 마찰계수 분리 없음**: MuJoCo는 표준 Coulomb friction (단일 μ) — `|F_tangent| ≤ μ·|F_normal|` 안이면 stick, 밖이면 slip. static μ_s / kinetic μ_k 분리 미지원. 모델 XML의 `friction` 속성도 단일 값. `elliptic` cone (`scene_go2.xml:4`)은 anisotropic friction 모양 변경이지 static/kinetic 분리와 무관.
- **Go2 friction XML 정의** (`scene_go2.xml`):
  | 영역 | friction (sliding/torsional/rolling) | condim | 활성 축 |
  |---|---|---|---|
  | 일반 body geom (`go2` default) | `0.4` (sliding만) | 1 | sliding만 |
  | 발 geom (`foot` default) | `0.4 0.02 0.01` | 6 | sliding + torsional + rolling 모두 |
- **`foot_friction` DR이 randomize 하는 축**: mjlab `dr.geom_friction` (`mjlab/envs/mdp/dr/geom.py:116-146`)이 `default_axes=[0]`. 호출처(`__init__.py:93`)가 `axes` 인자를 안 주므로 **axis 0 (sliding) 만** `U(0.3, 1.25)`로 흔들림. **torsional(0.02), rolling(0.01)은 nominal 고정**.
- **종합**: sliding friction만 4 발 geom 공유 random sample (`shared_random=True`). 정지/운동 분리 DR은 모델/함수/호출처 어디에도 없음.

## 확정된 차원 (이전 "불확실"에서 승격, 2026-05-27 검증)

| 그룹 | dim | 근거 |
|---|---|---|
| Actor (flat) | **47** | 3+3+3+2+12+12+12. `command` obs는 3D (lin_vel_xy + ang_vel_z) |
| Critic baseline (flat) | **74** | actor 47 + base_lin_vel 3 + foot_height 4 + foot_air_time 4 + foot_contact 4 + foot_contact_forces 12. ContactSensor docstring (contact_sensor.py:70, 145, 246)에서 shape 확정 |
| Critic DeployDR-Gait05-v0 (flat) | **80** | 74 + (foot_friction_coeff 1 + deploy_curriculum_level 1 + push_history_xy 2 + deploy_delay_steps 1 + deploy_action_noise_std 1) |

주석 "78D → 80D"(`__init__.py:183`)는 baseline 대비가 아니라 Phase 5 commit `ad1fe10` 시점(74→78)과 작업 3 commit `366e17f` 시점(78→80) 사이의 incremental dim 변화.
