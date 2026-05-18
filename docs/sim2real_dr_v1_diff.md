# sim2real-DR-v1 vs main 비교

브랜치 `experiment/sim2real-dr-v1` 가 `main` 대비 변경한 항목과 그대로 둔 항목을 정리합니다.
대상 태스크는 **Method A 전기모터** 계열 (`Unitree-Go2-Flat-MethodA-Electric` /
`Unitree-Go2-Flat-MethodA-Electric-Sim2Real`) 입니다. Coupled / Method B /
A+ tloop config 의 모터 파라미터는 변경되지 않았습니다.

태스크 의미 (rename 이후):
- `Unitree-Go2-Flat-MethodA-Electric` — 실측 모터 파라미터 + obs delay/history=5,
  baseline DR 만 (foot_friction 0.3..1.2, encoder_bias, base_com, push_robot).
- `Unitree-Go2-Flat-MethodA-Electric-Sim2Real` — 위 + 7 개 추가 DR + 확장된
  foot_friction (0.2, 1.5).
- 2-phase curriculum: 전자로 ~2000 iter (Phase 1) → 후자로 ~2000 iter (Phase 2 resume).

관련 커밋:
- `d59589e` — sim2real DR v1: obs delay+history, 7 DR terms, real motor params
- `7c0986c` — add MethodA-Electric-Phase1 task (no DR events) for 2-phase curriculum
- (이번) — rename: Phase1 → base `Electric`, base → `Electric-Sim2Real`

---

## 1. 변경된 전기모터 파라미터 (Method A 전용)

`_MA_MOTOR` 딕셔너리 한 곳에서 hip / thigh / calf 세 액추에이터에 동시에 적용됩니다.

**파일**: `src/assets/robots/unitree_go2/go2_constants.py:186-189`

| 파라미터 | main (`151bbfa`) | sim2real-DR-v1 | 비고 |
|---|---|---|---|
| `Kt` (토크 상수) | `0.128` N·m/A | **`0.26`** N·m/A | 실제 Go2 모터 데이터시트값 |
| `Ke` (역기전력 상수) | `0.128` V·s/rad | **`0.26`** V·s/rad | 물리적으로 Kt = Ke |
| `R` (권선 저항) | `0.3` Ω | **`0.66`** Ω | |
| `L` (권선 인덕턴스) | `1e-4` H = 100 μs | **`83e-6`** H = 83 μs | τ_e = L/R 일정한 수준 유지 |
| `gear_ratio` | `6.33` | `6.33` | 변경 없음 |
| `V_bus` | `cfg default (=24.0 V)` | **`30.8` V (명시 지정)** | DR 의 nominal 값으로 사용 |

> Coupled / Method B / A+ tloop 에 사용되는 `GO2_COUPLED_ELECTRIC_*`,
> `_MB_MOTOR`, `_APLUS_TLOOP_MOTOR` 의 Kt/Ke/R/L 은 모두 `0.128 / 0.128 / 0.3 / 1e-4`
> 로 main 과 동일합니다 (`go2_constants.py:155-180, 217-247, 256-282`).

### 액추에이터 코드 변경 (per-env V_bus 지원)

**파일**: `src/assets/robots/unitree_go2/mj_native_electric_actuator.py`

- `__init__` 에 `self._V_bus: torch.Tensor | None = None` 추가 (line 262-264)
- `init_state(num_envs)` 에서 `(num_envs, 1)` 텐서로 채워 둠 (line 424-427)
- `compute()` 의 버스 전압 포화 로직이 스칼라 `cfg.V_bus` 대신 per-env 텐서
  `self._V_bus` 를 사용하도록 변경 (line 555-557)

→ DR 이벤트 (`randomize_V_bus`) 가 이 텐서를 직접 덮어써서 환경별로 V_bus 를
  랜덤화합니다.

---

## 2. 추가된 Domain Randomization (DR)

전부 `Unitree-Go2-Flat-MethodA-Electric-Sim2Real` 태스크에만 적용됩니다.
Base `Unitree-Go2-Flat-MethodA-Electric` 에는 의도적으로 빠져 있어 2-phase
curriculum (base → Sim2Real resume) 구성이 가능합니다.

### 2-1. Observation 측 변경

**파일**: `src/tasks/velocity/config/go2/env_cfgs.py:200-204`
mjlab 측 정의: `mjlab/managers/observation_manager.py:25-65, 91-98`

| 항목 | main | sim2real-DR-v1 |
|---|---|---|
| actor group `history_length` | `0` (= 단일 frame) | **`5`** |
| `base_ang_vel` / `projected_gravity` / `joint_pos` / `joint_vel` 의 `delay_min_lag` | `0` | `0` |
| 같은 항목들의 `delay_max_lag` | `0` | **`4`** |
| 위 4개 외 (`command`, `phase`, `actions`, `height_scan`) | delay 없음 | delay 없음 |

#### (a) `history_length` = 1 → 5 의 의미

- mjlab 의 obs term 은 default `history_length=0` (= 단일 step). group 레벨에서
  `actor_group.history_length = 5` 로 덮어쓰면 그룹 내 **모든 텀**이 직전 5 개
  policy step 의 값을 한 obs vector 에 누적해서 정책에 전달합니다.
- group 의 `flatten_history_dim=True` (default) + `concatenate_terms=True`
  조합이라 obs 는 *term-major* 로 flatten 됩니다:
  `[base_ang_vel_{t-4..t}, projected_gravity_{t-4..t}, command, phase, joint_pos_{t-4..t}, ...]`.
  즉 history 가 적용되는 텀의 차원만 ×5 가 되고 `command` / `phase` 처럼
  per-step 신호 차원은 그대로입니다.
- (`mjlab/managers/observation_manager.py:163-170, 471-478`)

**왜 5 인가 — 두 가지 motivation 이 동시에 작동:**

1. **Observation delay 보상**: `delay_max_lag=4` 와 짝을 이룹니다. 정책이
   받은 sample 이 0–4 step 만큼 늦었을 수 있으므로, 윈도우에 최근 5 frame 이
   다 들어 있으면 "내가 본 신호가 몇 step 전 것인지" 를 시계열 패턴에서
   *implicit* 하게 추론할 수 있습니다 (Markov 가정 회복).
2. **속도/가속도 신호의 일관성 추정**: `base_ang_vel`, `joint_vel` 는 차분/필터
   noise 와 latency 모두에 취약합니다. 5 frame 윈도우면 정책이 단일 noisy
   sample 대신 짧은 추세를 보고 트랙 명령에 반응할 수 있습니다.

5 라는 숫자 자체는 obs delay 윈도우(`max_lag + 1 = 5`)와 매칭되도록 정한
값입니다. 더 큰 history 도 가능하지만 actor input dim 이 비례해 커지고
exploration 측면에서 노이즈가 늘어 실익이 줄어듭니다.

#### (b) `delay_min_lag=0`, `delay_max_lag=4` 의 의미

- 매 step 마다 env 별로 lag ∈ {0, 1, 2, 3, 4} 를 uniform sample → 그만큼 과거의
  obs 를 정책에 노출.
- policy dt = 20 ms (decimation 200 × physics dt 0.1 ms) 이므로
  **0–80 ms uniform random latency** 에 해당.
  실제 Go2 의 unitree_sdk2 + DDS 통신 + IMU/encoder 처리 지연(수 ms ~ 수십 ms)
  영역을 커버합니다.
- mjlab default 는 `delay_per_env=True` 이므로 env 마다 독립적으로 lag 가 샘플됩니다 (`observation_manager.py:42-44`). 기본 `delay_hold_prob=0.0`,
  `delay_update_period=0` 라 매 step 매 env 마다 새로 샘플. 시간상관(burst)
  latency 가 필요해지면 이 두 값을 키워서 모델링 확장 가능.

#### (c) delay 가 *왜 4 텀에만* 걸렸나

지연된 4 텀 (`base_ang_vel`, `projected_gravity`, `joint_pos`, `joint_vel`)
은 모두 **로봇이 외부로부터 측정해 들여오는 센서 신호** (IMU gyro, IMU 중력
projection, encoder 위치/속도) 입니다. 반면 delay 가 없는 4 텀은 정책 내부
신호이거나 주기 신호라 latency 가 정의되지 않습니다:

| term | 종류 | delay 적용 여부 |
|---|---|---|
| `base_ang_vel`      | IMU gyro       | O |
| `projected_gravity` | IMU 중력       | O |
| `joint_pos`         | joint encoder  | O |
| `joint_vel`         | encoder 차분   | O |
| `command`           | RL 입력 명령    | X (외부 명령은 즉시 반영) |
| `phase`             | gait 위상      | X (정책 내부 시계) |
| `actions`           | last action    | X (정책 자기 출력) |
| `height_scan`       | terrain raycast | X (실제 hw 에는 없는 oracle 정보) |

### 2-2. Foot friction 범위 확장

**파일**: `src/tasks/velocity/config/go2/env_cfgs.py:208`
(원본 정의: `src/tasks/velocity/velocity_env_cfg.py:225-234`)

| 항목 | main | sim2real-DR-v1 |
|---|---|---|
| `foot_friction.params.ranges` | `(0.3, 1.2)` | **`(0.2, 1.5)`** |

### 2-3. 새로 추가된 7 개 DR 이벤트 텀

**파일**: `src/tasks/velocity/config/go2/env_cfgs.py:210-313`
사용자 정의 함수 정의 위치: `src/tasks/velocity/mdp/events.py`
mjlab 측 정의: `mjlab/managers/event_manager.py:22 (EventMode)`,
`mjlab/managers/event_manager.py:100-140 (EventTermCfg)`

#### `mode` 필드 — 언제 이 텀이 발화하는가

`EventMode = Literal["startup", "reset", "interval", "step"]` 4 종이 있으며,
**이름이 곧 발화 시점** 입니다.

| mode | 발화 시점 | 한 번 샘플된 값의 수명 | 적합한 대상 |
|---|---|---|---|
| `startup`  | env build 시 단 한 번 | 시뮬레이션 종료까지 고정 | 로봇 고유 물성치 (mass, gain, motor strength, geom friction, mech zero offset) |
| `reset`    | 매 episode reset 마다 | 다음 reset 까지 (= 한 episode) | episode 단위로 변하는 상태 (initial pose, 배터리 전압, terrain 선택 등) |
| `interval` | `interval_range_s=(min,max)` 에서 uniform sample 한 시간 간격마다 | trigger 마다 새 값, 다음 trigger 까지 유지 | episode 도중 발생하는 외란 (push, 외력/토크 kick) |
| `step`     | 매 env step 마다 unconditional | 1 step | per-step state 관리 (예: `apply_body_impulse` 의 force lifetime) |

(`mjlab/managers/event_manager.py:106-126` docstring 인용)

부가 옵션 (모두 default 사용 중):

- `interval` 의 `is_global_time=False` (default) → env 마다 독립 timer.
  True 면 모든 env 가 동시에 trigger.
- `reset` 의 `min_step_count_between_reset=0` (default) → reset 마다 매번 발화.
  > 0 으로 두면 짧은 에피소드가 연속될 때 너무 자주 fire 하는 걸 막는 throttle.

#### 본 브랜치에서 mode 선택 근거

| 텀 | 선택된 mode | 왜 이 mode 인가 |
|---|---|---|
| `randomize_V_bus` | `reset` | 배터리 전압은 한 episode (수~수십 초) 동안은 거의 일정. episode 사이에 다시 sample 하면 "충전 상태가 다른 로봇" 분포를 학습 가능 |
| `randomize_actuator_gains` | `startup` | PD gain 은 컨트롤러 튜닝 차이라 한 로봇 안에서는 안 변함 |
| `randomize_motor_strength` | `startup` | Kt = Ke 는 자석/권선 특성이라 같은 로봇이면 epi 간 변동 없음. 또 dynprm 을 매번 덮어쓰는 비용을 회피 |
| `randomize_base_mass` / `randomize_link_mass` | `startup` | 로봇 mass property — 한 빌드 안에서 고정. mjwarp `expand_model_fields()` 가 startup 단계에서 per-world 메모리 할당 결정 |
| `joint_pos_bias` | `startup` | mechanical zero offset 은 조립/캘리브레이션 오차 → 로봇 고유. `_offset` 캡처가 `__init__` 직후 startup 시점에 일어나야 일관됨 |
| `external_force_torque` | `interval (8–12 s)` | episode 도중 갑자기 들어오는 외란을 학습. 기존 `push_robot` (1–3 s) 보다 훨씬 긴 timescale 의 disturbance 를 추가로 학습 |

#### 텀 카탈로그

| # | 이벤트 키 | mode | 함수 | 핵심 파라미터 |
|---|---|---|---|---|
| 3-1 | `randomize_V_bus` | `reset` | `src_mdp.randomize_V_bus` | `voltage_range=(28.0, 33.6)` V |
| 3-2 | `randomize_actuator_gains` | `startup` | `mjlab.envs.mdp.dr.pd_gains` | `kp_range=(0.8, 1.2)`, `kd_range=(0.8, 1.2)`, `distribution="log_uniform"`, `operation="scale"` |
| 3-3 | `randomize_motor_strength` | `startup` | `src_mdp.randomize_motor_strength` | `scale_range=(0.9, 1.1)` — Kt = Ke 동일 스케일로 `gainprm[0]`, `dynprm[1]`, `dynprm[3]` 동시 변경 (demag 검출 비활성 유지) |
| 3-4 | `randomize_base_mass` | `startup` | `dr.body_mass` | `body_names=("base_link",)`, `ranges=(-1.5, 3.0)` kg, `operation="add"` |
| 3-5 | `randomize_link_mass` | `startup` | `dr.body_mass` | `body_names=".*(hip\|thigh\|calf).*"`, `ranges=(0.9, 1.1)`, `operation="scale"` |
| 3-6 | `joint_pos_bias` | `startup` | `src_mdp.joint_pos_bias` | `bias_range=(-0.03, 0.03)` rad — `default_joint_pos` + JointPositionAction `_offset` 양쪽 동기 적용 |
| 3-7 | `external_force_torque` | `interval` | `envs_mdp.apply_external_force_torque` | `interval_range_s=(8.0, 12.0)`, `force_range=(-30, 30)` N, `torque_range=(-3, 3)` N·m, base_link 에 적용 |

### 2-4. Base 태스크와 Sim2Real 태스크 구조

**파일**: `src/tasks/velocity/config/go2/env_cfgs.py`
태스크 등록: `src/tasks/velocity/config/go2/__init__.py`

`unitree_go2_flat_methoda_electric_env_cfg` (base) 가 실측 모터 파라미터 +
obs delay/history 만 갖추고, `unitree_go2_flat_methoda_electric_sim2real_env_cfg`
가 base 위에 위 2-3 의 7 개 DR 텀 추가 + `foot_friction.ranges` 를
`(0.2, 1.5)` 로 확장합니다.

2-phase 커리큘럼:
1. `Unitree-Go2-Flat-MethodA-Electric` 로 ~2000 iter 학습 (base, baseline DR 만).
2. Phase 1 의 체크포인트를 `Unitree-Go2-Flat-MethodA-Electric-Sim2Real` 로 resume
   하여 ~2000 iter 추가 학습 (full sim2real DR).

---

## 3. 변경되지 않은 항목 (양 브랜치 동일)

### 3-1. PD gain (Kp, Kd) — nominal 값

**파일**: `src/assets/robots/unitree_go2/go2_constants.py`
(`GO2_METHODA_HIP/THIGH/CALF` line 220-227 및 다른 모든 액추에이터 cfg)

| 액추에이터 | `stiffness` (Kp) | `damping` (Kd) | `effort_limit` | `armature` |
|---|---|---|---|---|
| hip   | 20.0 | 1.0 | 23.5 | 0.01 |
| thigh | 20.0 | 1.0 | 23.5 | 0.01 |
| calf  | 40.0 | 2.0 | 45.0 | 0.02 |

> 단 sim2real 브랜치에서는 위 nominal Kp/Kd 가 학습 중 `randomize_actuator_gains`
> (2-3 #3-2) 에 의해 환경별 ±20 % 로 곱해집니다.

### 3-2. RL / PPO 하이퍼파라미터

**파일**: `src/tasks/velocity/config/go2/rl_cfg.py`

`unitree_go2_ppo_runner_cfg()` 가 그대로 재사용되며 `experiment_name` /
`run_name` 만 Method A / B 별로 바뀝니다. 핵심값:

| 항목 | 값 |
|---|---|
| actor / critic hidden dims | `(512, 256, 128)`, `elu`, obs normalize |
| init log_std | `1.0` (scalar) |
| `value_loss_coef` | 1.0 |
| `clip_param` | 0.2 |
| `entropy_coef` | 0.01 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `learning_rate` | 1e-3 (adaptive, `desired_kl=0.01`) |
| `gamma`, `lam` | 0.99, 0.95 |
| `max_grad_norm` | 1.0 |
| `num_steps_per_env` | 24 |
| `max_iterations` | 10001 |
| `save_interval` | 100 |

### 3-3. Reward 가중치

**파일**: `src/tasks/velocity/velocity_env_cfg.py:258-355`
(브랜치 간 diff 없음)

| reward term | weight |
|---|---|
| `track_linear_velocity` | +1.0 |
| `track_angular_velocity` | +1.0 |
| `body_orientation_l2` | -1.0 |
| `pose` | +1.0 |
| `body_ang_vel` | -0.05 |
| `angular_momentum` | -0.025 |
| `is_terminated` | -200.0 |
| `joint_acc_l2` | -2.5e-7 |
| `joint_pos_limits` | -10.0 |
| `action_rate_l2` | -0.05 |
| `foot_gait` | +0.5 |
| `foot_clearance` | -1.0 |
| `foot_slip` | -0.25 |
| `soft_landing` | -1e-3 |
| `stand_still` | -1.0 |
| heading control stiffness (cmd) | 0.5 |

### 3-4. 기존 DR / Event 텀 (base velocity cfg 에서 정의된 것)

**파일**: `src/tasks/velocity/velocity_env_cfg.py:187-256`

| 텀 | mode | 핵심 파라미터 (양 브랜치 동일) |
|---|---|---|
| `reset_base` | reset | `pose_range x/y=(-0.5, 0.5)`, `yaw=(-π, π)` |
| `reset_robot_joints` | reset | offset 0, vel 0 |
| `push_robot` | interval (1–3 s) | lin (-0.5, 0.5) / ang (≈ ±30°) |
| `foot_friction` | startup | **sim2real 에서만 `(0.3, 1.2) → (0.2, 1.5)` 로 확장** |
| `encoder_bias` | startup | `(-0.015, 0.015)` rad |
| `base_com` | startup | x,y `(-0.025, 0.025)`, z `(-0.03, 0.03)` |

### 3-5. 시뮬레이션 / 통합기

`_COUPLED_SUBSTEPS=200` (decimation 200, physics dt 0.1 ms, policy dt 20 ms),
`_PD_RECOMPUTE=50` (PD 재계산 5 ms 주기), `method="A"` 모두 동일.
(`go2_constants.py:144-145`)

---

## 4. 요약 한눈에

```
[Motor params]   Kt/Ke 0.128 → 0.26,  R 0.3 → 0.66 Ω,  L 100→83 μs,  V_bus 24→30.8 V (nominal)
[Obs]            actor history 1 → 5,  delay 0 → 0–4 step (4 terms)
[Existing DR]    foot_friction (0.3,1.2) → (0.2,1.5)
[New DR (7)]     V_bus / PD gain ±20% / Kt=Ke ±10% / base mass ±/ link mass ±10% / joint zero bias ±0.03 rad / 8–12 s 외력
[Same]           Kp/Kd nominal, PPO hyperparams, 모든 reward weight, 기존 reset/push/encoder/base_com DR
```

## 5. 파일 인덱스

| 위치 | 내용 |
|---|---|
| `src/assets/robots/unitree_go2/go2_constants.py:186-189` | 모터 파라미터 (`_MA_MOTOR`) |
| `src/assets/robots/unitree_go2/go2_constants.py:220-227` | Method A Kp/Kd, effort_limit, armature |
| `src/assets/robots/unitree_go2/mj_native_electric_actuator.py:262-264, 424-427, 555-557` | per-env `_V_bus` 텐서 인프라 |
| `src/tasks/velocity/config/go2/env_cfgs.py:172-313` | MethodA-Electric obs delay + 7 DR 텀 (full DR cfg) |
| `src/tasks/velocity/config/go2/env_cfgs.py` | `unitree_go2_flat_methoda_electric_sim2real_env_cfg` (base + DR 7 개) |
| `src/tasks/velocity/config/go2/__init__.py` | `Unitree-Go2-Flat-MethodA-Electric` (base), `-Sim2Real`, `-PlayPD` 태스크 등록 |
| `src/tasks/velocity/mdp/events.py` | `randomize_V_bus`, `randomize_motor_strength`, `joint_pos_bias` 정의 |
| `src/tasks/velocity/mdp/__init__.py:3` | 위 events 모듈 re-export |
| `src/tasks/velocity/config/go2/rl_cfg.py` | PPO hyperparam (양 브랜치 동일) |
| `src/tasks/velocity/velocity_env_cfg.py:187-256` | base DR/event 텀 (양 브랜치 공유) |
| `src/tasks/velocity/velocity_env_cfg.py:258-355` | Reward weight (양 브랜치 공유) |
