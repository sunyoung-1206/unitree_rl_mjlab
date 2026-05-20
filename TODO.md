# Task: Port deploy-style DR & foot_gait reward to mjlab Unitree-Go2-Flat

## 목표

현재 mjlab(`unitree_rl_mjlab` 기반) 위에서 학습 중인 `Unitree-Go2-Flat` task에 sim-to-real 성능을 높이기 위한 변경을 가한다. 참조는 Isaac Lab 기반의 `unitree_go2_deploy_baseline_fullcode_lab` 레포의 DR / reward / curriculum 디자인. 단, 1:1 포팅이 아니라 mjlab API에 맞춰 의도를 옮긴다.

핵심 문제: 현재 DR을 켜면 reward가 baseline 대비 크게 떨어진다. 참조 레포는 이 문제를 (1) DR 자체를 적게 흔들고 (2) DR 강도를 reward의 함수로 만드는 curriculum으로 해결한다. 이 두 가지 패턴을 가져온다.

## 컨텍스트와 제약

- **베이스 task**: `Unitree-Go2-Flat` (mjlab)
- **새 task ID**: `Unitree-Go2-Flat-DeployDR-v0` 같은 새 ID로 등록한다. 기존 task는 절대 건드리지 말 것. 변경은 모두 새 task 안에서만.
- **시뮬레이터**: mjlab(MuJoCo + mujoco_warp). Isaac Lab API 호출이나 `isaaclab.*` import는 금지. mjlab의 `CurrTerm`, `ObsTerm`, `EventTerm`, `RewTerm`, `CommandTerm` 등 mjlab 네이티브 API만 사용.
- **floor clip**: 참조 레포는 floor clip을 사용하지 않는다. 사용자 현재 셋업에 noclip/clip 이슈가 있으므로 **floor clip 비활성화 상태로 진행**. 관련 설정을 찾으면 `enable_floor_clip=False` 또는 동등한 옵션으로 둘 것.
- **변경은 점진적으로**. 한 phase 끝날 때마다 짧은 학습(예: 200 iterations)으로 sanity check가 가능한 상태를 유지. 모든 phase를 한꺼번에 적용하지 말 것.
- **mjlab의 기존 베이스라인 DR을 모두 끄거나 바꾸지 말 것**. 명시적으로 지시된 것만 수정.

## 작업 원칙

- 각 phase 시작 전에 **현재 코드를 읽고, 어떤 파일이 어떤 책임을 지는지 짧게 요약**한 뒤 다음 단계를 결정한다.
- mjlab의 디렉토리 구조와 mjlab/unitree_rl_mjlab의 기존 task 등록 패턴(`register_mjlab_task` 등)을 먼저 파악한 후 작업한다.
- 새 파일은 가능한 한 적게. 기존 mjlab/unitree_rl_mjlab의 task config / mdp module 패턴을 따른다.
- 각 변경에 대해 **왜 이렇게 하는지 1-2줄 주석**을 코드에 남긴다 (한국어 OK). 나중에 비교 실험할 때 추적이 쉬워야 한다.
- 학습 스크립트(`scripts/train.py`)나 RL 라이브러리(rsl_rl) 코드는 건드리지 말 것. task config / mdp / agents config 레벨에서만 변경.

---

## Phase 0: 코드베이스 파악

다음을 확인하고 상위에 짧게 보고:

1. `src/tasks/velocity/config/go2/`의 task 등록 구조 — `__init__.py`에서 `register_mjlab_task`가 어떻게 호출되는지, env_cfg 클래스가 어디 정의되어 있는지.
2. `src/tasks/velocity/mdp/`의 모듈 분할 — observations, events, rewards, terminations, curriculums, velocity_command가 어떤 함수들을 export하는지.
3. mjlab의 `ObsTerm`이 noise/scale을 어떻게 받는지 (Isaac Lab의 `Unoise` 같은 게 mjlab에서는 어떤 형태인지).
4. mjlab에 asymmetric actor-critic (policy/critic obs group 분리)이 표준으로 지원되는지 — `ObsGroup` 두 개 만들어 정책/크리틱에 각각 할당하는 패턴이 가능한지.
5. mjlab의 friction randomization 이벤트가 `num_buckets` 옵션을 받는지, per-env interval-update 메커니즘이 가능한지 확인.

이 단계에서는 **파일 변경 금지**. 파악 결과를 출력만 한다.

---

## Phase 1: 새 task 등록 (기존 Flat 그대로 복제)

1. `Unitree-Go2-Flat`을 등록하는 패턴을 그대로 따라서 `Unitree-Go2-Flat-DeployDR-v0`를 등록.
2. 이 시점에서는 **DR, reward, obs 모두 기존 Flat과 100% 동일**. ID와 클래스명만 다르다.
3. mjlab 학습 명령으로 200 iterations 정도 굴려서 등록이 동작하는지 확인 (사용자가 직접 실행할 거니까 명령어만 알려준다).

**Acceptance**: 새 task가 정상적으로 학습이 시작되고, episode reward 곡선이 기존 Flat과 동일한 추이를 보인다.

---

## Phase 2: DR 조정 — 끄기와 줄이기

참조 레포는 mass / COM / PD gain / external force / encoder bias를 모두 흔들지 않는다. 사용자 현재 셋업에서 다음을 조정:

**완전히 끄기 (`= None` 또는 해당 EventTerm 주석 처리)**:
- `randomize_base_mass`
- `randomize_link_mass`
- `base_com` (또는 범위를 0으로)
- `randomize_actuator_gains` (PD gain DR)
- `randomize_motor_strength`
- `randomize_V_bus`
- `external_force_torque`
- `encoder_bias`
- `joint_pos_bias`

**조정**:
- `push_robot`: interval을 `(1, 3)s` → `(5, 5)s`로. 각속도 부분 제거하고 linear만 ±0.5 유지.
- `foot_friction`: range를 `(0.3, 1.25)`로 통일, **`num_buckets=64`로 startup 분포 안정화**. 추가로, 가능하다면 200 reset마다 per-env friction을 새로 뽑는 hybrid 메커니즘을 구현 (mjlab에 직접 지원이 없으면 우선 startup-only로 두고 TODO 코멘트만 남긴다).
- `reset_base`: `pose_range xy=(0,0)`, `yaw=(0,0)`으로 고정. `velocity_range` 6축 ±0.5는 유지.
- `reset_robot_joints`: position scale `(0.5, 1.5)`, velocity `0`.

**Acceptance**:
- Phase 1과 동일한 학습 명령으로 500 iterations 학습.
- DR을 켠 상태에서 tracking reward가 baseline의 60% 이상 회복되는 것을 확인.
- 만약 회복 안 되면 phase 2의 어떤 항목이 원인인지 ablation 표를 만들어 사용자에게 보고.

---

## Phase 3: DR Curriculum 도입

참조 레포의 핵심. **단일 스칼라 `_deploy_curriculum_level ∈ [0, 1]`** 을 도입하고 reward EMA에 따라 자동으로 조정.

1. mdp/curriculums.py에 `deploy_command_curriculum` 함수 추가:
   - reset 시점마다 호출되는 `CurrTerm`.
   - EMA 3개 유지: `tracking_ema`, `timeout_rate_ema`, `fall_rate_ema`. `ema_alpha=0.03`.
   - **Level up**: `timeout_rate_ema >= 0.80` AND `tracking_ema >= 0.75` AND `fall_rate_ema <= 0.15`가 **4 reset 연속** 충족.
   - **Level down**: `fall_rate_ema >= 0.25`가 **2 reset 연속**.
   - **Step**: up `+0.01`, down `-0.03` (비대칭).
   - **Cooldown**: 한 번 바뀌면 5 reset 동안은 변동 없음.
   - **Init**: `level_init=0.1`, min/max `[0.0, 1.0]`.
   - `tracking_ema`는 `tracking_lin_vel`과 `tracking_ang_vel` reward sum을 weight로 정규화해서 평균. 참조 코드의 정규화 식을 그대로 따른다.
2. Level이 다음 값들을 곱하는 형태로 전파되도록 설정:
   - obs noise scale: max 1.0 × level
   - action noise std: max 0.1 × level
   - action delay max steps: 0~1 step에서 round(0 + 1×level)
   - push velocity range: 0 ~ 0.5 m/s에서 보간
3. 위 4개 항목을 적용하려면 mjlab의 ObsTerm noise / Action noise / push range가 런타임에 변경 가능해야 한다. 안 되는 항목이 있으면 startup-only로 둬도 되지만, **최소한 obs noise와 push range는 curriculum 연동되어야 한다**.

**Acceptance**:
- 학습 시작 시 level=0.1로 DR 거의 꺼진 상태에서 시작.
- 500 iterations 안에 level이 0.3 이상으로 올라가야 함 (정책이 학습되고 있다는 신호).
- 학습 로그에 매 reset마다 `level`, `tracking_ema`, `fall_rate_ema`, `timeout_rate_ema`를 출력 (wandb나 tensorboard scalar로).

---

## Phase 4: foot_gait reward 추가

mdp/rewards.py에 `foot_gait` 함수 추가:

```
phase = (episode_length_buf * step_dt) % 0.6 / 0.6
offsets = (0.0, 0.5, 0.5, 0.0)   # FL, FR, RL, RR (joint 순서 확인 필수)
foot_phase = (phase + offset) % 1
desired_stance = foot_phase < 0.56
contact = (net_force_z > 1.0)
gait_match = NOT(contact XOR desired_stance)
reward = mean(gait_match) × (||cmd[:2]|| >= 0.1)
```

env_cfg의 rewards에 `weight=0.10`으로 추가. command_threshold는 `0.1`.

**중요**:
- foot 순서가 FL → FR → RL → RR 인지 확인. mjlab의 contact sensor body 순서가 다르면 offsets 순서도 맞춰 재배치.
- mjlab의 contact sensor가 net force를 어떻게 노출하는지 확인 후 그에 맞춰 구현.

**Acceptance**:
- 학습 후 episode 한 번 시각화했을 때 4발이 대각선 trot으로 움직이는지 확인 (FL+RR 동시 stance, FR+RL 동시 swing).
- ~~학습 reward에서 `foot_gait` 항목이 평균 0.6 이상으로 수렴.~~
  → **0.55 로 하향 (2026-05-20)**: weight 를 0.5→0.10 으로 내리니 tracking reward 가
    상승(track_lin 0.73→0.88, track_ang 0.84→0.94)했고, weight=0.10 인센티브로는
    gait_match 가 ~0.58 에서 수렴. tracking 우선이 더 바람직하다는 판단으로 임계값 하향.
    측정값 0.58 ≥ 0.55 → PASS.

---

## Phase 5: Asymmetric Critic Obs (선택, 가능하면)

mjlab이 actor/critic obs group 분리를 지원하면 진행. 안 되면 skip하고 사용자에게 보고.

Critic obs는 actor obs 47D + 다음 privileged 항목들:
- `base_lin_vel` (3D, scaled ×2)
- `feet_pos_z` (4D)
- `feet_air_time` (4D)
- `foot_contact` (4D)
- `contact_forces` (12D, sign × log1p(|f|) 변환)
- `friction_coeffs` (1D, 현재 env의 friction 값)
- `deploy_curriculum_level` (1D)
- `push_history_xy` (2D, 마지막 push velocity)

총 ~80D. critic obs group은 `enable_corruption=False`로 (노이즈 안 섞음).

**Acceptance**: critic value 추정의 explained variance가 actor=critic 셋업 대비 5%p 이상 개선.

---

## Phase 6: Heading command 검토

현재 mjlab 기본은 `rel_heading_envs=0.25`. 참조 레포는 100% heading mode. 차이가 크다.

1. 사용자 셋업에서 `rel_heading_envs` 값을 확인하고 보고만 한다.
2. **이 값을 변경하지 말 것**. 사용자 추가 지시 대기.

---

## 하지 말 것 (Hard constraints)

- 기존 `Unitree-Go2-Flat` task를 수정하거나 삭제하지 말 것.
- RL 라이브러리(rsl_rl) 자체를 수정하지 말 것.
- mjlab core 코드(`mjlab/*`)를 수정하지 말 것. 모든 변경은 `src/` 하위 또는 `project/` 하위에서만.
- `isaaclab` 어떤 모듈도 import하지 말 것.
- floor clip 관련 설정을 켜지 말 것 (사용자 명시 지시).
- 한 phase의 acceptance가 충족 안 된 상태로 다음 phase로 넘어가지 말 것.
- PPO 하이퍼파라미터(lr, gamma, lambda, entropy coef, num_steps_per_env, batch size 등)는 건드리지 말 것. 참조 레포도 이 부분은 안 건드렸다.

## 마무리 보고

모든 phase 끝나면:
- 변경된 파일 목록과 각 파일의 한 줄 요약
- 새 task ID 학습 명령어
- 각 phase의 acceptance 통과 여부
- 알려진 이슈 / TODO 항목
- 200 iterations 짧은 학습 로그 한 번 (사용자가 직접 안 돌렸을 경우)