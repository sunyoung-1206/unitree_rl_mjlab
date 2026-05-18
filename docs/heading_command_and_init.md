# Heading command / Heading lock / Zero-init 정리

Velocity task 에서 yaw (= heading) 명령을 다루는 세 가지 메커니즘을 정리합니다.

1. **학습 시 25 % heading-command 분포** — 한 정책이 "직접 wz 트래킹" + "yaw 유지"
   두 가지 사용 패턴을 모두 학습하게 만드는 트릭.
2. **추론 시 heading-lock 래퍼** — 학습 분포의 heading-controlled mode 를 그대로
   재현해 운영 시 stick 을 떼면 yaw 가 고정되도록 만드는 헬퍼.
3. **Zero-init 래퍼** — 평가/비교 실험을 위해 reset 직후 결정론적 0 상태로 강제.

---

## 1. 학습 시 heading command — `rel_heading_envs = 0.25`

### 1-1. 설정 위치

**파일**: `src/tasks/velocity/velocity_env_cfg.py:165-181`

```python
commands: dict[str, CommandTermCfg] = {
  "twist": UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),       # 3-8 초 간격으로 명령 리샘플
    rel_standing_envs=0.05,                 # 5 % envs: 명령 모두 0 (정지)
    rel_heading_envs=0.25,                  # ← 25 % envs: heading-controlled
    heading_command=True,
    heading_control_stiffness=0.5,          # K_h
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 2.0),
      lin_vel_y=(-1.0, 1.0),
      ang_vel_z=(-1.0, 1.0),
      heading=(-math.pi, math.pi),          # heading_target sample 범위
    ),
  )
}
```

mjlab 측 default 값은 **두 레이어**로 갈라져 있습니다. 헷갈리지 않게 분리해서 비교:

- **레이어 ①** — dataclass 자체의 default (`UniformVelocityCommandCfg` 의 필드 기본값).
  cfg 에서 키워드를 *생략* 했을 때 적용되는 값.
  소스: `mjlab/tasks/velocity/mdp/velocity_command.py:251-256`
- **레이어 ②** — mjlab 이 예시로 제공하는 base velocity env cfg.
  소스: `mjlab/tasks/velocity/velocity_env_cfg.py:157-173`
- **우리 값** — 우리 repo 의 base velocity env cfg (= 레이어 ② 를 fork 해서 override).
  소스: `src/tasks/velocity/velocity_env_cfg.py:165-181`

| 항목 | ① dataclass default | ② mjlab velocity_env_cfg | **우리 (`src/...`)** |
|---|---|---|---|
| `rel_heading_envs`        | `1.0` | `0.3`  | **`0.25`** |
| `rel_standing_envs`       | `0.0` | `0.1`  | **`0.05`** |
| `heading_control_stiffness` | `1.0` | `0.5`  | `0.5` (동일) |
| `resampling_time_range`   | (필수) | `(3.0, 8.0)` 초 | `(3.0, 8.0)` 초 (동일) |
| `ang_vel_z` 범위          | (필수) | `(-0.5, 0.5)` | **`(-1.0, 1.0)`** |
| `lin_vel_x` 범위          | (필수) | `(-1.0, 1.0)` | **`(-1.0, 2.0)`** |

세 질문에 대한 직접 답:

- **"우리는 0.25 를 쓴다?"** — 네. mjlab 의 *dataclass* default 가 `1.0` 인 건
  단지 "cfg 에서 키를 생략하면" 이 값. mjlab 의 *예시* velocity_env_cfg 는 `0.3`
  을 명시 지정. 우리 repo 는 mjlab 예시를 그대로 쓰지 않고 한 번 더 낮춰서 `0.25`.
- **"velocity_env_cfg 는 0.3, 우리는 0.25 인가?"** — 정확히 그렇습니다. 위 표의 ②
  열이 0.3, 우리 열이 0.25. 0.05 만큼 더 낮춰 heading-controlled env 비율을 줄였습니다
  (= direct-wz 트래킹 학습 비중을 75 % 로 약간 늘림).
- **"`rel_standing_envs` 는 임의로 5 % 로?"** — 네. dataclass default `0.0` 도,
  mjlab 예시 `0.1` 도 아닌 `0.05` 를 우리 repo 가 선택. 학습 시 5 % env 는
  `(vx, vy, wz) = (0, 0, 0)` 명령으로 "가만히 서 있기" 를 학습 — 명령이 모두 0 인
  분포가 너무 많으면 (10 %) action_rate / pose 페널티가 트래킹 보상보다 강해져
  보행 학습이 둔해질 수 있어 5 % 로 절충한 값입니다.

### 1-2. 동작 메커니즘

**파일**: `src/tasks/velocity/mdp/velocity_command.py:72-107`

**(a) Resample 시점 (3-8 초 마다, env 별 독립)** — `_resample_command`:

```python
vel_command_b[env_ids, 0] = uniform(lin_vel_x)
vel_command_b[env_ids, 1] = uniform(lin_vel_y)
vel_command_b[env_ids, 2] = uniform(ang_vel_z)         # ← 직접 wz sample
if heading_command:
    heading_target[env_ids] = uniform(-π, π)            # ← yaw target sample
    is_heading_env[env_ids] = (uniform(0,1) <= 0.25)    # ← 25 % 만 True
is_standing_env[env_ids]  = (uniform(0,1) <= 0.05)
```

→ 매 resample 마다 env 별로 두 가지가 동시에 일어남:
- `vel_command_b[:, 2]` (= `wz`) 가 `(-1, 1)` 에서 직접 샘플됨
- 25 % 확률로 그 env 의 `is_heading_env = True`

**(b) 매 step (`_update_command`)** — heading-controlled env 의 wz 덮어쓰기:

```python
if heading_command:
    heading_error = wrap_to_pi(heading_target - robot.data.heading_w)
    env_ids = is_heading_env.nonzero().flatten()
    vel_command_b[env_ids, 2] = clip(                   # ← wz 를 매 step 재계산
        K_h * heading_error[env_ids],
        ang_vel_z[0], ang_vel_z[1]                      # (-1, 1) 로 clip
    )
```

→ 25 % env 는 매 step **`wz_cmd = clip(0.5 · (target_yaw - current_yaw), -1, 1)`** 로
P-제어된 wz 가 명령으로 들어가고, 나머지 75 % env 는 resample 시 뽑힌
`wz_cmd` 가 다음 resample 까지 그대로 유지됩니다.

**(c) Reward 트래킹** — `src/tasks/velocity/mdp/rewards.py:43-60`

```python
def track_angular_velocity(env, std, command_name, ...):
    command = env.command_manager.get_command(command_name)
    actual  = asset.data.root_link_ang_vel_b
    z_error  = (command[:, 2] - actual[:, 2])²          # ← wz 트래킹
    xy_error = (actual[:, :2])²                          # roll/pitch ω 패널티
    return exp(-(z_error + 0.05·xy_error) / std²)
```

reward 는 `command[:, 2]` (= `vel_command_b[:, 2]`) 를 비교 대상으로 쓰기 때문에,
heading env 든 wz-직접 env 든 같은 함수가 정확히 active 한 명령에 대해 보상을 줍니다.

### 1-3. 왜 25 % 만 heading-controlled 로 학습하는가

핵심: **하나의 정책이 운영 시 두 가지 명령 패턴을 모두 다뤄야 하기 때문**.

운영 시 외부에서 들어오는 yaw 관련 명령은 크게 두 종류:

| 패턴 | 사용 시나리오 | 정책 입력 (`wz_cmd`) 의 특성 |
|---|---|---|
| **A. Direct teleop** | joystick 스틱 우/좌로 회전 명령 | step 마다 작업자가 임의로 변할 수 있는 wz. 명령이 robot pose 와 무관 |
| **B. Heading-lock** | 스틱을 놓고 직진/정지 — yaw 가 드리프트 없이 유지되길 원함 | wz 가 `K_h · (target − current_yaw)` 형태로 robot 의 현재 yaw 에 의존, heading 이 수렴하면 0 으로 감쇠 |

두 분포가 정성적으로 다르기 때문에 한쪽만 학습하면 다른 쪽에서 망가집니다:

- **100 % 직접 wz** 로만 학습 → 운영 시 heading-lock 을 켜면 wz 가 yaw 오차에 비례해서
  "작아졌다 커졌다" 하는 패턴을 본 적이 없음 → wz 가 0 근처일 때 발이 미끄러지거나
  base 가 회전 의도 없이 흔들리는 식으로 망가짐.
- **100 % heading-controlled** 로만 학습 → 운영 시 큰 wz 명령 (예: 빠른 제자리
  회전) 을 본 적 없어 추종 실패.

25 % heading + 75 % direct wz 는 두 분포를 동시에 노출시키는 가장 단순한 mixture:
- 직접 wz 큰 값 (75 %): full ang_vel_z 범위 학습.
- heading-driven 작은 wz (25 %): yaw 가 target 에 가까워질수록 자연히 wz → 0 으로
  감쇠하는 패턴 학습 → 정지 시 heading 유지 능력 확보.

`heading_control_stiffness = K_h = 0.5` 는 그 결과로 정책이 보게 되는 wz 의 분포 폭을
결정. K_h 가 너무 크면 wz_cmd 가 ang_vel_z range 끝값에 자주 saturate 되어 direct
wz 분포와 구별이 없어지고, 너무 작으면 영원히 0 근처 → heading 수렴이 느려져 reward
신호가 약해집니다. mjlab default 1.0 → 0.5 로 낮춘 것은 heading 수렴을 부드럽게
만들고 wz 분포의 평균을 0 근처로 좁히기 위함.

> Tip — 디버깅 시 확인할 텐서:
> - `is_heading_env` (bool [num_envs]): 어느 env 가 P-제어되는지
> - `heading_target`, `heading_error` ([num_envs]): target / error 로깅 가능
> - 두 모드 사이에 reward 가 비대칭이 되어 보이면 K_h, rel_heading_envs 둘 다 영향 줌.

### 1-4. 자주 하는 오해 — "wz 가 안 주어진 env" 가 heading env 인가?

**오해**: 운영 시 매핑을 거꾸로 끌어와 "rel_heading_envs 는 vx/vy 만 주어지고 wz
가 안 주어진 env 를 위한 분기" 라고 생각하기 쉬움.

**실제 학습 코드의 분기 규칙**:

| 단계 | 동작 | 분기 조건 |
|---|---|---|
| Resample (3-8 초 마다) | `vx`, `vy`, **`wz`** 를 *모든 env* 에서 unconditional 하게 sample | 없음 — wz 도 항상 sample |
| Resample 직후 | `is_heading_env = uniform(0,1) <= 0.25` | **wz 값과 무관한 독립 Bernoulli** |
| 매 step | `is_heading_env=True` 인 env 만 `wz_cmd ← K_h·heading_error` 로 *덮어씀* | resample 때 들어 있던 wz 를 버리고 P-제어 출력으로 교체 |

→ 즉 학습 시 "wz 가 주어졌는지" 로 분기하는 코드는 없습니다. 모든 env 가 resample
때 wz 를 한 번씩 받아 두고, **그중 25 % 가 추후 P-제어 결과로 덮어씌워짐**.

큰 wz (예: 0.9) 가 뽑힌 env 도 25 % 확률로 heading-controlled 가 되면 그 큰 wz 는
다음 step 부터 무시되고 `K_h·(target − current_yaw)` 로 바뀝니다. "작은 wz 면 heading,
큰 wz 면 manual" 같은 *조건부* 분기는 학습 코드에는 존재하지 않습니다.

**그래도 사용자의 직관은 *기능적으로*는 맞음**:

| 운영 시 시나리오 | 학습 분포 어디에 해당하는가 |
|---|---|
| vx/vy 만 주고 wz=0 (스틱 떼고 직진/정지) → yaw 가 자동 유지 | **25 % heading env**: wz_cmd 가 yaw error 의 함수가 되는 분포 |
| vx/vy/wz 다 주는 직접 teleop (스틱으로 회전) | **75 % non-heading env**: wz_cmd 가 resample 때 뽑힌 임의값으로 고정 |

운영 시 이 두 분포를 사용자가 *선택* 하는 코드가 `heading_lock.py` 입니다. 거기서는
`|wz_user| < 0.1` 임계로 분기를 거는데, 이건 학습 코드에는 없는 *추가* 룰입니다 —
학습은 그냥 25/75 mixture 를 무조건 노출하고, 운영은 사용자 의도에 따라 둘 중 하나를
강제로 활성화시킵니다.

> 정리: 두 모드의 **목적** 은 "wz 가 주어졌느냐 아니냐" 로 보는 게 정확히 맞지만,
> 학습 코드의 **분기 변수** 는 wz 값이 아니라 독립적인 25 % Bernoulli 추첨입니다.
> 두 분포를 mixture 로 노출시키기만 하면 정책이 두 사용 패턴 모두에 일반화하므로
> 굳이 학습 시 조건부 분기를 둘 필요가 없습니다.

### 1-5. `resampling_time_range` 와 termination 의 상호작용

**질문**: 3-8 초 타이머가 다 가기 전에 로봇이 넘어지면 어떻게 되나?
**답**: 그 env 는 즉시 reset 되고, **명령 타이머와 명령값도 같이 새로 샘플됨**.
다음 episode 가 "이전 episode 의 남은 시간"을 이어받는 일은 없습니다.

코드 흐름 (`mjlab/envs/manager_based_rl_env.py:381-403`):

```python
# 매 env step 끝에:
reset_buf = termination_manager.compute()           # 넘어진 env 찾기
reset_env_ids = reset_buf.nonzero()
if len(reset_env_ids) > 0:
    self._reset_idx(reset_env_ids)                  # ← 넘어진 env 만 reset
...
self.command_manager.compute(dt=self.step_dt)        # 정상 env 의 time_left 감소
```

`_reset_idx` 안에서 `command_manager.reset(env_ids)` 가 호출되고
(`manager_based_rl_env.py:518`), 그 결과 각 command term 의 `reset()` 이 다음을
수행합니다 (`mjlab/managers/command_manager.py:87-95, 105-111`):

```python
def reset(self, env_ids):
    ...
    self.command_counter[env_ids] = 0
    self._resample(env_ids)                          # ← 즉시 새 명령 + 새 timer

def _resample(self, env_ids):
    self.time_left[env_ids] = uniform(3.0, 8.0)       # ← 타이머 리셋
    self._resample_command(env_ids)                   # ← lin/ang vel, heading 새로 sample
```

이 흐름이 갖는 의미:

| 시점 | 일어나는 일 |
|---|---|
| step t — 로봇 넘어짐 (terminated) | 해당 env 의 `reset_buf[i] = True` |
| 같은 step | `_reset_idx(i)` → robot state 리셋 + `command_manager.reset(i)` → `time_left[i]` 가 `uniform(3, 8)` 로 갱신, `vel_command_b[i, :]` / `heading_target[i]` / `is_heading_env[i]` 모두 새로 샘플 |
| step t+1 | 새 명령 / 새 타이머로 새 episode 시작 |

추가 함의:

- **명령 시간은 episode 와 묶임**: 즉 episode 길이가 짧으면 (자주 넘어지면)
  정책이 한 명령에 노출되는 시간이 평균적으로 짧아짐. 학습 초기 (자주 termination
  발생) → "명령이 자주 바뀌는 distribution" 으로 보일 수 있음.
- **명령 시간이 다 가는 정상 진행 경로**: 한 env 의 `time_left` 가 0 이하가 되면
  `compute()` 안의 `(time_left <= 0).nonzero()` 분기로 *그 env 만* `_resample` 됨.
  넘어지지 않은 채 명령만 갈아끼우는 경로. (`command_manager.py:97-103`)
- **`init_velocity_prob`** 옵션 (`velocity_command.py:84-97`): resample 시 일정 확률로
  base velocity 를 새 명령에 맞춰 초기화해 "이미 그 속도로 가는 중" 인 상태로
  시작시키는 기능. 우리 cfg 에서는 default `0.0` 이라 비활성.
- **TimeOut termination**: termination_manager 가 `terminated` 와 `time_outs` 를
  분리해서 추적. 시간 초과로 reset 되든 넘어져서 reset 되든 `_reset_idx` 경로는
  같으므로 명령 재샘플 거동도 동일.

> 즉 "3-8 초"는 *한 env 가 그 명령을 받는 최대 시간*이 아니라 *정상적으로 살아 있을
> 때 명령이 갈리는 평균 간격*. termination 이 명령을 끊는 또 하나의 경로입니다.

---

## 2. 추론 시 heading-lock 헬퍼

### 2-1. 위치와 호출 경로

**파일**: `src/utils/heading_lock.py`
호출: `scripts/play.py:68-82` (`_apply_fixed_velocity` 가 위임),
`results/.../run_demag_experiment.py` 등의 평가 스크립트도 공유.

```python
apply_heading_lock_velocity(
    env,
    vx=..., vy=..., wz=...,
    target_heading=None,              # 명시 target 없으면 첫 step 의 base yaw 사용
    heading_threshold=0.1,            # |wz| < 임계면 heading-lock
    no_heading_control=False,         # True 면 분기 끄고 wz 그대로
)
```

### 2-2. 무엇을 하는가 — 핵심 한 줄

**학습에서 25 % env 가 보던 분포 (heading-controlled wz) 와 100 % 동일한 신호를
모든 env 에 강제 주입**해서, stick 을 놓아도 정책이 *학습 분포 안에서* yaw 를
유지하도록 만드는 래퍼.

구체적으로 매 step 다음을 수행한 뒤 *원본* `_update_command` 를 호출합니다 — wz
계산을 mjlab 라이브러리에 위임하므로 학습과 정확히 같은 수식 (`K_h·error`, clip)
이 적용됩니다.

```python
term.vel_command_b[:, 0] = vx_user
term.vel_command_b[:, 1] = vy_user
term.is_standing_env[:]  = False

if |wz_user| >= threshold:    # manual mode
    term.is_heading_env[:]    = False
    term.vel_command_b[:, 2]  = wz_user
else:                          # heading-lock mode
    term.is_heading_env[:]    = True
    term.heading_target[:]    = stored_target
    # ↓ original_update() 이 vel_command_b[:, 2] 를 K_h·wrap_to_pi(target-heading_w)
    #   로 덮어씀 (= 학습 시 wz 계산과 동일 코드 경로)
```

### 2-3. 세부 동작

| 상황 | 동작 |
|---|---|
| 초기 `target_heading` 미지정 | 첫 step 의 `robot.data.heading_w` 를 자동 캡처해 target 으로 저장 (`heading_lock.py:122-124`) |
| Manual → heading_control 전환 (스틱 떼는 순간) | 현재 yaw 를 새 target 으로 갱신 → 잠금 위치가 "스틱 놓은 그 순간의 자세" 가 됨 (`:130-132`) |
| `_resample_command` 우회 | `lambda env_ids: None` 으로 덮어씀 — 학습 코드의 3-8 초 랜덤 재샘플이 추론 중에 작동하지 않게 차단 (`:70`) |
| `no_heading_control=True` (비교 실험용) | mode 분기 없이 `is_heading_env=False` 로 고정하고 `wz_user` 를 그대로 넣음. heading-lock 도입 효과를 측정할 때 baseline |
| `debug_cmd=True` | 50 step 마다 mode / cmd / target_yaw / cur_yaw 를 stdout 으로 (`:149-161`) |

### 2-4. 학습 분포와의 일대일 매칭

| 학습 (25 % env) | 추론 (heading_lock 분기 B) |
|---|---|
| `is_heading_env=True` | 동일 |
| `heading_target = uniform(-π, π)` | `stored_target` (첫 step 자동 캡처 또는 명시값) |
| `_update_command` 이 매 step `wz = clip(K_h · wrap_to_pi(target − heading_w), -1, 1)` | **동일 코드 경로** (`original_update()` 호출) |

이 일대일 매칭이 깨지면 (예: `K_h` 를 무시하고 직접 `wz=0` 박기) 정책이 학습 시 본 적
없는 분포에 노출됩니다. 그래서 헬퍼가 굳이 mjlab 의 `_update_command` 를 그대로
재사용하도록 설계됨.

---

## 3. Zero-init 래퍼

### 3-1. 위치와 호출

**파일**: `src/utils/init_state.py`
호출: `scripts/play.py:167-170` — `--zero-init` 플래그 (`PlayConfig.zero_init`) 가
켜져 있을 때 한 번 호출.

```python
if cfg.zero_init:
    from src.utils.init_state import apply_zero_initial_state
    apply_zero_initial_state(env)
```

### 3-2. 왜 도입했는가

**원인**: mjlab 의 base velocity cfg 에서 `reset_base` 이벤트가 reset 마다 base 의
yaw 를 `uniform(-π, π)` 로 무작위 샘플합니다
(`src/tasks/velocity/velocity_env_cfg.py:188-200`,
`pose_range.yaw = (-3.14, 3.14)`).

이 때문에:
- **같은 seed 로도** reset 마다 시작 yaw 가 다름 → run 간 비교 실험이 어려움.
- 평가 시 "정확히 같은 초기 자세에서 정책 A vs B" 같은 페어 비교 불가능.
- heading-lock 디버깅 시 target 캡처 결과가 매번 달라져 추적이 까다로움.

학습용 randomization 은 그대로 둬야 하지만 (DR 의 일부), 평가/play 에서는
결정론적이어야 한다는 요구라서 cfg 변경 대신 *런타임 패치* 방식을 택함.

### 3-3. 무엇을 강제하는가

`_write_zero_state(inner_env, env_ids)` 가 reset 직후 다음을 덮어씁니다
(`init_state.py:32-65`):

| 필드 | 강제 값 | 비고 |
|---|---|---|
| `root_state[:, 0:2]` (x, y) | `0.0, 0.0` | 평면 좌표 원점 |
| `root_state[:, 2]` (z)     | `default_root_state[:, 2]` (= `init_state.pos[2]`) | 서있는 높이는 보존 |
| `root_state[:, 3:7]` (quat)| `(1, 0, 0, 0)` | identity → roll=pitch=yaw=0 |
| `root_state[:, 7:13]`      | `0.0 × 6` | linear + angular velocity 전부 0 |
| joint pos | `default_joint_pos` | startup DR 에 의해 env 별로 다를 수 있음 (joint_pos_bias) — 그 분포는 그대로 |
| joint vel | `0.0` | 전부 정지 |

> joint pos 만 `default_joint_pos` (= entity.data.default_joint_pos) 를 그대로
> 사용하므로 **sim2real-DR-v1 의 `joint_pos_bias` startup 이벤트가 만든 env 별
> mechanical zero offset 은 zero-init 후에도 유지**됩니다 (= zero-init 은 "조립
> 오차가 있는 로봇을 그 오차 그대로 똑바로 세워" 시작시키는 것).

### 3-4. `_reset_idx` 래핑

단순히 한 번 덮어쓰는 게 아니라, `ManagerBasedRlEnv._reset_idx` 자체를 monkey-patch
해서 **이후 발생하는 모든 reset 직후 자동 적용** (`init_state.py:100-106`):

```python
def patched_reset_idx(env_ids=None):
    original_reset_idx(env_ids)   # 1) 라이브러리 reset 진행 (random yaw 포함)
    _write_zero_state(inner, env_ids=env_ids)  # 2) 결정론적 0 상태로 덮어쓰기
```

- viewer.run() 안에서 reset 이 일어나든, wrapper.reset() 이 명시 호출되든 동일 적용.
- `_zero_init_patched` 플래그로 **idempotent** — 중복 호출해도 패치는 한 번만.
- 즉시 한 번 적용 (`_write_zero_state` 직접 호출) 도 함께 수행해 *이미 reset 되어
  있는* 환경의 상태도 덮어씀.

### 3-5. 사용 예

```bash
# 평가 — 결정론적 시작
python scripts/play.py --task Unitree-Go2-Flat-MethodA-Electric --zero-init \
       --vx 0.5 --vy 0.0 --wz 0.0           # heading-lock 자동 (|wz|<0.1)

# heading-lock 끄고 manual wz 비교
python scripts/play.py --task ... --zero-init \
       --vx 0.5 --wz 0.0 --no-heading-control

# Demag/Ke-sweep 같은 페어 비교
#   동일 starting yaw=0 에서 파라미터만 바꿔 가며 trajectory 차이 확인.
```

---

## 4. 한 줄 요약

```
[Training]  25 % env 는 매 step wz = 0.5·(target_yaw − current_yaw), 75 % 는 직접 sample
            → 한 정책이 "직접 wz 추종" + "스틱 놓고 yaw 유지" 두 분포 모두 학습
[Inference] heading_lock 헬퍼가 동일 코드 경로 (mjlab original_update) 로 학습 분포 B
            를 재현. |wz_user|≥0.1 면 manual, 미만이면 heading-lock 자동 분기
[Init]      zero_init 헬퍼가 _reset_idx 를 패치 → 모든 reset 직후 pos/quat/vel = 0,
            joint = default_joint_pos. 결정론적 페어 비교 가능
```

## 5. 파일 인덱스

| 위치 | 내용 |
|---|---|
| `src/tasks/velocity/velocity_env_cfg.py:165-181` | twist 커맨드 cfg (`rel_heading_envs=0.25`, `K_h=0.5`, resample 3–8 s) |
| `src/tasks/velocity/mdp/velocity_command.py:72-82` | `_resample_command` — 25 % `is_heading_env=True` 샘플링 |
| `src/tasks/velocity/mdp/velocity_command.py:99-107` | `_update_command` — heading env 의 `wz = clip(K_h·error, ...)` |
| `src/tasks/velocity/mdp/rewards.py:43-60` | `track_angular_velocity` — `command[:, 2]` 비교 |
| `src/utils/heading_lock.py` | 추론용 heading-lock / manual 분기 헬퍼 |
| `scripts/play.py:68-82` | heading_lock 호출 어댑터 |
| `src/utils/init_state.py` | zero-init `_reset_idx` 패치 |
| `scripts/play.py:167-170` | zero-init 호출 (CLI 플래그) |
| `mjlab/tasks/velocity/mdp/velocity_command.py` | mjlab 측 원본 구현 (참고용) |
