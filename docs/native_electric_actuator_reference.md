# NativeElectricActuator 레퍼런스

`src/assets/robots/unitree_go2/mj_native_electric_actuator.py` 의 액추에이터 cfg와 `src/assets/robots/unitree_go2/go2_constants.py` 에 정의된 인스턴스, 각 method (A / A+ / B), 그리고 `scripts/train.py` / `scripts/play.py` 에서의 사용법을 정리한다.

---

## 1. 클래스 상속 관계

```
ActuatorCfg                   (mjlab/actuator/actuator.py)
└── IdealPdActuatorCfg        (mjlab/actuator/pd_actuator.py)
    └── DcMotorActuatorCfg    (mjlab/actuator/dc_actuator.py)
        └── NativeElectricActuatorCfg   (src/.../mj_native_electric_actuator.py)
```

`build()` 메서드는 각 cfg가 자기 짝의 액추에이터 객체를 반환하므로, 런타임 인스턴스 클래스도 같은 라인을 따른다.

```
Actuator → IdealPdActuator → DcMotorActuator → NativeElectricActuator
```

---

## 2. NativeElectricActuatorCfg 전체 필드

상속 단계별로 표기. 모든 필드는 `@dataclass(kw_only=True)` — 키워드 인자로만 전달.

### 2.1 ActuatorCfg (모든 액추에이터 공통)

| 필드 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `target_names_expr` | tuple[str, ...] | (필수) | 매칭할 관절 이름(또는 정규식) 튜플 |
| `transmission_type` | TransmissionType | JOINT | 전달 방식 (JOINT / TENDON / SITE) |
| `armature` | float | 0.0 | 모터 회전자 reflected inertia |
| `frictionloss` | float | 0.0 | 정마찰 한계 |

### 2.2 IdealPdActuatorCfg (PD 게인)

| 필드 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `stiffness` | float | (필수) | PD Kp |
| `damping` | float | (필수) | PD Kd |
| `effort_limit` | float | inf | 출력 토크 상한 [N·m] |

### 2.3 DcMotorActuatorCfg (DC 모터 토크-속도 곡선 포화)

| 필드 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `saturation_effort` | float | (필수) | stall(=정지) 토크 [N·m]. 토크-속도 곡선 절편 |
| `velocity_limit` | float | (필수) | no-load 속도 [rad/s]. 토크-속도 곡선 0점 |

### 2.4 NativeElectricActuatorCfg (전기 모터 모델 + 토크 적분 루프)

#### 모터/전기 파라미터

| 필드 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `Kt` | float | 0.128 | 토크 상수 [N·m/A] (모터축 기준, 기어비 별도) |
| `Ke` | float | 0.128 | 역기전력 상수 [V·s/rad_motor] |
| `R` | float | 0.3 | 권선 저항 [Ω] |
| `L` | float | 1e-4 | 권선 인덕턴스 [H] |
| `gear_ratio` | float | 6.33 | 감속비 |
| `V_bus` | float | inf | 버스 전압 한계 [V]. inf이면 back-EMF 보상 무제한 |

#### 시간 위계 / 서브스테핑

| 필드 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `substeps` | int | 1 | policy step당 physics sub-steps (= decimation) |
| `pd_substeps` | int | 0 | PD 재계산 주기 (physics step 단위). 0이면 policy step 경계에서만 |

#### 적분기 / coupling 옵션

| 필드 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `use_callback` | bool | False | True → `dyntype=user` (Python callback, CPU only); False → `dyntype=filterexact` (mjwarp 호환) |
| `use_coupled` | bool | False | True → Schur complement 기반 cross-Jacobian 항을 implicit solver에 추가. coupled 모드 활성화 |
| `method` | Literal["A", "A+", "B"] | "A+" | coupled 모드에서 사용할 적분기. dynprm[4]에 인코딩되어 patched mjwarp 가 읽음 |

#### 토크 추종 적분 제어 루프 (드라이버 5 ms 게이팅, 신규)

| 필드 | 타입 | 기본값 | 의미 |
|---|---|---|---|
| `use_torque_loop` | bool | False | True → 드라이버에 적분 루프 추가. False → `I_cmd = τ_des / Kt_nom·gr` (기존 동작) |
| `Ki` | float | 50.0 | 적분 게인 [(A·s)/(N·m·s)]. `use_torque_loop=True`일 때만 유효 |
| `integral_max` | float \| None | None | anti-windup 클램프 [A·s]. None → `(I_max − τ_cmd_max/Kt_nom·gr)/Ki` 자동, headroom ≤ 0이면 0.2 fallback |

---

## 3. method A / A+ / B (coupled 모드 적분기)

`use_coupled=True` 일 때만 의미 있음. dynprm[4] 슬롯에 인코딩되어 patched mjwarp 의 `FILTEREXACT` 커널이 읽는다.

| method | β_int (integrator) | β_imp (Schur/Force RHS) | dynprm[4] | CPU/GPU | 의미 |
|---|---|---|---|---|---|
| **A** | 1/(1+h/τ) (BE) | 1/(1+h/τ) (BE) | 0.0 | 양쪽 | 전부 BE consistent. mjwarp / patched 양쪽에서 정합. 가장 보수적 |
| **A+** | exp(−h/τ) (ZOH) | exp(−h/τ) (ZOH) | 1.0 | 양쪽 | 전부 ZOH consistent. 1차 선형 ODE의 해석해. 권장 (기본값) |
| **B** | exp(−h/τ) (ZOH) | 1/(1+h/τ) (BE) | 2.0 | **GPU 전용** | Integrator는 ZOH, Schur/Force RHS만 BE. patched mjwarp에서만 의미 있음 |

> **CPU 주의**: stock CPU MuJoCo는 dynprm[4]를 무시하고 항상 ZOH integrator를 쓰며 Schur/Force RHS 보정 자체가 patched mjwarp에만 존재. 따라서 CPU에서는 A/A+/B 모두 ZOH integrator + (Schur/Force 보정 없음) 으로 동작 → 학습은 GPU 권장.

> **dynprm[3] 별도**: `dynprm[3]` 은 demag 기준 `Ke_nom·gr` 보존용. 절대 수정 금지 (감자 주입은 `dynprm[1]`만 변경).

---

## 4. go2_constants.py 인스턴스 정리

각 그룹은 (HIP, THIGH, CALF) 3개 묶음으로 정의 + `*_ARTICULATION` 묶음 + `get_*_robot_cfg()` 팩토리 함수.

공통 모터 파라미터 (전 그룹 동일): `Kt = Ke = 0.128, R = 0.3, L = 1e-4, gear_ratio = 6.33`.
HIP/THIGH: `effort_limit = saturation_effort = 23.5, armature = 0.01`.
CALF: `effort_limit = saturation_effort = 45.0, armature = 0.02`.

| 그룹 | 클래스 | 핵심 차이 | substeps / pd_substeps | 팩토리 |
|---|---|---|---|---|
| `GO2_ACTUATOR_*` | `BuiltinPositionActuatorCfg` | 기본 PD only (전기 ODE 없음) | n/a | `get_go2_robot_cfg()` |
| `GO2_ELECTRIC_*` | `ElectricMotorActuatorCfg` | Python BE ODE → ctrl=motor (구버전) | n/a | `get_go2_electric_robot_cfg()` |
| `GO2_NATIVE_ELECTRIC_*` | `NativeElectricActuatorCfg` | filterexact, no coupling | 50 / 0 | `get_go2_native_electric_robot_cfg()` |
| `GO2_COUPLED_ELECTRIC_*` | `NativeElectricActuatorCfg` | `use_coupled=True`, `method="A+"` (default) | 200 / 50 | `get_go2_coupled_electric_robot_cfg()` |
| `GO2_METHODA_*` | `NativeElectricActuatorCfg` | `use_coupled=True`, `method="A"` | 200 / 50 | `get_go2_methoda_robot_cfg()` |
| `GO2_METHODB_*` | `NativeElectricActuatorCfg` | `use_coupled=True`, `method="B"` (GPU 전용) | 200 / 50 | `get_go2_methodb_robot_cfg()` |
| `GO2_APLUS_TLOOP_*` | `NativeElectricActuatorCfg` | A+ + `use_torque_loop=True, Ki=50, integral_max=0.5` | 200 / 50 | `get_go2_aplus_tloop_robot_cfg()` |

상수 별칭:
- `_COUPLED_SUBSTEPS = 200` (= decimation, 0.1 ms × 200 = 20 ms policy dt)
- `_PD_RECOMPUTE = 50` (= 5 ms PD 재계산 주기)

---

## 5. 등록된 task ID 와 env_cfg 함수

`src/tasks/velocity/config/go2/__init__.py` 에서 `register_mjlab_task()` 호출. env_cfg 정의는 `src/tasks/velocity/config/go2/env_cfgs.py`.

| task_id | env_cfg 함수 | 사용 actuator | 시간 위계 |
|---|---|---|---|
| `Unitree-Go2-Rough` | `unitree_go2_rough_env_cfg()` | `GO2_ACTUATOR` (PD) | rough terrain |
| `Unitree-Go2-Flat` | `_go2_flat_pd_cfg()` | `GO2_ACTUATOR` (PD) | dt=5 ms, dec=4, policy 20 ms |
| `Unitree-Go2-Flat-Electric` | `unitree_go2_flat_electric_env_cfg()` | `GO2_ELECTRIC` (Python ODE) | dt=5 ms, dec=4 |
| `Unitree-Go2-Flat-Native-Electric` | `unitree_go2_flat_native_electric_env_cfg()` | `GO2_NATIVE_ELECTRIC` | dt=0.1 ms, dec=50, policy 5 ms |
| `Unitree-Go2-Flat-Coupled-Electric` | `unitree_go2_flat_coupled_electric_env_cfg()` | `GO2_COUPLED_ELECTRIC` (A+) | dt=0.1 ms, dec=200, policy 20 ms |
| `Unitree-Go2-Flat-Coupled-Tloop-Electric` | `unitree_go2_flat_aplus_tloop_electric_env_cfg()` | `GO2_APLUS_TLOOP` (A+ + integral loop) | dt=0.1 ms, dec=200, policy 20 ms |
| `Unitree-Go2-Flat-MethodA-Electric` | `unitree_go2_flat_methoda_electric_env_cfg(use_velocity_action=False)` | `GO2_METHODA` | dt=0.1 ms, dec=200 |
| `Unitree-Go2-Flat-MethodB-Electric` | `unitree_go2_flat_methodb_electric_env_cfg(use_velocity_action=False)` | `GO2_METHODB` (GPU 전용) | dt=0.1 ms, dec=200 |

---

## 6. train / play 사용법

스크립트는 `tyro` 기반. 첫 위치 인자는 task_id, 그 뒤로 nested dataclass 옵션을 `--` flag로 override 가능.

### 6.1 공통 호출 패턴

```bash
# Conda 환경
conda activate mjlab
PY=/home/rbdo/miniconda3/envs/mjlab/bin/python
```

### 6.2 train

`scripts/train.py` (`TrainConfig` dataclass)

| 인자 | 기본값 | 의미 |
|---|---|---|
| `<TASK_ID>` | (필수, 위치인자) | 위 task 표 중 하나 |
| `--env.<...>` | task default | env_cfg dataclass 필드 override (예: `--env.scene.num-envs 4096`) |
| `--agent.<...>` | task default | rl_cfg dataclass 필드 override (예: `--agent.seed 7`, `--agent.max-iterations 3000`) |
| `--motion-file STR` | None | tracking task 전용 motion 파일 경로 |
| `--video / --no-video` | False | 학습 중 비디오 녹화 |
| `--video-length INT` | 200 | 비디오 길이 (step) |
| `--video-interval INT` | 2000 | 녹화 주기 (step) |
| `--enable-nan-guard / --no-enable-nan-guard` | False | NaN 디버그 가드 |
| `--gpu-ids LIST` | [0] | GPU 인덱스 리스트 또는 "all". 다중 GPU = torchrunx 자동 |
| `--torchrunx-log-dir STR` | None | 다중 GPU 로그 디렉터리 |

**예시** — A+ 정책 + 적분 루프 신규 학습 (4096 env, seed 42):

```bash
"$PY" scripts/train.py Unitree-Go2-Flat-Coupled-Tloop-Electric \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 2000 \
  --agent.experiment-name aplus_tloop \
  --agent.run-name seed42
```

**예시** — Method A로 학습 + 학습 중 비디오:

```bash
"$PY" scripts/train.py Unitree-Go2-Flat-MethodA-Electric \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --video --video-interval 2000 --video-length 400
```

**예시** — 다중 GPU:

```bash
"$PY" scripts/train.py Unitree-Go2-Flat-Coupled-Electric \
  --gpu-ids 0 1 2 3 \
  --env.scene.num-envs 8192
```

체크포인트는 `logs/rsl_rl/<experiment_name>/<timestamp>_<run_name>/model_<iter>.pt` 에 저장.

### 6.3 play

`scripts/play.py` (`PlayConfig` dataclass)

| 인자 | 기본값 | 의미 |
|---|---|---|
| `<TASK_ID>` | (필수, 위치인자) | task 표 중 하나 |
| `--agent {trained,zero,random}` | trained | 정책 종류 |
| `--checkpoint-file PATH` | None | 사용할 .pt 체크포인트 파일 (trained 모드) |
| `--num-envs INT` | None | env 수 override (기본 1) |
| `--device STR` | auto | "cuda:0" 또는 "cpu" |
| `--video / --no-video` | False | mp4 녹화 |
| `--video-length INT` | 200 | 비디오 길이 (step) |
| `--video-height INT` | None | 비디오 높이 px |
| `--video-width INT` | None | 비디오 너비 px |
| `--camera (INT\|STR)` | None | 카메라 인덱스/이름 |
| `--viewer {auto,native,viser}` | auto | 뷰어 백엔드 |
| `--no-terminations / --terminations` | False | 종료 조건 비활성화 (에피소드 무한) |
| `--vx FLOAT` | None | 고정 전진 속도 명령 [m/s]. 지정 시 random resample 비활성화 |
| `--vy FLOAT` | None | 고정 좌우 속도 [m/s] |
| `--wz FLOAT` | None | 고정 yaw 각속도 [rad/s] |
| `--num-steps INT` | None | 실행할 step 수. 미지정 시 무한 |
| `--motion-file STR` | None | tracking task용 motion 파일 |

**예시** — A+ 적분 루프 정책 시각화 (속도 0.5 m/s 고정, native viewer):

```bash
"$PY" scripts/play.py Unitree-Go2-Flat-Coupled-Tloop-Electric \
  --checkpoint-file logs/rsl_rl/aplus_dynprm4/2026-04-28_13-27-02_seed42/model_1999.pt \
  --vx 0.5 --num-envs 1
```

**예시** — Method B 정책 + mp4 녹화 (GPU 필수, 1000 step):

```bash
"$PY" scripts/play.py Unitree-Go2-Flat-MethodB-Electric \
  --checkpoint-file logs/rsl_rl/methodB_seed42/2026-04-..._.../model_1999.pt \
  --vx 0.5 --vy 0.0 --wz 0.0 \
  --video --video-length 1000 --video-width 640 --video-height 480 \
  --num-steps 1000
```

**예시** — 더미 정책 (zero/random) 으로 모델만 시각화:

```bash
"$PY" scripts/play.py Unitree-Go2-Flat-Coupled-Electric --agent zero
```

체크포인트 호환성: `Unitree-Go2-Flat-Coupled-Electric` / `Unitree-Go2-Flat-Coupled-Tloop-Electric` 두 task는 관측·액션 공간이 동일하므로 **같은 PPO 체크포인트를 양쪽에서 재사용 가능**. 적분 루프 토글 비교 실험 시 유용.

---

## 7. 새 cfg 변형 만드는 패턴

기존 method 위에 옵션 한 줄만 바꾸고 싶다면:

```python
# 1) src/assets/robots/unitree_go2/go2_constants.py 에 추가
_MY_MOTOR = dict(
    Kt=0.128, Ke=0.128, R=0.3, L=1e-4, gear_ratio=6.33,
    substeps=_COUPLED_SUBSTEPS, pd_substeps=_PD_RECOMPUTE,
    use_coupled=True, method="A+",
    use_torque_loop=True, Ki=80.0, integral_max=1.0,  # 바뀌는 부분만
)
GO2_MY_HIP = NativeElectricActuatorCfg(
    target_names_expr=(".*hip_.*",), stiffness=20.0, damping=1.0,
    effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0,
    armature=0.01, **_MY_MOTOR,
)
# THIGH/CALF도 동일 패턴
GO2_MY_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(GO2_MY_HIP, GO2_MY_THIGH, GO2_MY_CALF),
    soft_joint_pos_limit_factor=0.9,
)

def get_go2_my_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(FULL_COLLISION,),
        spec_fn=get_spec,
        articulation=GO2_MY_ARTICULATION,
    )

# 2) src/tasks/velocity/config/go2/env_cfgs.py 에 env_cfg 함수 추가
def unitree_go2_flat_my_env_cfg(play: bool = False):
    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.scene.entities = {"robot": get_go2_my_robot_cfg()}
    cfg.sim.mujoco.timestep = 0.0001
    cfg.decimation = 200
    return cfg

# 3) src/tasks/velocity/config/go2/__init__.py 에 등록
register_mjlab_task(
    task_id="Unitree-Go2-Flat-My-Electric",
    env_cfg=unitree_go2_flat_my_env_cfg(),
    play_env_cfg=unitree_go2_flat_my_env_cfg(play=True),
    rl_cfg=unitree_go2_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
```

이후 `train.py` / `play.py` 첫 인자에 새 task_id를 그대로 사용.

---

## 8. 핵심 신호 흐름 요약

policy step (20 ms) → action(q_des) → 매 5 ms마다:
1. PD: `τ_des = envelope(Kp·(q_des−q) − Kd·qd)` (effort_limit + DC 토크-속도 envelope)
2. (옵션) 토크 적분 루프: `error = τ_des − τ_actual_prev` → `integral += error·5ms` → `I_des = τ_des/Kt_nom·gr + Ki·integral`
3. 가상 전압 제어기: `V = R·I_des + Ke·gr·ω` → `clamp(±V_bus)`
4. filterexact `ctrl = (V − Ke·gr·ω)/R` (또는 use_callback 모드면 `ctrl = V`)

매 0.1 ms physics step:
5. MuJoCo가 `dI/dt` 적분 (filterexact ZOH; coupled면 Schur cross-Jacobian 추가)
6. force = `gainprm[0] × act = Kt_real·gr × I_actual`

읽기 위치:
- `data.act` ↔ I_actual
- `data.actuator_force` ↔ τ_actual (post-gain, post-clamp; 1 physics step lag)
- `data.qfrc_actuator` ↔ DOF별 토크 (Σactuators)
- 컨트롤러 측 nominal `Kt·gr` ↔ `actuator._Ktgr` (init 시 캐시)
- 플랜트 측 fault `Kt·gr` ↔ `model.actuator_gainprm[idx, 0]` (감자 주입 지점)
- 플랜트 측 fault `Ke·gr` ↔ `model.actuator_dynprm[idx, 1]` (감자 주입 지점)
- 컨트롤러 nominal `Ke·gr` ↔ `model.actuator_dynprm[idx, 3]` (수정 금지)
- method 선택 ↔ `model.actuator_dynprm[idx, 4]` (0/1/2 = A/A+/B)

---

## 9. 빠른 참조 — task별 시간 위계

| task | dt_physics | decimation | dt_policy | pd_substeps | dt_driver |
|---|---|---|---|---|---|
| `Unitree-Go2-Flat` | 5 ms | 4 | 20 ms | n/a | 5 ms (= dt_physics) |
| `Unitree-Go2-Flat-Native-Electric` | 0.1 ms | 50 | 5 ms | 0 (= policy 경계만) | 5 ms (= dt_policy) |
| `Unitree-Go2-Flat-Coupled-Electric` | 0.1 ms | 200 | 20 ms | 50 | 5 ms |
| `Unitree-Go2-Flat-Coupled-Tloop-Electric` | 0.1 ms | 200 | 20 ms | 50 | 5 ms |
| `Unitree-Go2-Flat-MethodA-Electric` | 0.1 ms | 200 | 20 ms | 50 | 5 ms |
| `Unitree-Go2-Flat-MethodB-Electric` (GPU) | 0.1 ms | 200 | 20 ms | 50 | 5 ms |
