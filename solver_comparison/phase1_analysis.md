# Phase 1: MuJoCo Implicit Solver 코드 구조 분석

## MuJoCo 버전: 3.6.0
## 소스 위치: `/home/rbdo/mujoco_src` (tag 3.6.0)

---

## 1. Implicit Integration 전체 흐름

```
mj_step()
  └─ mj_implicitSkip()                         [engine_forward.c:1325]
       ├─ mjd_smooth_vel(m, d, flg_bias)        [engine_derivative.c:1788]
       │    ├─ mjd_actuator_vel(m, d)            [engine_derivative.c:1071]  ← d(qfrc_actuator)/d(qvel)
       │    ├─ mjd_passive_vel(m, d)             [engine_derivative.c:1692]  ← d(qfrc_passive)/d(qvel)
       │    └─ mjd_rne_vel(m, d)                 [engine_derivative.c:596]   ← d(qfrc_bias)/d(qvel)
       │
       ├─ 행렬 조립: qLU = M - dt·qDeriv        [engine_forward.c:1367-1368]
       ├─ LU 분해                                 [engine_forward.c:1391-1396]
       ├─ 선형 시스템 풀이: qacc                   [engine_forward.c:1399-1411]
       └─ mj_advance(m, d, act_dot, qacc, NULL)  [engine_forward.c:1419]
            ├─ act(t+h) = nextActivation(act_dot) [engine_forward.c:884-892]
            ├─ qvel += dt·qacc                     [engine_forward.c:904-910]
            └─ qpos = integratePos(qvel, dt)       [engine_forward.c:915]
```

**핵심 관찰: MuJoCo는 기계 DOF(qvel)에 대해서만 implicit solve를 수행하고, activation은 별도로 적분하는 staggered scheme을 사용한다.**

---

## 2. 현재 Implicit 행렬 구조

### 풀리는 시스템 (nv × nv):
```
(M - dt · qDeriv) · qacc = qfrc_smooth + qfrc_constraint
```

여기서 `qDeriv = d(qfrc_smooth)/d(qvel)` 이며:
- `qfrc_smooth = qfrc_actuator + qfrc_passive - qfrc_bias`

### 프롬프트 문서에서 기대하는 **완전 결합 시스템** (nv+na × nv+na):
```
[ M - dt·(∂f/∂v)    |  -dt·(∂f/∂act)    ] [Δv  ]   [rhs_mech]
[                    |                    ] [    ] = [         ]
[ -dt·(∂ȧ/∂v)       |  I - dt·(∂ȧ/∂act) ] [Δact]   [rhs_act ]
```

### 현재 MuJoCo 상태:
| 블록 | 수식 | 현재 상태 |
|------|------|-----------|
| 좌상 (nv×nv) | `M - dt·∂f/∂v` | ✅ 반영 (`qDeriv` 포함) |
| 우상 (nv×na) | `-dt·∂f/∂act` = `-dt·Kt·gr·J^T` | ⚠️ **간접 반영** (actearly로 act를 미리 갱신) |
| 좌하 (na×nv) | `-dt·∂ȧ/∂v` = `-dt·(-Ke·gr/L)·J` | ❌ **완전 누락** (0으로 가정) |
| 우하 (na×na) | `I - dt·∂ȧ/∂act` = `I + dt·R/L` | ❌ **없음** (activation은 explicit 적분) |

---

## 3. 핵심 코드 위치 상세

### 3.1 act_dot 계산 (`mj_actuate`)
- **파일**: `engine_forward.c:342-351`
- filterexact의 act_dot: `(ctrl - act) / tau`
- `tau = dynprm[0]` (= L/R for motor)
- **qvel 의존성 없음** — 이것이 cross-Jacobian이 빠진 근본 원인

### 3.2 filterexact Exact Integration (`mj_nextActivation`)
- **파일**: `engine_support.c:708-732`
- `act(h) = act(0) + act_dot(0) · τ · (1 - exp(-h/τ))`
- `mj_advance()`에서 호출 (engine_forward.c:890)

### 3.3 ∂(qfrc_actuator)/∂(qvel) (`mjd_actuator_vel`)
- **파일**: `engine_derivative.c:1071-1147`
- gain의 velocity component (`gainprm[2]`)와 bias의 velocity component (`biasprm[2]`)를 사용
- `addJTBJSparse()` (engine_derivative.c:746)로 `J^T · B · J` 형태로 qDeriv에 추가
- **act에 대한 derivative는 계산하지 않음**

### 3.4 ∂(act_dot)/∂(qvel) — **존재하지 않는 코드**
- 현재 `act_dot = (ctrl - act) / tau`로 계산되므로 qvel 의존성이 없다고 가정
- DC 모터의 실제 ODE: `dI/dt = (V - R·I - Ke·gr·ω) / L`
  - 여기서 `ω = J · qvel` (moment arm 통한 매핑)
  - 따라서 `∂(act_dot)/∂(qvel) = -Ke·gr·J / L` ≠ 0
- **이 항이 implicit 행렬에 들어가야 할 위치**: 좌하 블록 (na×nv)

### 3.5 dynprm / gainprm / biasprm 읽기
- **dynprm**: `engine_forward.c:335` — `m->actuator_dynprm + i*mjNDYN` (mjNDYN=10)
- **gainprm**: `engine_forward.c:411` — `m->actuator_gainprm + i*mjNGAIN` (mjNGAIN=10)
- **biasprm**: `engine_forward.c:456` — `m->actuator_biasprm + i*mjNBIAS` (mjNBIAS=10)
- 각각 10개 슬롯, 대부분 비어있어 새 파라미터 저장 가능

---

## 4. 데이터 구조 매핑

### mjModel 주요 필드 (include/mujoco/mjmodel.h)
| 필드 | 크기 | 설명 |
|------|------|------|
| `nv` | scalar | 자유도 수 (velocity DOF) |
| `nu` | scalar | 액추에이터 수 |
| `na` | scalar | activation 상태 수 |
| `actuator_dyntype` | nu×1 | 동역학 타입 (mjtDyn enum) |
| `actuator_dynprm` | nu×10 | 동역학 파라미터 (slot 0 = τ) |
| `actuator_gainprm` | nu×10 | gain 파라미터 (slot 0 = Kt·gr for FIXED) |
| `actuator_biasprm` | nu×10 | bias 파라미터 |
| `actuator_actadr` | nu×1 | act 배열 내 시작 주소 |
| `actuator_actnum` | nu×1 | activation 변수 개수 |
| `actuator_trnid` | nu×2 | 전달 대상 (joint ID 등) |
| `actuator_gear` | nu×6 | gear ratio (slot 0 = gr) |

### mjData 주요 필드 (include/mujoco/mjdata.h)
| 필드 | 크기 | 설명 |
|------|------|------|
| `qvel` | nv×1 | 일반화 속도 |
| `act` | na×1 | activation 상태 (= 전류 I) |
| `act_dot` | na×1 | activation 시간미분 |
| `qDeriv` | nD×1 | sparse Jacobian: d(qfrc_smooth)/d(qvel) |
| `qLU` | nD×1 | implicit 행렬의 LU 분해 |
| `actuator_moment` | sparse | moment arm J (actuator→DOF 매핑) |
| `actuator_velocity` | nu×1 | 액추에이터 속도 (= J·qvel) |
| `qfrc_actuator` | nv×1 | 액추에이터 힘 (joint space) |

### mjtDyn enum (mjmodel.h:241-248)
```c
mjDYN_NONE          = 0,  // ctrl → force 직접
mjDYN_INTEGRATOR    = 1,  // da/dt = u
mjDYN_FILTER        = 2,  // da/dt = (u-a)/τ, Euler 적분
mjDYN_FILTEREXACT   = 3,  // da/dt = (u-a)/τ, exact 적분
mjDYN_MUSCLE        = 4,  // 근육 모델
mjDYN_USER          = 5,  // 사용자 정의
```

---

## 5. 수정 전략 (Phase 2를 위한 예비 분석)

### 접근법 A: 완전 결합 시스템 (nv+na × nv+na)
- implicit 행렬을 확장하여 activation도 implicit으로 풀기
- **장점**: 물리적으로 완전, 큰 dt에서 안정
- **단점**: 행렬 크기 변경 → 기존 sparse structure와 solver 대폭 수정 필요

### 접근법 B: 기존 nv×nv 행렬에 Schur complement 반영
- 좌하/우하 블록을 Schur complement로 축약하여 nv×nv 행렬에 포함
- 수정된 행렬: `(M - dt·∂f/∂v) - dt²·(∂f/∂act)·(I - dt·∂ȧ/∂act)⁻¹·(∂ȧ/∂v)`
- **장점**: 기존 solver 구조 유지, 최소 수정
- **단점**: Schur complement 계산 추가

### 접근법 C: act_dot에 qvel 의존성 추가 + qDeriv 확장
- `act_dot = (ctrl - act - Ke·gr·ω·τ/...) / τ` 형태로 act_dot 수정
- qDeriv에 cross-term 반영
- **장점**: 기존 staggered scheme 내에서 수정 가능
- **단점**: 완전한 implicit coupling은 아님

**→ 접근법 B (Schur complement)가 가장 실용적. 기존 nv×nv sparse solver를 그대로 사용하면서 cross-Jacobian 효과를 반영할 수 있음.**

---

## 6. 수정할 파일 목록 (예상)

| 파일 | 수정 내용 |
|------|-----------|
| `include/mujoco/mjmodel.h` | 새 dyntype enum 값 또는 flag 추가 |
| `src/engine/engine_forward.c` | act_dot 계산에 qvel 항 추가, advance 로직 수정 |
| `src/engine/engine_derivative.c` | qDeriv에 cross-Jacobian term 추가 |
| `src/xml/xml_native_reader.c` | 새 XML attribute 파싱 (필요시) |
