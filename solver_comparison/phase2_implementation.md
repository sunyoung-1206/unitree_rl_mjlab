# Phase 2: Cross-Jacobian 삽입 구현 완료

## 수정 요약

MuJoCo 3.6.0 소스에 `mjDYN_FILTEREXACT_COUPLED` dyntype을 추가하여,
Schur complement 방식으로 back-EMF cross-coupling을 implicit solver에 주입했다.

## 수정 파일 (6개, +79/-10 lines)

### 1. `include/mujoco/mjmodel.h` (+2/-1)
- `mjtDyn` enum에 `mjDYN_FILTEREXACT_COUPLED = 6` 추가

### 2. `src/engine/engine_forward.c` (+46)
- `mj_actuate()`: act_dot 계산에서 `mjDYN_FILTEREXACT_COUPLED` → filterexact와 동일 경로
- `mj_advance()`: coupled 액추에이터에 대해:
  - J·qacc sparse dot product로 `omega_new` 계산
  - `ctrl_eff = ctrl - Ke·gr·omega_new / R` 로 back-EMF 반영
  - exact integration으로 act 업데이트
  - actrange clamping 적용

### 3. `src/engine/engine_derivative.c` (+19)
- `mjd_actuator_vel()`: 기존 루프 내에서 coupled 액추에이터에 대해
  - `schur_scale = dt² · Kt·gr · Ke·gr / (L · (1 + dt/τ_e))`
  - `addJTBJSparse(schur_scale)` 호출하여 qDeriv에 Schur complement 항 추가

### 4. `src/engine/engine_support.c` (+2/-1)
- `mj_nextActivation()`: `mjDYN_FILTEREXACT_COUPLED`도 exact integration 경로로 처리

### 5. `src/xml/xml_native_reader.cc` (+8/-7)
- `dyn_map`에 `"filterexact_coupled"` 추가, `dyn_sz` = 7

### 6. `src/user/user_objects.cc` (+2/-1)
- `inheritrange` 체크에 `mjDYN_FILTEREXACT_COUPLED` 추가

## XML 사용법

```xml
<actuator>
  <general name="motor" joint="hinge"
           dyntype="filterexact_coupled"
           dynprm="0.000333 0.81024 0.0001"
           gainprm="0.81024" biasprm="0 0 0"
           actlimited="true" actrange="-29 29"/>
</actuator>
```

### dynprm 슬롯:
| 슬롯 | 파라미터 | Go2 값 |
|------|----------|--------|
| dynprm[0] | τ_e = L/R | 0.000333 s |
| dynprm[1] | Ke·gr | 0.81024 |
| dynprm[2] | L (인덕턴스) | 0.0001 H |

### gainprm 슬롯:
| 슬롯 | 파라미터 | Go2 값 |
|------|----------|--------|
| gainprm[0] | Kt·gr | 0.81024 |

## 빌드 & 테스트 결과

- **빌드**: 100% 성공 (에러 없음)
- **테스트**: 907/907 통과 (기존 동작 영향 없음)
- **소스 위치**: `/home/rbdo/mujoco_src` (branch: `feature/filterexact-coupled`)

## Schur Complement 수학

완전 결합 시스템:
```
[ A   B ] [Δv  ]   [f]
[ C   D ] [Δact] = [g]
```

Schur complement: `(A - B·D⁻¹·C)·Δv = f - B·D⁻¹·g`

추가 항: `B·D⁻¹·C = dt²·Kt·gr·Ke·gr / (L·(1+dt·R/L)) · Jᵀ·J`

이 항이 `qDeriv`에 추가되어 기존 nv×nv sparse solver에서 cross-coupling 효과를 반영.
act 업데이트는 새 qacc 기반으로 omega를 보정한 후 수행.
