# Phase 1 — mjwarp filterexact 에 Ke coupling 추가 (수식 유도)

목적: mjwarp `forward.py`의 FILTEREXACT 분기를 "Ke-coupled filterexact" 로 확장하여,
컨트롤러가 nominal Ke 로 back-EMF FF 를 수행하고 플랜트가 실제 Ke_plant 를 가질 때
정확한 전류 응답이 나오도록 한다. 코드 변경 전, 수식만으로 설계 확정이 목적.

---

## 1. 현재 mjwarp FILTEREXACT 분기 (원본)

### 1a. `_actuator_force` kernel — act_dot 계산 (forward.py:671-673)

```python
elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:
    act = act_in[worldid, act_last]
    act_dot = (ctrl - act) / wp.max(dynprm[0], MJ_MINVAL)
```

### 1b. `_next_act` function — exact integration (forward.py:147-149)

```python
if actuator_dyntype == DynType.FILTEREXACT:
    tau = wp.max(MJ_MINVAL, actuator_dynprm[0])
    act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))
```

합쳐서 현재 적분하는 ODE 는:

```
dI/dt = (ctrl - I) / τ,  τ = dynprm[0] = L/R
```

`dynprm[1..3]` 는 완전히 무시됨.

---

## 2. 풀어야 할 진짜 ODE (plant 물리)

모터 전기적 방정식 (모터 단자 기준):

```
L · dI/dt = V_applied - R · I - Ke_plant · ω_motor                         (1)
```

- `I` : 모터 코일 전류 [A]
- `L` : 모터 인덕턴스 [H]
- `R` : 모터 권선 저항 [Ω]
- `Ke_plant` : 실제 플랜트의 back-EMF 상수 [V·s/rad_motor]
  - demag 고장 시 `Ke_plant = factor · Ke_nom`
- `ω_motor` : 모터 샤프트 각속도 [rad/s]
  - `ω_motor = gr · ω_joint`  (gr = 기어비, 모터 샤프트가 관절 1회전당 gr 회전)
- `V_applied` : 모터 단자에 인가되는 실제 전압 [V]

관절 공간으로 재표현 (mjwarp 는 `qvel` = ω_joint 보유):

```
L · dI/dt = V_applied - R · I - (Ke_plant · gr) · ω_joint                  (2)
```

`Ke·gr` 는 "joint-space 등가 back-EMF 상수" [V·s/rad_joint]. 이후 편의상
`Ke_p := Ke_plant · gr`, `Ke_n := Ke_nom · gr` 로 표기.

---

## 3. 컨트롤러 의 ctrl 의미 (변경하지 않음)

`mj_native_electric_actuator.py` 의 compute() 는 nominal Ke 로 back-EMF FF 를
수행한 뒤 "전류 형태 ctrl"을 출력:

```
V_cmd  = R · I_des + Ke_n · ω_joint              (nominal Ke 로 보상)
V_sat  = clamp(V_cmd, ±V_bus)                    (bus 전압 clamp)
ctrl   = (V_sat − Ke_n · ω_joint) / R            (전류 형태 재변환)
```

주요 관계식 (clamp 여부 무관, 항상 성립):

```
V_applied = R · ctrl + Ke_n · ω_joint            (3)
```

(비포화 시 V_applied = V_cmd, 포화 시 V_applied = V_sat. 어느 쪽이든 식 (3) 로부터
ctrl 을 구했기 때문에 역방향도 자동 성립.)

---

## 4. 올바른 kernel ODE 유도

식 (3) 을 식 (2) 에 대입:

```
L · dI/dt = (R · ctrl + Ke_n · ω) − R · I − Ke_p · ω
         = R · (ctrl − I) + (Ke_n − Ke_p) · ω                              (4)
```

양변을 L = R·τ 로 나누고 τ = dynprm[0] 사용:

```
dI/dt = (ctrl − I) / τ  +  (Ke_n − Ke_p) · ω / L                           (5)
```

### 건전성 check

- 건강한 모터 (Ke_p = Ke_n): 두 번째 항 = 0 → vanilla filterexact 로 환원 ✓
- Demag (Ke_p < Ke_n, factor<1): 두 번째 항 양수(ω>0 일 때) → I_ss 가 I_des 보다
  증가 → plant 가 back-EMF 로부터 "덜 저항"받음 → 전류가 더 흐름 ✓
- 부호: ω<0 이면 두 번째 항도 음수 → I_ss 감소. 물리적으로 자연스러움.

---

## 5. Analytic solution (filterexact 형태)

ω 가 한 스텝 동안 근사적으로 상수라고 두면 (dt=0.1ms, τ=0.33ms 에서 타당):

```
dI/dt + I/τ = ctrl/τ + β·ω,   β = (Ke_n − Ke_p) / L
```

1차 선형 ODE. 특이점 제외 해:

```
I(t) = I_ss  +  (I_0 − I_ss) · exp(−t/τ)
I_ss = τ · (ctrl/τ + β·ω)  =  ctrl + β·ω·τ  =  ctrl + (Ke_n−Ke_p)·ω·τ/L
                                              = ctrl + (Ke_n−Ke_p)·ω/R   (τ=L/R)
```

dt 한 스텝 업데이트:

```
I(t+dt) = I + (I_ss − I) · (1 − exp(−dt/τ))                                (6)
        = I + ( (ctrl − I) + (Ke_n − Ke_p)·ω/R ) · (1 − exp(−dt/τ))        (6')
```

이는 **기존 filterexact 적분을 I_ss 로 바꾼 것과 동일** — 코드 변경 최소.
_next_act 에서 exact 적분 공식은 그대로 두고, act_dot 만 바꾸면 됨.

### `_next_act` 공식과의 호환 확인

기존: `act = act_in + act_dot · τ · (1 − exp(−dt/τ))` 에서 `act_dot = (ctrl − act)/τ`

새 act_dot:
```
act_dot_new = (I_ss − act) / τ
            = (ctrl − act) / τ  +  (Ke_n − Ke_p) · ω / L                   (7)
```

그러면:
```
act + act_dot_new · τ · (1 − exp(−dt/τ))
 = act + (I_ss − act) · (1 − exp(−dt/τ))
```

식 (6) 과 완전히 일치 ✓. **`_next_act` 함수는 수정 불필요**.
`_actuator_force` kernel 의 act_dot 계산만 수정.

---

## 6. dynprm 슬롯 매핑

현재 `mj_native_electric_actuator.py` (use_coupled=True 분기, line 273-283) 은 다음 슬롯 세팅:

| slot | 값 (build 시) | 값 (demag 후 run_demag_experiment.py 가 수정) |
|---|---|---|
| `dynprm[0]` | `τ_e = L/R` | 불변 |
| `dynprm[1]` | `Ke_nom · gr` | **`Ke_plant · gr`** (demag 스크립트가 덮어씀) |
| `dynprm[2]` | `L` | 불변 |
| `dynprm[3]` | `1.0` or `0.0` (Schur flag) | 불변 |

**문제**: 현재 scheme 은 `dynprm[1]` 에 Ke_plant 를 담도록 되어 있어서, 커널이
`Ke_nom` 을 따로 얻을 방법이 없음. 식 (7) 의 `(Ke_n − Ke_p)` 계산 불가.

### 제안: 슬롯 재할당

| slot | 새 의미 | build 시 값 | demag 후 |
|---|---|---|---|
| `dynprm[0]` | `τ_e` | `L/R` | 불변 |
| `dynprm[1]` | **`Ke_plant · gr`** (실제 plant) | `Ke_nom · gr` (healthy 시 동일) | `Ke_plant · gr` ← demag 스크립트 수정 |
| `dynprm[2]` | `L` | `L` | 불변 |
| `dynprm[3]` | **`Ke_nom · gr`** (controller nominal) | `Ke_nom · gr` | **불변** — 절대 수정 금지 |

이러면 kernel 은 `(dynprm[3] − dynprm[1]) · ω / dynprm[2]` 로 correction term 계산 가능.

### 호환성
- Healthy 케이스: `dynprm[3] == dynprm[1]` → correction = 0 → vanilla 와 동일한 거동.
- 기존 `use_filterexact_schur` Schur flag 는 사용 중단 (mjwarp 가 어차피 무시하던 슬롯).

### 변경 범위
- `mj_native_electric_actuator.py:280` 수정: `dynprm[3] = 1.0 if schur else 0.0` → `dynprm[3] = cfg.Ke * cfg.gear_ratio` (Ke_nom·gr, 항상).
- `run_demag_experiment.py:87, 92` 는 그대로 (dynprm[1] 만 수정, dynprm[3] 건드리지 않음).
- mjwarp `forward.py:671-673` 수정 (다음 Phase).

---

## 7. 제안 kernel 패치 (코드는 Phase 2 에서, 여기선 의사코드)

```python
elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:
    act = act_in[worldid, act_last]
    tau_e = wp.max(dynprm[0], MJ_MINVAL)
    # Vanilla filterexact term
    act_dot = (ctrl - act) / tau_e
    # Ke coupling (no-op when dynprm[3] == dynprm[1])
    Ke_plant_gr = dynprm[1]
    L = dynprm[2]
    Ke_nom_gr   = dynprm[3]
    if L > MJ_MINVAL:
        omega = actuator_velocity_in[worldid, uid]   # 이미 kernel 입력에 있음
        act_dot += (Ke_nom_gr - Ke_plant_gr) * omega / L
```

- `actuator_velocity_in` 은 `_actuator_force` kernel 입력에 이미 존재 (line 645).
  이것은 `Σ_dof moment[uid, dof] · qvel[dof]` 로, 1:1 joint-transmission 인 경우
  `qvel` at that joint. Go2 hip/thigh/calf 모두 1:1 transmission 이므로 등가.
- `_next_act` 함수는 무변경.
- 다른 dyntype (INTEGRATOR/MUSCLE/USER/NONE) 분기는 그대로 → **다른 actuator 타입
  영향 없음** (규칙 검증 Phase 2 에서 대조 실험).

---

## 8. 부호·단위 최종 검증

| 식 | 차원 check | 값 예시 (factor=0.6, ω=10 rad/s) |
|---|---|---|
| `(Ke_n − Ke_p)` | V·s/rad | `(1−0.6)·0.128·6.33 = 0.324 V·s/rad_joint` |
| `L` | H = V·s/A | `1e-4` |
| `(Ke_n − Ke_p)·ω / L` | V·rad·1/(V·s/A·rad·1/s·s) = A/s | `0.324·10/1e-4 = 3.24e4 A/s` |

`A/s` = act_dot 단위 ✓ (act = 전류 [A], act_dot = 전류변화율 [A/s])

**Steady-state 증가량** (hand check):
```
ΔI_ss = (Ke_n − Ke_p)·ω/R = 0.324·10/0.3 = 10.8 A
```

factor=0.6 에서 ω=10 rad/s 일 때 I_actual 이 I_des 보다 **10.8 A 만큼 증가**.
이는 이전에 계산한 Δ_I slope = 1.08 A·s/rad 와 일치 ✓.

**ratio 예시** (I_des ≈ 5A 상황):
```
ratio = factor · (1 + ΔI_ss/I_des) = 0.6 · (1 + 10.8/5) = 0.6 · 3.16 = 1.90
```

현재 broken 케이스의 ratio mean ≈ 0.6 → 수정 후 1.0~1.5 수준으로 상승 예상
(정확한 값은 gait 평균 ω/I_des 에 의존).

---

## 9a. 검증 기준표 (수정됨 — 2026-04-21 user review 반영)

**이전 기준의 정정**: "ratio < factor, slope 음수" 는 **적응제어 가정 하에서만** 맞음.
현재 시나리오는 컨트롤러가 결함 모르고 `Ke_nom` 으로 V 과보상 → I 폭증 → ratio 증가.

| 케이스 | 기대 ratio (= τ_actual/τ_cmd) | 기대 Δ_I/ω slope |
|---|---|---|
| healthy (factor=1.0) | ≈ 1.0 (ω 무관) | ≈ 0 |
| demag factor=0.6 | 1.0~2.5 (> factor, ω 따라 증가) | **+1.08** A·s/rad |
| demag factor=0.4 | > factor, 더 큼 | **+1.62** A·s/rad |

판정:
- ✅ healthy 에서 ratio≈1, slope≈0 → 부호 sanity 통과 (moment matrix 부호 OK).
- ✅ demag 에서 ratio > factor, slope 양수, 크기 ±20% 일치 → 수식 반영 성공.
- ❌ healthy ratio ≠ 1 → moment matrix 부호 문제 (§9 가정 4).
- ❌ demag slope **음수** → §4 부호 실수 (Ke_n, Ke_p 순서 뒤바뀜 의심).
- ❌ demag slope 양수지만 크기 안 맞음 → motor vs joint ω 혼동 (gear ratio 위치).

## 9b. 추가 안전 장치 (user review 보완 1~5)

### 보완 1 — dynprm[3] 침해 방지 가드
Build 직후 `dynprm[:, 3]` snapshot 저장. 매 rollout 시작 직전 assertion:
```python
assert np.allclose(mj_model.actuator_dynprm[:, 3], ke_nom_gr_snapshot), \
    "dynprm[3] (Ke_nom·gr) was modified — silent failure risk"
```
→ `run_demag_experiment.py` 에 guard 추가 (Phase 2 코드 작업 시).

### 보완 2 — I 폭증 stability 모니터링
`ΔI_ss = 10.8 A @ factor=0.6, ω=10 rad/s` 는 Go2 모터 current limit 근접 가능.
- Phase 2·3 에서 `|I_actual|` max/mean 로깅.
- Go2 motor spec (effort_limit/Kt ≈ 45/0.8 = 56 A 수준) 과 비교 출력.
- mjwarp FILTEREXACT 분기에 `actuator_actlimited + actrange` clip 있는지 확인
  → `_next_act` line 155-157 에서 `if clamp: act = wp.clamp(act, actrange[0], actrange[1])` 확인됨. OK.

### 보완 3 — ω 시점 명시
의사코드에 주석 추가:
```python
omega = actuator_velocity_in[worldid, uid]  # ω at step start; ZOH over dt
```
(실제로 actuator_velocity_in 은 `_actuator_velocity` kernel 이 매 dt 초마다 갱신하는
"current qvel" 이므로 엄밀히는 ZOH 가 아니라 "step 시작 시점 값" — 여전히 ω dt
동안 상수 근사).

### 보완 4 — FILTER vs FILTEREXACT 분기 격리
현재 mjwarp 원본은 둘이 같은 elif 묶음 (forward.py:671):
```python
elif dyntype == DynType.FILTER or dyntype == DynType.FILTEREXACT:
    ...
```
→ Phase 2 에서 분리:
```python
elif dyntype == DynType.FILTER:                     # 변경 없음
    act = act_in[worldid, act_last]
    act_dot = (ctrl - act) / wp.max(dynprm[0], MJ_MINVAL)
elif dyntype == DynType.FILTEREXACT:                 # Ke coupling 추가
    act = act_in[worldid, act_last]
    tau_e = wp.max(dynprm[0], MJ_MINVAL)
    act_dot = (ctrl - act) / tau_e
    L_dyn = dynprm[2]
    if L_dyn > MJ_MINVAL:
        omega = actuator_velocity_in[worldid, uid]
        act_dot += (dynprm[3] - dynprm[1]) * omega / L_dyn
```
FILTER 는 건드리지 않음 → 기존 MJ actuator 전원 무영향.

### 보완 5 — `_next_act` sanity test (Phase 2 unit test)
현재 `_next_act` 는 `act = act_in + act_dot·τ·(1−exp(−dt/τ))` 형태. 이것이
실제로 analytic 1차 응답과 일치하는지 독립 확인:
```python
# constant act_dot 케이스에서 직접 적분과 비교
# _next_act 호출 결과 vs I_ss + (I_0 − I_ss)·exp(−dt/τ) 여기서 I_ss = I_0 + act_dot·τ
```
만약 불일치 → §5 가정이 틀린 것 → 즉시 Phase 1 복귀.

---

## 9. 확인이 필요한 가정 목록 (Phase 2 검증 전에 결론)

1. **`actuator_velocity_in[worldid, uid]` 가 정말 joint qvel 과 같은가?**  
   → Go2 MethodA actuator 들은 모두 JOINT transmission + unit moment 이므로 동일하다고 가정.
   1:N multi-joint transmission 이 있다면 식 (5) 자체가 달라짐. Phase 2 kernel 수정 전 1회 검증.

2. **`_next_act` 의 exact 공식이 정말 act_dot 기반인가?**  
   `act = act_in + act_dot · τ · (1 − exp(−dt/τ))` 이 맞는지 재확인:
   - act_dot = (I_ss − act)/τ 로 치환 시 `act + (I_ss − act)·(1 − exp(−dt/τ))` 가 맞음 ✓
   - Phase 2 에서 실수가 있는지 합성 코드로 재검증.

3. **`dynprm[3]` 을 새로 점유해도 되는가?**  
   - mjwarp: 현재 무시 중 → 무해.
   - standard mujoco CPU 경로: filterexact 는 dynprm[0] 만 씀 → 무해.
   - 포크 `.coupled` 바이너리: `dynprm[3]` 을 flag 로 해석했으나 우리가 쓰는 active
     바이너리는 vanilla 이므로 무관.

4. **부호 관례**: mjwarp `_actuator_velocity` 부호 검증. moment matrix 부호 관례가
   우리가 생각하는 "ω_joint = qvel[dof]" 와 일치하는지. Phase 2 에서 healthy 케이스
   실행 후 ratio≈1, slope≈0 확인으로 간접 검증 (추가 항의 부호가 틀리면 healthy
   에서도 ratio ≠ 1 이 나옴).

---

## 10. 요약

- **수식**: `act_dot = (ctrl − act)/τ + (Ke_nom·gr − Ke_plant·gr) · ω_joint / L`
- **적분 공식**: 현재 `_next_act` 그대로 사용 가능 (구조 동일).
- **슬롯**: `dynprm[1]=Ke_plant·gr`, `dynprm[3]=Ke_nom·gr` 로 재할당. 기타 불변.
- **호환성**: healthy 시 correction=0 → vanilla 거동. 다른 dyntype 무영향.
- **건드릴 파일 (Phase 2)**: `mujoco_warp/_src/forward.py` (kernel), 
  `mj_native_electric_actuator.py` (dynprm[3] 세팅).
- **검증 지표** (Phase 3): factor=0.6 에서 Δ_I vs ω slope ≈ +1.08 A·s/rad, 
  healthy 에서 slope ≈ 0.
