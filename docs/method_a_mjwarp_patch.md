# Method A Coupling on mjwarp (GPU) — Patch Documentation

**Date:** 2026-04-27
**Backend:** `mujoco_warp` (GPU)
**Scope:** Implements pure Method A back-EMF coupling for FILTEREXACT motor actuators.

이 문서는 mjwarp에 적용한 3가지 패치의 작동 원리와 실제 코드 위치를 정리합니다.

---

## 0. 표기 및 약속

| 기호 | 의미 |
|------|------|
| `I` | 모터 전류 (= `act` slot) |
| `ω` | 관절 각속도 (`qvel`) |
| `τ` | 전기 시정수 = L/R |
| `h` | physics timestep |
| `β` | 1-step decay factor |
| `Kt·gr` | 토크 상수 × 기어비 (= `gainprm[0]`) |
| `Ke·gr` | 역기전력 상수 × 기어비 |
| `J` | 액추에이터 모멘트 행렬 (`actuator_moment`) |

**dynprm 슬롯 (FILTEREXACT motor):**
| 슬롯 | 의미 |
|------|------|
| `dynprm[0]` | τ_e = L/R |
| `dynprm[1]` | **Ke_plant·gr** — 실제 plant의 Ke (demag로 변경 가능) |
| `dynprm[2]` | L (인덕턴스) |
| `dynprm[3]` | **Ke_nom·gr** — 명목 Ke (controller가 가정; 변경 안 됨) |

healthy 상태: `dynprm[1] == dynprm[3]`. demag 발생: `dynprm[1] < dynprm[3]`.

---

## 1. Method A 적분기 — `β = 1/(1+h/τ)`

### 수식
1차 ODE `dI/dt = (ctrl − I)/τ`를 implicit Euler로 이산화:
$$
I_{n+1} = I_n + h \cdot \frac{\mathrm{ctrl} - I_{n+1}}{\tau}
\quad\Longrightarrow\quad
I_{n+1} = \beta\,I_n + (1-\beta)\,\mathrm{ctrl}, \quad
\beta = \frac{1}{1+h/\tau}
$$

vanilla MuJoCo의 filterexact는 β = exp(−h/τ) (closed-form exponential)를 쓰고, Method A는 implicit Euler β를 씁니다. dt가 작으면 두 β의 차이는 작지만, implicit damping의 일관성을 위해서는 Schur와 같은 β를 쓰는 게 자연스러움.

### 패치 위치
**`forward.py:147-160`** (function `_next_act`):
```python
if actuator_dyntype == DynType.FILTEREXACT:
    tau = wp.max(MJ_MINVAL, actuator_dynprm[0])
    # Motor coupling detection: dynprm[1]=Ke*gr != 0 AND dynprm[2]=L > 0
    # → Pure Method A integrator: β = 1/(1+h/τ)
    # Otherwise → vanilla filterexact: β = exp(-h/τ)
    if actuator_dynprm[1] != 0.0 and actuator_dynprm[2] > MJ_MINVAL:
        # Method A: act = act + h * act_dot / (1 + h/τ)
        act = act_in + act_dot_scale * act_dot_in * opt_timestep / (1.0 + opt_timestep / tau)
    else:
        act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))
```

**작동 조건:** `dynprm[1] ≠ 0 && dynprm[2] > 0` (motor coupling 활성).
**fallback:** 위 조건 불충족 시 vanilla filterexact 그대로 (다른 FILTEREXACT actuator에 영향 없음).

---

## 2. Schur Cross-Jacobian — implicit solver의 좌변

### 유도 (요약)
전기-역학 coupled 1-step 풀이에서 ω의 Δq̈ 의존성을 풀면:
$$
[M + (1-\beta)\,K_t g_r\,K_e g_r\,\frac{h}{R}\,J^\top J]\,\ddot q = (\text{constants})
$$

MuJoCo implicit 컨벤션 `M_\text{eff} = qM - h\cdot\mathrm{qDeriv}`와 비교:
$$
\mathrm{qDeriv} \mathrel{+}= -(1-\beta)\,\frac{K_t g_r\,K_e g_r}{R}\,J^\top J
$$

→ B 스칼라로 표현: `B = -(1-β)·Kt·gr·Ke·gr/R` (음수, implicit damping).

부호 확인: `M_eff = qM − h·qDeriv`의 음수 qDeriv → `+h·|qDeriv|` 가 M에 더해져 양정정 강화. 안정.

### 패치 위치
**`derivative.py:32-93`** (function `_qderiv_actuator_passive_vel`):
```python
# Method A Schur: -(1-β)·Kt·gr·Ke·gr/R for FILTEREXACT motor coupling
# β = 1/(1+h/τ) → 1-β = h/(τ+h)
schur = float(0.0)
if actuator_dyntype[actid] == DynType.FILTEREXACT:
    dynprm_act = actuator_dynprm[actuator_dynprm_id, actid]
    Ke_gr = dynprm_act[1]
    L_val = dynprm_act[2]
    if Ke_gr != 0.0 and L_val > MJ_MINVAL:
        tau_e = wp.max(MJ_MINVAL, dynprm_act[0])
        Kt_gr = actuator_gainprm[actuator_gainprm_id, actid][0]
        h_dt = opt_timestep[worldid % opt_timestep.shape[0]]
        one_minus_beta = h_dt / (tau_e + h_dt)
        R_val = L_val / tau_e
        schur = -one_minus_beta * Kt_gr * Ke_gr / R_val
...
vel = float(bias) + schur
```

여기서 `vel`은 이후 `_qderiv_actuator_passive_actuation_sparse` 커널 (`derivative.py:97-150`)에서 다음 식으로 누적됩니다:
```
qderiv_contrib += moment_i * moment_j * vel
```
즉, `qDeriv += JᵀBJ` 형태로, B = vel = bias + schur.

### Caller 패치
**`derivative.py:248-269`** (function `deriv_smooth_vel`):
`_qderiv_actuator_passive_vel`에 `actuator_dynprm`과 `opt_timestep`을 새로 전달.

### 중요한 부호/스케일 메모
이전 patch (mujoco_src commit `cf35fd22`, 2026-04-10)에서는 `+h²·...` 스케일 + 부호도 반대로 잘못 들어가 있었음. 오늘 commit `6a7115de`에서 부호 및 dt 차수 모두 교정됨. mjwarp 패치는 그 교정된 식을 그대로 따름.

---

## 3. RHS 보정 — Force가 I_predicted를 사용하도록

### 수식
vanilla mjwarp의 `_actuator_force` 커널은 `force = gain·ctrl_act + bias`를 계산하는데, 여기서 `ctrl_act = act_in[act_last] = I_n` (현재 전류). Method A에서는 force가 **다음 step의 예측 전류** I_{n+1}을 사용해야 합니다:

$$
F = K_t g_r \cdot I_{n+1} = K_t g_r \cdot \big[\beta I_n + (1-\beta)\,\mathrm{ctrl}\big]
$$
$$
\Delta F = K_t g_r \cdot (1-\beta)\,(\mathrm{ctrl} - I_n)
$$

이 ΔF를 force에 더해주는 것이 RHS 보정.

### 패치 위치
**`forward.py:752-770`** (kernel `_actuator_force`):
```python
force = gain * ctrl_act + bias

# Method A RHS correction (Schur complement RHS):
# force currently uses I_old (act_in[act_last]) when not actearly.
# Method A predicts I_new = β·I_old + (1-β)·ctrl  with  β = 1/(1+h/τ).
# ΔF = gain·(1-β)·(ctrl - I_old).
# Active only for FILTEREXACT motor actuators (dynprm[1]≠0 AND dynprm[2]>0).
if na and act_first >= 0:
    if actuator_dyntype[uid] == DynType.FILTEREXACT:
        dynprm_uid = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]
        if dynprm_uid[1] != 0.0 and dynprm_uid[2] > MJ_MINVAL and not actuator_actearly[uid]:
            tau_e = wp.max(MJ_MINVAL, dynprm_uid[0])
            h_dt = opt_timestep[worldid % opt_timestep.shape[0]]
            one_minus_beta = h_dt / (tau_e + h_dt)  # Method A: 1 - 1/(1+h/τ)
            I_old = act_in[worldid, act_last]
            force += gain * one_minus_beta * (ctrl - I_old)
```

**작동 조건:**
- FILTEREXACT motor (`dynprm[1] ≠ 0 && dynprm[2] > 0`)
- `actearly == False` (actearly=True면 force가 이미 `_next_act` 결과를 쓰므로 중복 보정 안 됨)

**왜 `ctrl`인가?** 이 시점의 `ctrl`은 위쪽에서 이미 clamp된 값. 사용자의 Python에서 `ctrl = (V − Ke_nom·gr·ω_old)/R` 형태로 들어와 있어, 이게 "current 단위의 steady-state" 의미. β-blend로 I_predicted를 만들어 force에 반영.

---

## 4. Demag 보정 — `act_dot += (Ke_nom − Ke_plant)·gr·ω/L`

### 수식 배경
Plant의 진짜 ODE:
$$
L\,\dot I = V - R\,I - K_e^{\text{plant}}\,g_r\,\omega
$$

Controller는 nominal Ke 가정하여 V를 계산:
$$
V = R\,I_\text{des} + K_e^{\text{nom}}\,g_r\,\omega
$$

대입하면:
$$
\dot I = \frac{I_\text{des} - I}{\tau} + \frac{(K_e^{\text{nom}} - K_e^{\text{plant}})\,g_r\,\omega}{L}
$$

오른쪽 두 번째 항이 controller-plant Ke mismatch에서 오는 추가 보정. healthy 상태(Ke_nom = Ke_plant)에서 0.

### 패치 위치
**`forward.py:679-693`** (kernel `_actuator_force`, FILTEREXACT branch — 이미 4/21에 적용된 기존 패치):
```python
elif dyntype == DynType.FILTEREXACT:
    # Coupled filterexact: standard filter + Ke mismatch correction.
    #   dI/dt = (ctrl - act) / tau  +  (Ke_nom*gr - Ke_plant*gr) * omega / L
    # Slots: dynprm[0]=tau=L/R, [1]=Ke_plant*gr, [2]=L, [3]=Ke_nom*gr.
    # Healthy: dynprm[3]==dynprm[1] -> correction=0 -> vanilla behaviour.
    act = act_in[worldid, act_last]
    tau_e = wp.max(MJ_MINVAL, dynprm[0])
    act_dot = (ctrl - act) / tau_e
    L_dyn = dynprm[2]
    if L_dyn > MJ_MINVAL:
        # omega at step start; ZOH over dt (kernel picks latest actuator_velocity).
        omega = actuator_velocity_in[worldid, uid]
        act_dot += (dynprm[3] - dynprm[1]) * omega / L_dyn
```

**healthy:** `dynprm[3] − dynprm[1] = 0` → 보정항 0 → 표준 1차 필터.
**demag:** `dynprm[1]`이 작아짐 → `(dynprm[3] − dynprm[1]) > 0` → ω가 양수일 때 act_dot 증가 (전류가 더 빠르게 따라옴, 손실된 토크 일부 회복 시도).

이 패치는 Method A 패치(2026-04-27)와 **독립적으로 작동**하며, 우리가 추가한 RHS 보정 / Schur와 호환됩니다 (ω가 같은 시점의 값이고, dynprm 슬롯 의미가 일관).

---

## 5. 실행 순서 (한 step 안에서)

```
fwd_actuation()                         (forward.py:861)
 └─ wp.launch(_actuator_force)          (forward.py:868)
     ├─ ctrl 계산 + clamp                (line 663-668)
     ├─ act_dot 계산                     (line 669-700)
     │   └─ FILTEREXACT 분기에 demag 보정 (line 679-693, 기존 패치)
     ├─ gain·ctrl_act + bias            (line 752)
     └─ Method A RHS 보정 (line 752-770, 새 패치) ★
         force += gain·(1−β)·(ctrl − I_old)

fwd_solve()                             (구체 호출은 추적 안 함)
 └─ deriv_smooth_vel()                  (derivative.py:228)
     ├─ wp.launch(_qderiv_actuator_passive_vel)   ★
     │   └─ Method A Schur scale → vel  (derivative.py:67-83, 새 패치)
     ├─ wp.launch(_qderiv_actuator_passive_actuation_sparse)
     │   └─ qDeriv += JᵀBJ              (derivative.py:97-150)
     └─ wp.launch(_qderiv_actuator_passive)
         └─ M_eff = qM − h·qDeriv       (derivative.py:153-190)

→ Cholesky / linesearch / Euler step (solver.py / euler.py)
 └─ wp.launch(_next_activation)
     └─ Method A β=1/(1+h/τ) 적분 (forward.py:147-160, 새 패치) ★
```
★ = Method A 패치 적용 부분.

---

## 6. dynprm[1] = 0 비활성화 (vanilla 회귀)

테스트나 비교를 위해 motor coupling을 끄고 싶다면 `dynprm[1] = 0.0`으로 설정.
- `_next_act`: `dynprm[1] != 0` 조건 false → vanilla filterexact β = exp(−h/τ)
- `_actuator_force` Method A 보정: 같은 조건 false → 보정 안 됨
- `_qderiv_actuator_passive_vel` Schur: `Ke_gr != 0` 조건 false → schur=0
- `_actuator_force` demag 보정: `(dynprm[3] − dynprm[1]) ≠ 0`이면 여전히 작동 (의도와 다름)

→ **완전히 vanilla로 가려면 `dynprm[1] = dynprm[2] = 0`** 같이 끄는 게 안전.

---

## 7. CPU 동등 코드 (참고)

`mujoco_src/src/engine/` 에 동등한 C 구현 (commit `6a7115de`, 2026-04-27):
| mjwarp (GPU) | mujoco_src (CPU) | 항목 |
|---|---|---|
| `forward.py:147-160` | `engine_forward.c:_next_act` 인근 | act 적분기 |
| `forward.py:752-770` | `engine_forward.c:486-514` | RHS 보정 |
| `derivative.py:67-83` + `addJTBJSparse` 흉내 | `engine_derivative.c:1147-1170` | Schur cross-Jacobian |

CPU는 `dynprm[3] > 0 ? exp : 1/(1+h/τ)`로 A+/A 분기를 지원하지만, GPU 패치는 **Method A 단일 모드**로 단순화됨. A+ 필요 시 별도 dynprm 슬롯에 플래그 배정 + β 분기 추가 필요.

---

## 8. 백업 및 롤백

| 백업 파일 | 원본 시점 |
|-----------|-----------|
| `mujoco_warp/_src/forward.py.original_vanilla` | 2026-04-21 12:04 (mjwarp 데모 패치 직전) |
| `mujoco_warp/_src/derivative.py.original_vanilla` | 2026-04-27 17:11 (Method A 패치 직전) |

**롤백:**
```bash
cp /home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/_src/forward.py.original_vanilla \
   /home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/_src/forward.py

cp /home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/_src/derivative.py.original_vanilla \
   /home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/_src/derivative.py
```

**재적용:** conda env 재생성 / mjwarp upgrade 시 패치가 사라지므로, 영구화하려면 별도 git 포크나 install hook이 필요.

---

## 9. 검증

`method_a_gpu_flow.ipynb` 실행으로 다음을 확인했음:

1. **Cell 5 (단일 step)**: I_1 = 1.1547 (이론 1.155), F_1 = 0.9353 (이론 0.935) — Method A β·ctrl 일치.
2. **Cell 6 (다단계)**: residual `I_sim - I_methodA_analytic` 최대 abs ≈ 1e-15 → 수치 오차 한계 내 정확.
3. **Cell 7 (Schur 토글)**: `dynprm[1]=0` vs `dynprm[1]=Ke·gr` 비교에서 후자가 ω 발산 억제 → Schur implicit damping 작동.
4. **Cell 8 (Demag)**: `dynprm[1]=0.5·dynprm[3]` 시 ω 증가 폭 커짐 → demag 보정 분기 활성, controller-plant mismatch 반영.
