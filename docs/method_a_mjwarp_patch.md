# Method A / A+ / B Coupling on mjwarp (GPU) — Patch Documentation

**Date:** 2026-04-27 (최초) / 2026-08-12 (vendored 코드에 맞춰 개정)
**Backend:** `mujoco_warp` (GPU)
**Scope:** back-EMF coupling for FILTEREXACT motor actuators — β 선택자(`dynprm[4]`)로 적분기/implicit 항의 감쇠 계수를 각각 고름.
**코드 기준:** `vendor/mujoco_warp_3.6.0_patch/_src/{forward,derivative}.py` (아래 줄 번호는 이 vendored 사본 기준)

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
| `β_int` | **적분기**가 쓰는 1-step decay factor (`_next_act`) |
| `β_imp` | **implicit 항**(Schur 좌변 + force RHS)이 쓰는 decay factor |

`β_int`와 `β_imp`는 서로 다른 값일 수 있고, `dynprm[4]`가 그 조합을 고릅니다 (§1 참조).

**dynprm 슬롯 (FILTEREXACT motor):**
| 슬롯 | 의미 |
|------|------|
| `dynprm[0]` | τ_e = L/R |
| `dynprm[1]` | **Ke_plant·gr** — 실제 plant의 Ke (demag로 변경 가능) |
| `dynprm[2]` | L (인덕턴스) |
| `dynprm[3]` | **Ke_nom·gr** — 명목 Ke (controller가 가정; 변경 안 됨) |
| `dynprm[4]` | **β 선택자** — 0 / 1 / ≥2 의 세 분기 (아래) |

`dynprm[4]` 분기 (coupling 활성 시에만 의미 있음):
| 값 | β_int (적분기) | 1−β_imp (Schur·RHS) | 코드 라벨 |
|---|---|---|---|
| `0` (기본) | `1/(1+h/τ)` | `h/(τ+h)` | A |
| `1` | `exp(−h/τ)` | `1−exp(−h/τ)` | A+ |
| `≥2` | `exp(−h/τ)` | `h/(τ+h)` | B |

즉 `0`은 세 사이트 모두 `1/(1+h/τ)` 계열로 일관, `1`은 세 사이트 모두 지수형으로 일관, `≥2`는 적분기만 지수형이고 implicit 항은 `1/(1+h/τ)` 계열을 씁니다.
설정 쪽 매핑은 `src/assets/robots/unitree_go2/mj_native_electric_actuator.py:77-84`의 `_METHOD_TO_DYNPRM4` (`method="A"/"A+"/"B"` → 0/1/2).

healthy 상태: `dynprm[1] == dynprm[3]`. demag 발생: `dynprm[1] < dynprm[3]`.

---

## 1. 적분기 β_int — `dynprm[4]` 3분기 선택자

### 수식
1차 ODE `dI/dt = (ctrl − I)/τ`의 1-step 갱신은 어느 쪽이든
$$
I_{n+1} = \beta_\text{int}\,I_n + (1-\beta_\text{int})\,\mathrm{ctrl}
$$
형태이고, 어떤 이산화 가정을 쓰느냐가 β_int를 정합니다.

- `ctrl`을 step 끝에서 평가하고 미분을 차분으로 근사 — `I_{n+1} = I_n + h(ctrl − I_{n+1})/τ`:
  $$\beta_\text{int} = \frac{1}{1+h/\tau}$$
- `ctrl`을 step 구간에서 상수로 두고 ODE를 닫힌 형태로 적분:
  $$\beta_\text{int} = e^{-h/\tau}$$

`dynprm[4]`가 이 둘 중 하나를 고릅니다: `0` → `1/(1+h/τ)`, `1`과 `≥2` → `exp(−h/τ)`.
(`1`과 `≥2`는 적분기는 같고 implicit 항의 β_imp에서 갈립니다 — §2·§3.)

vanilla MuJoCo의 filterexact는 항상 `exp(−h/τ)`를 씁니다. 즉 coupling이 꺼져 있을 때의 fallback과 `dynprm[4] ≥ 1`의 적분기는 같은 식이고, 차이는 Schur/RHS 보정이 붙느냐에 있습니다. h가 작으면 두 β_int의 차이는 O(h²/τ²)로 작지만, `dynprm[4]=0`은 implicit 항과 β를 일치시키는 쪽을 택한 것입니다.

### 패치 위치
**`forward.py:147-166`** (function `_next_act`):
```python
if actuator_dyntype == DynType.FILTEREXACT:
    tau = wp.max(MJ_MINVAL, actuator_dynprm[0])
    # Motor coupling detection: dynprm[1]=Ke*gr != 0 AND dynprm[2]=L > 0
    # Integrator β_int selector via dynprm[4]:
    #   0 → Method A  : β_int = 1/(1+h/τ)        (BE)
    #   1 → Method A+ : β_int = exp(-h/τ)        (ZOH)
    #   2 → Method B  : β_int = exp(-h/τ)        (ZOH integrator, BE Schur/Force elsewhere)
    # No coupling     → vanilla filterexact: β = exp(-h/τ)
    if actuator_dynprm[1] != 0.0 and actuator_dynprm[2] > MJ_MINVAL:
        if actuator_dynprm[4] > 1.5:
            # B: ZOH integrator (Schur/Force RHS sites use BE — kept BE there).
            act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))
        elif actuator_dynprm[4] > 0.0:
            # A+: act = act + τ * act_dot * (1 - exp(-h/τ))
            act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))
        else:
            # A IE: act = act + h * act_dot / (1 + h/τ)
            act = act_in + act_dot_scale * act_dot_in * opt_timestep / (1.0 + opt_timestep / tau)
    else:
        act = act_in + act_dot_scale * act_dot_in * tau * (1.0 - wp.exp(-opt_timestep / tau))
```

**작동 조건:** `dynprm[1] ≠ 0 && dynprm[2] > 0` (motor coupling 활성).
**분기 비교:** 임계값이 `> 1.5` / `> 0.0`이므로 `dynprm[4]`는 float으로 넣어도 되고, 2 이상은 모두 마지막 분기로 들어갑니다.
**fallback:** coupling 조건 불충족 시 vanilla filterexact 그대로 (다른 FILTEREXACT actuator에 영향 없음). 이때 `dynprm[4]`는 읽히지 않습니다.

---

## 2. Schur Cross-Jacobian — implicit solver의 좌변

### 유도 (요약)
전기-역학 coupled 1-step 풀이에서 ω의 Δq̈ 의존성을 풀면:
$$
[M + (1-\beta_\text{imp})\,K_t g_r\,K_e g_r\,\frac{h}{R}\,J^\top J]\,\ddot q = (\text{constants})
$$

MuJoCo implicit 컨벤션 `M_\text{eff} = qM - h\cdot\mathrm{qDeriv}`와 비교:
$$
\mathrm{qDeriv} \mathrel{+}= -(1-\beta_\text{imp})\,\frac{K_t g_r\,K_e g_r}{R}\,J^\top J
$$

→ B 스칼라로 표현: `B = -(1-β_imp)·Kt·gr·Ke·gr/R` (음수, implicit damping).

여기서 `1-β_imp`는 `dynprm[4]`로 갈립니다:
$$
1-\beta_\text{imp} =
\begin{cases}
\dfrac{h}{\tau+h} & \texttt{dynprm[4]} = 0 \\[6pt]
1 - e^{-h/\tau} & \texttt{dynprm[4]} = 1 \\[6pt]
\dfrac{h}{\tau+h} & \texttt{dynprm[4]} \ge 2
\end{cases}
$$
`dynprm[4] ≥ 2`는 적분기(§1)만 지수형이고 여기 좌변은 `h/(τ+h)`를 유지합니다. `R`은 `dynprm[2]/dynprm[0] = L/τ`로 복원합니다.

부호 확인: `M_eff = qM − h·qDeriv`의 음수 qDeriv → `+h·|qDeriv|` 가 M에 더해져 양정정 강화. 안정.

### 패치 위치
**`derivative.py:31-113`** (kernel `_qderiv_actuator_passive_vel`, Schur 블록은 `68-89`):
```python
# Schur term: -(1-β_imp)·Kt·gr·Ke·gr/R for FILTEREXACT motor coupling.
# β_imp selector via dynprm[4]:
#   0 → Method A  : 1-β_imp = h/(τ+h)            (BE)
#   1 → Method A+ : 1-β_imp = 1 - exp(-h/τ)      (ZOH)
#   2 → Method B  : 1-β_imp = h/(τ+h)            (BE — paired with ZOH integrator)
schur = float(0.0)
if actuator_dyntype[actid] == DynType.FILTEREXACT:
    dynprm_act = actuator_dynprm[actuator_dynprm_id, actid]
    Ke_gr = dynprm_act[1]
    L_val = dynprm_act[2]
    if Ke_gr != 0.0 and L_val > MJ_MINVAL:
        tau_e = wp.max(MJ_MINVAL, dynprm_act[0])
        Kt_gr = actuator_gainprm[actuator_gainprm_id, actid][0]
        h_dt = opt_timestep[worldid % opt_timestep.shape[0]]
        if dynprm_act[4] > 1.5:
            one_minus_beta = h_dt / (tau_e + h_dt)        # B (BE)
        elif dynprm_act[4] > 0.0:
            one_minus_beta = 1.0 - wp.exp(-h_dt / tau_e)  # A+
        else:
            one_minus_beta = h_dt / (tau_e + h_dt)        # A IE
        R_val = L_val / tau_e
        schur = -one_minus_beta * Kt_gr * Ke_gr / R_val
...
vel = float(bias) + schur
```

**주의:** 여기 Schur 항은 `dynprm[1] = Ke_plant·gr` (plant 값)을 씁니다. demag로 `dynprm[1]`이 줄면 좌변 implicit damping도 같이 줄어들고, 그 mismatch는 RHS 쪽(§3)에서 별도 항으로 반영됩니다.

여기서 `vel`은 이후 `_qderiv_actuator_passive_actuation_sparse` 커널 (`derivative.py:124-178`)에서 다음 식으로 누적됩니다:
```
qderiv_contrib += moment_i * moment_j * vel
```
즉, `qDeriv += JᵀBJ` 형태로, B = vel = bias + schur.

### Caller 패치
**`derivative.py:256-294`** (function `deriv_smooth_vel`):
`_qderiv_actuator_passive_vel`에 `actuator_dynprm`과 `opt_timestep`을 새로 전달.

### 중요한 부호/스케일 메모
이전 patch (mujoco_src commit `cf35fd22`, 2026-04-10)에서는 `+h²·...` 스케일 + 부호도 반대로 잘못 들어가 있었음. 오늘 commit `6a7115de`에서 부호 및 dt 차수 모두 교정됨. mjwarp 패치는 그 교정된 식을 그대로 따름.

---

## 3. RHS 보정 — Force가 I_predicted를 사용하도록

### 수식
vanilla mjwarp의 `_actuator_force` 커널은 `force = gain·ctrl_act + bias`를 계산하는데, 여기서 `ctrl_act = act_in[act_last] = I_n` (현재 전류). coupling 모드에서는 force가 **다음 step의 예측 전류** I_{n+1}을 사용해야 합니다.

plant의 전류 미분(§4)은 두 항으로 이루어져 있습니다:
$$
\dot I = \frac{\mathrm{ctrl} - I_n}{\tau} + \frac{(K_e^{\text{nom}} - K_e^{\text{plant}})\,g_r\,\omega}{L}
\;=\;\frac{1}{\tau}\Big[(\mathrm{ctrl} - I_n) + \frac{(K_e^{\text{nom}} - K_e^{\text{plant}})\,g_r\,\omega}{R}\Big]
$$
(두 번째 등식은 τ = L/R 사용.) 따라서 예측 전류와 그 force 보정은
$$
I_{n+1} = I_n + (1-\beta_\text{imp})\Big[(\mathrm{ctrl} - I_n) + \frac{(K_e^{\text{nom}} - K_e^{\text{plant}})\,g_r\,\omega}{R}\Big]
$$
$$
\Delta F = \underbrace{K_t g_r (1-\beta_\text{imp})(\mathrm{ctrl} - I_n)}_{\text{filter piece}}
\;+\; \underbrace{K_t g_r (1-\beta_\text{imp})\,\frac{(K_e^{\text{nom}} - K_e^{\text{plant}})\,g_r\,\omega}{R}}_{\text{demag piece}}
$$

즉 RHS 보정은 **두 조각**입니다. 두 번째 조각은 controller가 가정한 Ke(`dynprm[3]`)와 plant의 Ke(`dynprm[1]`)가 다를 때만 살아나며, healthy(`dynprm[3] == dynprm[1]`)에서는 0이 되어 예전 문서의 단일 항 식과 같아집니다. `R`은 `dynprm[2]/dynprm[0] = L/τ`로 복원하고, ω는 §4의 act_dot 보정과 **같은 시점 값**(`actuator_velocity_in`)을 씁니다.

`1-β_imp`는 §2와 동일한 `dynprm[4]` 분기: `0` → `h/(τ+h)`, `1` → `1−exp(−h/τ)`, `≥2` → `h/(τ+h)`.

### 패치 위치
**`forward.py:763-796`** (kernel `_actuator_force`):
```python
force = gain * ctrl_act + bias

# Method A/A+/B RHS correction (Schur complement RHS):
# force currently uses I_old (act_in[act_last]) when not actearly.
# Schur RHS contribution: (1-β_imp)·(K_t·gr/R)·F_elec, where
#   F_elec = R·(ctrl - I_old) + (Ke_nom·gr - Ke_plant·gr)·omega
# gives two pieces:
#   ΔF_filter = gain·(1-β_imp)·(ctrl - I_old)                          (vanilla)
#   ΔF_demag  = gain·(1-β_imp)·(dynprm[3] - dynprm[1])·omega / R       (Ke mismatch)
# Healthy (dynprm[3]==dynprm[1]) ⇒ ΔF_demag = 0 ⇒ vanilla behaviour.
# β_imp selector via dynprm[4]:
#   0 → A   : 1-β_imp = h/(τ+h)            (BE)
#   1 → A+  : 1-β_imp = 1 - exp(-h/τ)      (ZOH)
#   2 → B   : 1-β_imp = h/(τ+h)            (BE — paired with ZOH integrator)
# Active only for FILTEREXACT motor actuators (dynprm[1]≠0 AND dynprm[2]>0).
if na and act_first >= 0:
    if actuator_dyntype[uid] == DynType.FILTEREXACT:
        dynprm_uid = actuator_dynprm[worldid % actuator_dynprm.shape[0], uid]
        if dynprm_uid[1] != 0.0 and dynprm_uid[2] > MJ_MINVAL and not actuator_actearly[uid]:
            tau_e = wp.max(MJ_MINVAL, dynprm_uid[0])
            h_dt = opt_timestep[worldid % opt_timestep.shape[0]]
            if dynprm_uid[4] > 1.5:
                one_minus_beta = h_dt / (tau_e + h_dt)        # B (BE)
            elif dynprm_uid[4] > 0.0:
                one_minus_beta = 1.0 - wp.exp(-h_dt / tau_e)  # A+
            else:
                one_minus_beta = h_dt / (tau_e + h_dt)        # A IE
            I_old = act_in[worldid, act_last]
            force += gain * one_minus_beta * (ctrl - I_old)
            # Demag mismatch piece of the Schur RHS (Ke_nom ≠ Ke_plant).
            # R is reconstructed as L/τ from dynprm[2]/dynprm[0]; same omega as act_dot.
            R_eff = dynprm_uid[2] / tau_e
            omega_uid = actuator_velocity_in[worldid, uid]
            force += gain * one_minus_beta * (dynprm_uid[3] - dynprm_uid[1]) * omega_uid / R_eff
```

**작동 조건:**
- FILTEREXACT motor (`dynprm[1] ≠ 0 && dynprm[2] > 0`)
- `actearly == False` (actearly=True면 force가 이미 `_next_act` 결과를 쓰므로 중복 보정 안 됨)
- demag 조각은 추가로 `dynprm[3] ≠ dynprm[1]`일 때만 비-0

**왜 `ctrl`인가?** 이 시점의 `ctrl`은 위쪽에서 이미 clamp된 값. 사용자의 Python에서 `ctrl = (V − Ke_nom·gr·ω_old)/R` 형태로 들어와 있어, 이게 "current 단위의 steady-state" 의미. β-blend로 I_predicted를 만들어 force에 반영.

**§4와의 관계:** §4의 demag 항은 `act_dot`(→ 다음 step의 act)에, 여기 demag 조각은 **이번 step의 force**에 들어갑니다. 둘은 같은 mismatch 항을 서로 다른 사이트에서 반영하는 것이라 중복이 아니며, `(ctrl − I_old)` 조각이 `act_dot`의 필터 항과 짝을 이루는 구조와 동일합니다.

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
**`forward.py:692-704`** (kernel `_actuator_force`, FILTEREXACT branch — 이미 4/21에 적용된 기존 패치):
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

이 항은 `dynprm[4]`와 무관하게 항상 같은 식(`(dynprm[3] − dynprm[1])·ω/L`)이며, β 선택자는 이 act_dot이 적분/force에 얹히는 방식(§1·§3)에만 영향을 줍니다. §3의 RHS 보정에 있는 demag 조각이 이 항의 force-side 짝입니다.

---

## 5. 실행 순서 (한 step 안에서)

```
fwd_actuation()                         (forward.py:904)
 └─ wp.launch(_actuator_force)          (forward.py:911)
     ├─ ctrl 계산 + clamp                (line 673-678)
     ├─ act_dot 계산                     (line 680-714)
     │   └─ FILTEREXACT 분기에 demag 보정 (line 692-704, 기존 패치)
     ├─ gain·ctrl_act + bias            (line 763)
     └─ RHS 보정 (line 765-796, 새 패치) ★
         force += gain·(1−β_imp)·(ctrl − I_old)                        (filter 조각)
         force += gain·(1−β_imp)·(dynprm[3]−dynprm[1])·ω/R_eff         (demag 조각)

fwd_solve()                             (구체 호출은 추적 안 함)
 └─ deriv_smooth_vel()                  (derivative.py:256)
     ├─ wp.launch(_qderiv_actuator_passive_vel)   ★
     │   └─ Schur scale → vel           (derivative.py:68-89, 새 패치)
     ├─ wp.launch(_qderiv_actuator_passive_actuation_sparse)
     │   └─ qDeriv += JᵀBJ              (derivative.py:124-178)
     └─ wp.launch(_qderiv_actuator_passive)
         └─ M_eff = qM − h·qDeriv       (derivative.py:180-220)

→ Cholesky / linesearch / Euler step (solver.py / euler.py)
 └─ wp.launch(_next_activation)
     └─ β_int 분기 적분 (forward.py:147-166, 새 패치) ★
```
★ = coupling 패치 적용 부분. `dynprm[4]`는 ★ 표시된 세 사이트에서 각각 읽힙니다 (적분기 1곳 = β_int, Schur·RHS 2곳 = β_imp).

---

## 6. dynprm[1] = 0 비활성화 (vanilla 회귀)

테스트나 비교를 위해 motor coupling을 끄고 싶다면 `dynprm[1] = 0.0`으로 설정.
- `_next_act`: `dynprm[1] != 0` 조건 false → vanilla filterexact β = exp(−h/τ) (`dynprm[4]` 무시)
- `_actuator_force` RHS 보정: 같은 조건 false → 두 조각 모두 보정 안 됨 (`dynprm[4]` 무시)
- `_qderiv_actuator_passive_vel` Schur: `Ke_gr != 0` 조건 false → schur=0 (`dynprm[4]` 무시)
- `_actuator_force` act_dot demag 보정: `(dynprm[3] − dynprm[1]) ≠ 0`이면 여전히 작동 (의도와 다름)

→ **완전히 vanilla로 가려면 `dynprm[1] = dynprm[2] = 0`** 같이 끄는 게 안전.

---

## 7. CPU 동등 코드 (참고)

`mujoco_src/src/engine/` 에 동등한 C 구현 (최초 대응: commit `6a7115de`, 2026-04-27):
| mjwarp (GPU) | mujoco_src (CPU) | 항목 |
|---|---|---|
| `forward.py:147-166` | `engine_forward.c:_next_act` 인근 | act 적분기 (β_int) |
| `forward.py:763-796` | `engine_forward.c:486-514` 인근 | RHS 보정 (β_imp, filter + demag 조각) |
| `derivative.py:68-89` + `addJTBJSparse` 흉내 | `engine_derivative.c:1147-1170` 인근 | Schur cross-Jacobian (β_imp) |

**β 분기 상태 (2026-08-12 기준):** GPU 패치는 더 이상 단일 모드가 아니라 `dynprm[4]` 선택자로 세 조합을 지원합니다 (§0 표). 이전 판 문서의 "GPU는 Method A 단일 모드", "A+ 필요 시 별도 슬롯 배정 + β 분기 추가 필요"라는 서술은 그 작업이 이미 완료되어 무효입니다. 초기 판에서 A/A+ 분기 플래그로 쓰던 `dynprm[3]`은 이제 demag용 `Ke_nom·gr` 전용이고, 분기 플래그는 `dynprm[4]`로 분리되었습니다.

**CPU 쪽 주의:** stock(패치 안 된) MuJoCo는 `dynprm[4]`를 읽지 않고 항상 `exp(−h/τ)` 적분기를 쓰며 Schur/RHS 보정 자체가 없습니다. 따라서 stock CPU에서는 `dynprm[4]` 0/1/2가 모두 같은 동작으로 축약됩니다 (`src/assets/robots/unitree_go2/mj_native_electric_actuator.py:77-84`, `:135-146`). 패치된 `mujoco_src` 브랜치의 β 분기 동기화 여부는 이 저장소에 vendored 되어 있지 않으므로 해당 브랜치에서 직접 확인해야 합니다.

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

`method_a_gpu_flow.ipynb` 실행으로 다음을 확인했음 (2026-04-27, `dynprm[4]` 도입 이전 = 현재의 `dynprm[4] = 0` 분기에 해당. `1`/`≥2` 분기는 이 노트북으로 검증되지 않았음):

1. **Cell 5 (단일 step)**: I_1 = 1.1547 (이론 1.155), F_1 = 0.9353 (이론 0.935) — Method A β·ctrl 일치.
2. **Cell 6 (다단계)**: residual `I_sim - I_methodA_analytic` 최대 abs ≈ 1e-15 → 수치 오차 한계 내 정확.
3. **Cell 7 (Schur 토글)**: `dynprm[1]=0` vs `dynprm[1]=Ke·gr` 비교에서 후자가 ω 발산 억제 → Schur implicit damping 작동.
4. **Cell 8 (Demag)**: `dynprm[1]=0.5·dynprm[3]` 시 ω 증가 폭 커짐 → demag 보정 분기 활성, controller-plant mismatch 반영.
