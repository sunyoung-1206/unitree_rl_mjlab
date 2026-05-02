# 감자(Demagnetization) Fault Injection 실험 — 학습된 Policy 평가

## 목적

이미 학습된 RL policy를 로드한 뒤, 평가(inference) 시점에 모터 감자 현상을 주입하여 세 가지 액추에이터 모델의 응답 차이를 비교한다.

감자 = 영구자석 자속 감소 → Ke, Kt가 동시에 줄어드는 현상.

### 핵심 가설

| Ke·gr 감소 시 | PD (전기모델 없음) | Native (staggered) | Coupled (Schur) |
|---|---|---|---|
| 전류 | 없음 | ↑ 증가 | ↑ 증가 |
| 토크 | **변화 없음** | ↓ 감소 | ↓ 감소 |
| 보행 성능 | **변화 없음** | ↓ 저하 | ↓ 저하 |

- PD vs (Native, Coupled) 차이 = **전기 모델링의 효과**
- Native vs Coupled 차이 = **Schur 커플링의 추가 효과**
- 감자가 심할수록 Native-Coupled 차이가 벌어질 수 있음 (back-EMF 댐핑 약화 → intra-step Δω 증가)

---

## 환경 정보

- conda 환경: `mjlab`
- 프로젝트 루트: IsaacLab 기반 (정확한 경로는 기존 학습 로그 위치에서 추정)
- MuJoCo: 수정된 3.6.0 소스 빌드 (Schur complement 코드 포함)
- GPU: NVIDIA RTX 5080

### 세 가지 Task

| Task ID | 액추에이터 | dynprm | gainprm[0] |
|---|---|---|---|
| `Unitree-Go2-Flat` | PD 제어 (전기모델 없음) | 해당없음 | 해당없음 |
| `Unitree-Go2-Flat-Native-Electric` | filterexact (기본) | `[0.000333, 0, 0]` | `0.81024` (Kt·gr) |
| `Unitree-Go2-Flat-Coupled-Electric` | filterexact + Schur | `[0.000333, 0.81024, 0.0001]` | `0.81024` (Kt·gr) |

### 모터 파라미터 (정상 상태)

```
R     = 0.3      Ω
L     = 0.0001   H
Kt    = 0.128    N·m/A
Ke    = 0.128    V·s/rad
gr    = 6.33
Kt·gr = 0.81024  N·m/A (= gainprm[0])
Ke·gr = 0.81024  V·s/rad (= dynprm[1])
τ_e   = L/R = 0.000333 s (= dynprm[0])
```

---

## Step 0: 학습된 체크포인트 찾기

학습 로그는 `logs/rsl_rl/` 하위에 있을 것이다. 세 Task에 대해 각각 학습된 checkpoint를 찾아라.

```bash
# 로그 디렉토리 구조 확인
find logs/rsl_rl/ -name "model_*.pt" | head -30
```

각 Task별로 가장 높은 iteration의 checkpoint 경로를 기록하라. 예:
- PD: `logs/rsl_rl/.../model_XXXX.pt`
- Native: `logs/rsl_rl/.../model_XXXX.pt`
- Coupled: `logs/rsl_rl/.../model_XXXX.pt`

> **만약 특정 Task의 checkpoint가 없다면**: 해당 Task를 먼저 학습시켜야 한다. 하지만 학습 없이 실험을 먼저 구성하고, 어떤 checkpoint가 누락인지 나에게 보고하라.

---

## Step 1: 평가 스크립트 작성

`scripts/eval_demagnetization.py`를 만들어라.

### 핵심 로직

1. 학습된 policy를 로드한다 (play.py 방식과 동일)
2. 환경을 생성하되, **dynprm과 gainprm을 런타임에 수정**하여 감자를 주입한다
3. N 에피소드 동안 rollout하면서 데이터를 수집한다
4. 결과를 저장한다

### 감자 주입 방법

감자는 Ke와 Kt가 동시에 같은 비율로 줄어드는 것이다 (Kt = Ke in SI).

**감자 수준 (demagnetization factor, 정상 대비 비율):**
- `demag_factor = [1.0, 0.8, 0.6, 0.4]`

각 factor에 대해:
```python
new_Ke_gr = 0.81024 * demag_factor   # dynprm[1]
new_Kt_gr = 0.81024 * demag_factor   # gainprm[0]
```

**수정 위치 — MuJoCo 모델 파라미터를 직접 변경:**

```python
# MuJoCo 모델의 actuator 파라미터 수정
# model.actuator_dynprm: shape (nu, 10) — dynprm[1] = Ke·gr
# model.actuator_gainprm: shape (nu, 10) — gainprm[0] = Kt·gr
for i in range(model.nu):
    model.actuator_dynprm[i, 1] = new_Ke_gr    # Coupled에서만 의미 있음
    model.actuator_gainprm[i, 0] = new_Kt_gr   # 토크 계산에 사용됨
```

> **중요**: 이 수정이 IsaacLab/MuJoCo 인터페이스에서 어떻게 접근되는지 확인해야 한다.
> GPU 시뮬레이션(mujoco_warp)에서는 접근 방식이 다를 수 있다.
> IsaacLab의 환경 설정 코드를 조사하여 올바른 수정 지점을 찾아라.

### Task별 감자 주입 차이

| Task | dynprm[1] 수정 | gainprm[0] 수정 | 예상 효과 |
|---|---|---|---|
| PD (`Unitree-Go2-Flat`) | 해당없음 (PD는 dynprm 안 씀) | 해당없음 (PD는 gainprm 안 씀) | **변화 없음** — 이게 baseline |
| Native | dynprm[1]=0 (원래도 0) | gainprm[0] 수정 | 토크 감소 있음 |
| Coupled | dynprm[1] 수정 | gainprm[0] 수정 | 토크 감소 + Schur 보정 변화 |

> **PD Task 주의**: PD 제어 방식에서는 `action → PD controller → 토크`로 직접 가기 때문에 gainprm이 토크 계산에 관여하지 않을 수 있다. PD 환경의 액추에이터 설정을 확인하여, 감자 주입이 정말로 효과가 없는지 확인하라. 만약 PD에서도 gainprm이 토크에 영향을 준다면, PD 환경에서는 감자 주입을 하지 않고 그냥 정상 상태로 반복 실행하여 baseline으로 사용하라.

### 수집할 데이터 (매 timestep)

```python
data = {
    "act":              [],  # 전류 (actuator activation), shape: (nu,)
    "qfrc_actuator":    [],  # 액추에이터 토크, shape: (nv,)
    "qvel":             [],  # 관절 속도, shape: (nv,)
    "ctrl":             [],  # 제어 입력, shape: (nu,)
    "reward":           [],  # 스텝 보상 (scalar)
    "base_lin_vel":     [],  # 몸체 선속도, shape: (3,)
    "command_vel":      [],  # 명령 속도, shape: (3,)
}
```

> `act`, `qfrc_actuator`, `qvel`, `ctrl`은 MuJoCo data 객체에서 직접 읽는다.
> `reward`, `base_lin_vel`, `command_vel`은 IsaacLab 환경 인터페이스에서 읽는다.
> 정확한 접근 방법은 기존 play.py 코드를 참고하라.

### 실행 구성

```
3 Tasks × 4 demag_factors × 1 seed = 12 runs
각 run: 1000 steps (또는 적절한 에피소드 길이)
환경 수: 평가이므로 적은 수 가능 (예: 64~256)
```

### 출력 파일

각 run의 결과를 pickle 또는 npz로 저장:

```
results/demagnetization/
├── PD_demag_1.0.npz
├── PD_demag_0.8.npz
├── PD_demag_0.6.npz
├── PD_demag_0.4.npz
├── Native_demag_1.0.npz
├── Native_demag_0.8.npz
├── ...
├── Coupled_demag_1.0.npz
├── Coupled_demag_0.8.npz
├── ...
└── summary.json   ← 각 run의 통계 요약
```

---

## Step 2: 분석 및 시각화

`scripts/analyze_demagnetization.py`를 만들어라.

### Figure 1: 전류 응답 비교 (3×4 subplot grid)

- 행: Task (PD, Native, Coupled)
- 열: demag_factor (1.0, 0.8, 0.6, 0.4)
- 각 subplot: 12개 관절의 평균 |전류| 시계열
- PD 행은 전류가 없으므로 "N/A" 또는 빈 플롯

### Figure 2: 토크 응답 비교 (3×4 subplot grid)

- 동일 구조
- 각 subplot: 12개 관절(floating base 6 DOF 제외)의 평균 |토크| 시계열

### Figure 3: 성능 저하 요약 (bar chart)

- x축: demag_factor [1.0, 0.8, 0.6, 0.4]
- y축: 정규화된 성능 지표 (demag=1.0 대비 비율)
- 3개 bar group (PD, Native, Coupled)
- 성능 지표 후보:
  - 누적 보상 (episode return)
  - 속도 추종 오차 (|command_vel - base_lin_vel| RMS)
  - 생존 시간 (넘어지기 전까지 step 수)

### Figure 4: 전류-토크 관계 scatter plot

- x축: 평균 |전류|
- y축: 평균 |토크|
- 4개 점 (demag_factor별), Native와 Coupled를 같은 축에
- 감자가 심해질수록: 전류 ↑, 토크 ↓ 방향으로 이동해야 정상
- PD는 전류가 없으므로 제외

### Figure 5: Native vs Coupled 차이 (핵심 그래프)

- x축: demag_factor
- y축: |Native 성능 - Coupled 성능| / Coupled 성능
- 감자가 심해질수록 차이가 벌어지는지 확인
- 이 그래프가 "Schur 커플링이 fault 조건에서 더 의미 있다"는 것을 보여주는 핵심

### 모든 그래프

- matplotlib 사용
- 한글 폰트 설정 (NanumGothic 또는 시스템에서 사용 가능한 한글 폰트)
- 저장 경로: `results/demagnetization/figures/`
- DPI: 150

---

## Step 3: 결과 해석 가이드

실험이 끝나면 다음을 확인하고 보고하라:

### 필수 확인 사항

1. **PD baseline이 정말 변화 없는가?**
   - demag_factor를 바꿔도 PD의 토크와 보행 성능이 일정한지 확인
   - 일정하다면: 전기 모델 없이는 감자가 반영 안 됨 → 실험 설계 정당화
   - 일정하지 않다면: PD 환경에서도 gainprm이 토크에 영향을 주는 것 → 실험 설계 수정 필요, 나에게 보고

2. **Native/Coupled에서 전류 증가 + 토크 감소 패턴이 나타나는가?**
   - 이 패턴이 나타나면: 전기 모델 커플링이 올바르게 작동
   - 전류가 증가하는데 토크도 증가하면: 뭔가 잘못됨

3. **감자가 심해질수록 Native-Coupled 차이가 벌어지는가?**
   - 벌어지면: Schur 커플링이 fault 조건에서 더 가치 있음을 보여줌
   - 차이가 없으면: 정상 조건과 마찬가지로 기계적 low-pass filtering이 차이를 흡수

4. **숫자 범위 이상 없는가?**
   - 전류가 actrange (±29A)를 초과하는가
   - NaN이나 inf가 발생하는가
   - 로봇이 즉시 넘어지는가 (demag=0.4에서는 가능)

---

## 실행 순서 요약

```
1. 학습된 checkpoint 3개 확인
2. IsaacLab에서 dynprm/gainprm 런타임 수정 방법 조사
3. eval_demagnetization.py 작성
4. 12개 run 실행 (3 Tasks × 4 demag levels)
5. analyze_demagnetization.py 작성 및 실행
6. 결과 + 그래프를 results/demagnetization/에 저장
7. 핵심 발견을 터미널에 출력
```

---

## 제약사항

- 학습은 하지 않는다. 이미 학습된 policy를 그대로 사용한다.
- 감자 주입은 평가 시점에만 적용한다 (학습 시에는 정상 파라미터로 학습됨).
- 이 실험의 핵심은 "학습 때와 다른 물리 조건에서 policy가 어떻게 반응하는가"이다.
- 세 Task 모두 동일한 command velocity를 사용하라 (예: 전진 0.5 m/s 또는 환경 기본값).
