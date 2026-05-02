# 수정한 코드의 위치 / 영향 / 복구 — 쉬운 설명

이 프로젝트에는 평범한 파이썬 코드 외에 **두 가지 종류의 "외부 코드 수정"** 이 있다.
- **GPU용 시뮬레이터(mjwarp) 안의 파이썬 파일을 직접 고친 것**
- **C 언어로 작성된 모터 콜백 파일**

각각 어디에 있고, 무슨 일을 하고, 사라질 위험이 있는지 / 어떻게 복구하는지 정리한다.

---

## 큰 그림 한 장으로

```
[ 파이썬 패키지 = 남이 만든 것, pip가 관리 ]
   ~/miniconda3/envs/mjlab/lib/python3.11/site-packages/
        └── mujoco_warp/_src/derivative.py   ← ★ 우리가 수정함 (GPU 시뮬 코드)
        └── mujoco_warp/_src/forward.py      ← ★ 우리가 수정함 (GPU 시뮬 코드)
   ↑ 이 폴더는 pip가 관리. pip install 하면 위 파일들이 원래대로 덮어쓰여짐.

[ 우리 프로젝트 = 우리가 만든 것, git이 관리 ]
   /home/rbdo/unitree_rl_mjlab/
        ├── src/.../electric_motor_callback.c  ← C 파일
        └── vendor/mujoco_warp_3.6.0_patch/    ← GPU 수정의 백업본 + 패치 파일
              ├── _src/derivative.py           (수정 완료된 사본)
              ├── _src/forward.py              (수정 완료된 사본)
              └── mujoco_warp_3.6.0_coupling.patch  (수정 내용 diff)
   ↑ 이 폴더는 git이 관리. pip install 과 무관, 절대 안 사라짐.
```

**핵심 한 줄**:
- **GPU 수정본**은 site-packages 라는 "pip가 관리하는 폴더" 안에 들어 있어서 pip 가 덮어쓸 수 있다.
- **C 파일**은 우리 프로젝트 폴더 안에 들어 있어서 pip 와 전혀 무관, 절대 안 사라진다.

---

## 1. GPU 코드 수정 (mjwarp 패치)

### 1-1. 무엇을 고쳤나

**파일 2개**:
- `mujoco_warp/_src/derivative.py` — 시뮬레이터의 미분 계산 부분
- `mujoco_warp/_src/forward.py` — 시뮬레이터의 정방향 적분 부분

**고친 내용** (요약):
1. 전기 모터의 **back-EMF (역기전력) 와 토크가 서로 영향을 주는 부분** 을 시뮬레이터가 동시에 풀도록 항을 추가 (Schur complement coupling).
2. 적분 방법을 **3 가지 (Method A / A+ / B)** 중에서 고를 수 있게 만들었음 (`dynprm[4]` 슬롯에 0/1/2 로 인코딩).
3. 감자(demag) 고장 시 컨트롤러가 쓰는 `Ke_nominal` 과 실제 플랜트의 `Ke_real` 차이를 시뮬레이터 내부 ODE 에 반영하는 항 추가.

### 1-2. 어디에 쓰이나

- 학습 (`scripts/train.py`) 과 평가 (`scripts/play.py`) 에서 `mujoco_warp` 를 import 하면 자동으로 이 수정본이 동작.
- 다음 cfg 들이 GPU 수정본의 새 기능을 실제로 활용:
  - `Unitree-Go2-Flat-Coupled-Electric` (Method A+)
  - `Unitree-Go2-Flat-Coupled-Tloop-Electric` (Method A+ + 적분 루프)
  - `Unitree-Go2-Flat-MethodA-Electric` (Method A)
  - `Unitree-Go2-Flat-MethodB-Electric` (Method B, **GPU 전용**)
- 이 수정이 **없으면 Method B 자체가 동작하지 않고**, A/A+ 도 cross-Jacobian 항 없이 부정확하게 풀린다.

### 1-3. 왜 pip install 하면 사라지나 — "패키지" 와 "폴더" 의 관계

파이썬 패키지(예: `mujoco_warp`) 는 인터넷 어딘가의 서버(PyPI) 에 올라가 있고, `pip install` 명령은 그 서버에서 파일을 받아 **`site-packages` 폴더에 통째로 복사** 하는 동작이다.

```
PyPI 서버 (인터넷)         pip install                 내 컴퓨터의 site-packages
mujoco-warp 3.6.0    ──────────────────▶              mujoco_warp/_src/derivative.py
                                                       mujoco_warp/_src/forward.py
                                                       ...
```

우리는 이 site-packages 안의 derivative.py / forward.py 를 **직접 손으로 고쳤다**.

문제는 — 누가 다음 중 하나를 실행하면:
- `pip install mujoco-warp`  (재설치)
- `pip install --upgrade mujoco-warp`  (업그레이드)
- `pip install -r requirements.txt` 안에 mujoco-warp 가 있을 때
- `conda env create` 로 환경을 처음부터 다시 만들 때
- 다른 사람이 같은 코드를 자기 컴퓨터에서 실행할 때

→ pip 는 **PyPI 의 깨끗한 원본 파일** 을 site-packages 에 다시 복사해서 우리 수정본을 **깨끗이 덮어쓴다.** 이 시점에 GPU 수정은 사라지고, 위 cfg 들은 정확도가 떨어지거나 (Method B 는) 동작하지 않게 된다.

### 1-4. 수정본의 "원본" 은 어디에 있나 — vendor/ 폴더의 역할

이런 사고를 대비해 **수정한 derivative.py / forward.py 의 사본** 과 **수정 내용을 기록한 .patch 파일** 을 프로젝트 안에 보관해 둠:

```
/home/rbdo/unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch/
   ├── _src/derivative.py          ← 수정 끝난 derivative.py (그냥 복사하면 됨)
   ├── _src/forward.py             ← 수정 끝난 forward.py
   └── mujoco_warp_3.6.0_coupling.patch  ← 무엇을 어떻게 고쳤는지의 diff 기록
```

이 폴더는 우리 프로젝트 안에 있고 git 이 관리하므로, **우리가 손으로 지우지 않는 한 절대 사라지지 않는다.**

추가로, 첫 패치 적용 시 원본을 백업해 둔 파일이 site-packages 에도 남아있다:
```
~/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/_src/
   ├── derivative.py.original_vanilla   ← 패치 전 mjwarp 3.6.0 의 원본
   └── forward.py.original_vanilla      ← 패치 전 mjwarp 3.6.0 의 원본
```

### 1-5. 사라졌을 때 복구 절차

상황: `pip install mujoco-warp` 를 실행했더니 학습이 이상하게 동작 / Method B 가 에러를 냄.

**복구 방법 1 — 수정 사본을 직접 복사 (가장 간단)**
```bash
PATCH=/home/rbdo/unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch
DEST=/home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/_src

cp "$PATCH/_src/derivative.py" "$DEST/derivative.py"
cp "$PATCH/_src/forward.py"    "$DEST/forward.py"
```

**복구 방법 2 — patch 명령으로 적용 (mjwarp 버전이 같은 3.6.0 일 때)**
```bash
cd /home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp
patch -p1 < /home/rbdo/unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch/mujoco_warp_3.6.0_coupling.patch
```

복구 후 확인:
```bash
md5sum /home/rbdo/unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch/_src/derivative.py \
       /home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/_src/derivative.py
# 두 줄의 해시값이 같으면 OK
```

### 1-6. mjwarp 버전이 올라가면? (3.6.0 → 3.7.x)

PyPI 에 mujoco-warp 3.7.0 이 나오고 누군가 그걸 설치하면, 우리의 patch 는 더 이상 적용되지 않을 가능성이 높다 (코드 라인이 바뀌어 있을 수 있음). 이 경우:
1. 새 버전의 derivative.py / forward.py 를 받아서
2. patch 파일의 변경 사항을 **새 버전에 다시 손으로 옮겨야** 한다.
3. 옮긴 결과를 다시 vendor/ 에 보관 (`mujoco_warp_3.7.0_patch/`).

지금은 3.6.0 에 고정해 둔 상태가 안전.

---

## 2. C 파일 (electric_motor_callback.c)

### 2-1. 무엇이고 어디 있나

```
/home/rbdo/unitree_rl_mjlab/src/assets/robots/unitree_go2/electric_motor_callback.c
```

**우리 프로젝트 폴더 안에 있는 파일**. git 으로 관리 (`git ls-files` 로 확인됨, 커밋 `d6871fd` 에서 추가됨).

### 2-2. 무슨 일을 하나

표준 MuJoCo (CPU 시뮬레이터) 의 "act_dyn 콜백" 인터페이스를 통해 전기 모터의 미분방정식을 직접 푸는 작은 프로그램:
```
L · dI/dt = V_cmd − R·I − Ke·gr·ω
```

### 2-3. 어디에 쓰이나 — **사실은 거의 안 씀**

- 활성화 조건: `NativeElectricActuatorCfg(use_callback=True)` 로 설정되어 있을 때 + `dyntype=user` 모드
- 현재 등록된 모든 task 와 cfg 인스턴스는 **`use_callback=False` (기본값)** → `dyntype=filterexact` 사용 → C 파일을 건드릴 일이 없다.
- 또한 C 파일을 안 쓰더라도 **같은 일을 하는 파이썬 함수가 이미 존재**:
  - `mj_native_electric_actuator.py` 의 `_act_dyn_callback()` 함수 (line 186)
  - 즉 `use_callback=True` 로 켜도 기본은 파이썬 콜백이 동작.

C 파일은 "파이썬 콜백이 너무 느릴 때 직접 컴파일해서 더 빠르게 쓸 수도 있다" 는 **선택지** 일 뿐. 실제 학습 / 평가에는 안 쓰임.

### 2-4. C 파일은 자동으로 컴파일되나?

**아니다.** 그냥 텍스트 파일로만 존재. 사용하려면 사용자가 직접 gcc 명령을 쳐야 한다 (파일 상단 주석에 빌드 명령이 적혀 있다):

```bash
gcc -shared -fPIC -O2 \
    -I$(python3 -c "import mujoco; print(mujoco.mj_path())")/include \
    -o electric_motor_callback.so electric_motor_callback.c
```

→ `electric_motor_callback.so` 라는 컴파일된 라이브러리가 생기고, 그걸 파이썬에서 `ctypes.CDLL("./electric_motor_callback.so")` 로 로드한다.

지금 시점에 이 `.so` 파일을 만든 적은 없음 → 그냥 소스 코드만 보관.

### 2-5. "다시 다운받으면 사라지나?" — 결론부터

| "다시 다운받는다" 가 의미하는 상황 | C 파일 안전한가? |
|---|---|
| `pip install <뭐든>` 를 실행 | ✅ **안전** (pip 와 무관) |
| `conda env create` 로 환경 재생성 | ✅ **안전** (환경에 안 들어 있음) |
| `git clone <이 프로젝트 URL>` 로 다른 컴퓨터에 받음 | ✅ **안전** (git 에 들어 있음) |
| 누군가 mujoco 자체를 PyPI 에서 새로 받음 | ✅ **안전** (mujoco 패키지에 안 들어 있음) |
| `git checkout` 으로 옛날 커밋(d6871fd 이전) 으로 이동 | ❌ 그 시점엔 이 파일이 없음 (다시 최신 커밋으로 이동하면 부활) |
| 누군가 손으로 파일을 `rm` 함 | ❌ **사라짐** (`git checkout HEAD -- <경로>` 로 복원 가능) |

**핵심**: C 파일은 우리 프로젝트의 일부 → git 이 관리 → pip / conda / mujoco 재설치 와 **완전히 무관**. 무서워할 일 없음.

### 2-6. 사라졌을 경우 복구

```bash
cd /home/rbdo/unitree_rl_mjlab
git checkout HEAD -- src/assets/robots/unitree_go2/electric_motor_callback.c
# 또는
git restore src/assets/robots/unitree_go2/electric_motor_callback.c
```

---

## 3. 두 가지 비교 표

| 항목 | GPU 수정 (derivative.py / forward.py) | C 파일 (electric_motor_callback.c) |
|---|---|---|
| 위치 | site-packages 안 (= pip 영역) | 우리 프로젝트 src/ 안 |
| 누가 관리? | pip | git |
| 학습 / 평가에서 실제 쓰임? | **예** (Coupled / MethodA/A+/B / Tloop 모두 사용) | **아니오** (use_callback=False 기본, 파이썬 콜백으로 충분) |
| 컴파일 필요? | 아니오 (파이썬 파일) | 네 (`.so` 만들려면 gcc 필요) |
| `pip install ...` 으로 사라짐? | **예** (덮어쓰기됨) | 아니오 |
| `conda env create` 로 사라짐? | 예 (환경 새로 만들면 mjwarp 도 새로 받음) | 아니오 |
| `git clone` 으로 사라짐? | 예 (vendor/ 폴더는 옴, 하지만 site-packages 적용은 따로 해야) | 아니오 (그대로 받음) |
| 사라졌을 때 복구 | `cp vendor/.../*.py site-packages/...` | `git checkout HEAD -- <경로>` |

---

## 4. 환경을 새로 셋업할 때의 체크리스트

새 컴퓨터 / 새 conda env 에서 학습을 시작하려 할 때:

```bash
# 1) 환경 만들고 의존성 설치
conda create -n mjlab python=3.11
conda activate mjlab
pip install -e .                 # 또는 requirements.txt 등

# 2) GPU 패치 적용 (이게 핵심)
PATCH=/home/rbdo/unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch
DEST=$(python -c "import mujoco_warp, os; print(os.path.dirname(mujoco_warp.__file__))")/_src
cp "$PATCH/_src/derivative.py" "$DEST/"
cp "$PATCH/_src/forward.py"    "$DEST/"

# 3) 적용 확인
python -c "import mujoco_warp" && echo "import OK"
md5sum "$PATCH/_src/derivative.py" "$DEST/derivative.py"
md5sum "$PATCH/_src/forward.py"    "$DEST/forward.py"
# 두 쌍의 해시가 각각 일치하면 패치 적용 완료
```

(C 파일은 어차피 `use_callback=False` 가 기본이라 이 단계에서 신경 쓸 일이 없다.)

---

## 5. "왜 이렇게 복잡한가" — 한 줄 정리

mjwarp 는 외부 패키지(우리가 만들지 않은 것) 라서 "내 프로젝트 폴더 안에 그냥 두기" 가 안 됨 → site-packages 에서 직접 수정해야 함 → 그런데 그 폴더는 pip 가 언제든 덮어쓸 수 있음 → 그래서 **수정한 결과의 사본** 과 **수정 명세 (.patch 파일)** 를 vendor/ 에 보관해서 언제든 다시 적용할 수 있게 만들어 둠.

이 구조만 이해하면 다른 외부 라이브러리에도 같은 패턴을 적용할 수 있음.
