# mujoco_warp 패치 — 뉴비를 위한 차근차근 가이드

이 문서는 **"GPU 시뮬레이터(mujoco_warp)를 우리가 직접 고쳐서 쓰고 있다"** 라는 사실을 처음 듣는 사람을 위한 안내서다. 파이썬/pip/conda를 막 배운 정도면 따라올 수 있게 풀어 썼다.

---

## 0. 한 줄 요약

> GPU 시뮬레이터 코드(`mujoco_warp`)는 남이 만든 것이라 우리 프로젝트 폴더에 없다. 그래서 **남의 폴더(site-packages)** 에 들어가서 직접 고쳐 썼는데, 그 폴더는 pip가 마음대로 덮어쓸 수 있어서 **수정본의 사본을 우리 프로젝트 안(`vendor/`)에 따로 보관**해 두었다. 사라지면 사본을 다시 복사해 넣으면 된다.

---

## 1. 사전 지식 — "파이썬 패키지"가 컴퓨터 어디에 있는가

### 1-1. `pip install` 이 실제로 하는 일

`pip install numpy` 같은 명령을 치면, pip는 인터넷(PyPI 서버)에서 파일을 받아서 내 컴퓨터의 **정해진 폴더 한 곳**에 통째로 복사한다. 그 폴더 이름이 **site-packages** 다.

```
PyPI 서버 (인터넷)            pip install            내 컴퓨터
mujoco-warp 3.6.0      ─────────────────────▶      site-packages/mujoco_warp/...
                                                     (수십~수백 개의 .py 파일들)
```

지금 이 프로젝트의 site-packages 경로는:

```
/home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/
```

`mjlab` 이라는 conda 환경 안에 들어 있는 site-packages다. (conda 환경마다 별도의 site-packages가 있음.)

### 1-2. "import" 는 site-packages 에서 파일을 찾는다

파이썬 코드에서 `import mujoco_warp` 라고 쓰면, 파이썬은 site-packages 폴더를 뒤져서 `mujoco_warp/` 라는 폴더를 찾고, 그 안의 `__init__.py` 를 실행한다. 즉:

```python
import mujoco_warp
# ↑ 이 한 줄은 실제로는 다음 파일을 읽는다:
#   /home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/__init__.py
```

**핵심**: 우리 프로젝트의 `train.py` 가 `mujoco_warp` 를 import 하면, **남의 폴더(site-packages)에 있는 코드가 실제로 실행된다.**

---

## 2. 왜 site-packages 안의 파일을 직접 고쳤나

### 2-1. 우리가 고친 파일

site-packages 안에 있는 다음 두 파일을 **직접 손으로 수정**했다:

```
~/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/_src/
   ├── derivative.py    ← GPU 시뮬의 미분 계산 부분
   └── forward.py       ← GPU 시뮬의 정방향 적분 부분
```

### 2-2. 무엇을 위해서

전기 모터의 **back-EMF (역기전력)** 와 토크가 서로 영향을 주는 커플링 방정식을 GPU에서 정확하게 풀기 위해서. (Schur complement coupling, Method A / A+ / B 등의 적분 방법, demag 고장 시 Ke 차이 반영 등.)

자세한 수식은 이 문서의 범위를 넘어선다 — 여기서는 **"왜 남의 폴더 파일을 건드렸나"** 만 이해하면 된다.

### 2-3. 왜 우리 프로젝트 폴더에 못 두나

이상적으로는 우리 프로젝트 안에 두고 싶지만 — `mujoco_warp` 는 외부 라이브러리다. `import mujoco_warp` 라고 쓰면 파이썬은 무조건 site-packages를 본다. 그래서 **그 파일이 있는 곳에서 직접 고치는 수밖에 없었다.**

---

## 3. 문제 — pip가 덮어쓴다

site-packages 폴더는 **pip의 영역**이다. 다음 중 하나가 일어나면 pip는 PyPI에서 깨끗한 원본을 받아와서 우리 수정본을 **흔적도 없이 덮어쓴다**:

| 상황 | 무슨 일이 일어나나 |
|---|---|
| `pip install mujoco-warp` | 재설치 → 덮어쓰기 |
| `pip install --upgrade mujoco-warp` | 업그레이드 → 덮어쓰기 |
| `pip install -r requirements.txt` (안에 mujoco-warp 포함) | 덮어쓰기 |
| `conda env create -f environment.yml` | 환경 새로 만들면서 새로 설치 → 덮어쓰기 |
| 다른 사람이 git clone 해서 자기 컴퓨터에 환경 세팅 | 그쪽 site-packages 는 깨끗한 원본 상태 |

덮어쓰기가 일어나면:
- 학습/평가가 **부정확하게** 돌아간다 (cross-Jacobian 항이 빠짐).
- Method B 같은 일부 cfg 는 **에러를 내며 죽는다** (필요한 함수가 없음).
- 가장 무서운 건: **에러 없이 조용히 잘못된 결과를 낸다.**

---

## 4. 해결책 — vendor 폴더에 백업 보관

우리 프로젝트 안에 **수정본의 사본 + 수정 내용의 diff** 를 따로 보관해 두었다:

```
/home/rbdo/unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch/
   ├── _src/
   │    ├── derivative.py        ← 수정이 끝난 derivative.py 사본
   │    └── forward.py           ← 수정이 끝난 forward.py 사본
   └── mujoco_warp_3.6.0_coupling.patch
                                  ← "원본 대비 무엇이 바뀌었나" 의 diff
```

이 폴더는 **우리 프로젝트의 일부**다. git이 관리하므로 pip / conda 와 완전히 무관하고, 우리가 직접 `rm` 하지 않는 한 사라지지 않는다.

추가로, 처음 패치를 적용할 때 원본도 site-packages 안에 백업해 두었다:

```
~/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp/_src/
   ├── derivative.py.original_vanilla    ← 패치 전 원본
   └── forward.py.original_vanilla
```

---

## 5. 패치를 적용하는 방법 — 단계별

상황: 새 환경을 만들었거나, `pip install` 을 했더니 수정이 사라진 것 같다.

### Step 1: site-packages 경로 확인

먼저 `mujoco_warp` 가 어디에 설치돼 있는지 확인:

```bash
python -c "import mujoco_warp, os; print(os.path.dirname(mujoco_warp.__file__))"
```

출력 예:
```
/home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp
```

이 경로 뒤에 `/_src` 를 붙인 곳이 `derivative.py` / `forward.py` 가 있는 폴더다.

### Step 2: 사본을 복사 (가장 쉬운 방법)

```bash
# 두 경로를 변수로 잡아두면 실수가 줄어든다
PATCH=/home/rbdo/unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch
DEST=$(python -c "import mujoco_warp, os; print(os.path.dirname(mujoco_warp.__file__))")/_src

# 덮어쓰기 실행
cp "$PATCH/_src/derivative.py" "$DEST/derivative.py"
cp "$PATCH/_src/forward.py"    "$DEST/forward.py"
```

이게 끝이다. 두 줄이면 복구 완료.

### Step 3: 잘 적용됐는지 확인 (해시 비교)

복사한 두 파일이 사본과 정확히 같은지 `md5sum` 으로 비교:

```bash
md5sum "$PATCH/_src/derivative.py" "$DEST/derivative.py"
md5sum "$PATCH/_src/forward.py"    "$DEST/forward.py"
```

각 쌍의 해시가 **같으면 OK**.

예시 (값은 매번 다를 수 있음):
```
abc123...  /home/rbdo/unitree_rl_mjlab/vendor/.../derivative.py
abc123...  /home/.../site-packages/mujoco_warp/_src/derivative.py
              ↑ 두 줄의 해시가 일치하면 정상
```

### Step 4: 간단 동작 확인

```bash
python -c "import mujoco_warp; print('import OK')"
```

`import OK` 만 뜨면 일단 import 단계에서는 문제 없음.

---

## 6. (대안) patch 파일로 적용하기

`.patch` 파일은 "원본 대비 무엇이 어떻게 바뀌었나" 를 텍스트로 적은 명세다. `patch` 명령으로 적용할 수 있다.

```bash
cd /home/rbdo/miniconda3/envs/mjlab/lib/python3.11/site-packages/mujoco_warp
patch -p1 < /home/rbdo/unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch/mujoco_warp_3.6.0_coupling.patch
```

이 방법은 mujoco_warp 의 **버전이 정확히 3.6.0 일 때**만 안정적이다. 다른 버전에서는 라인 번호가 안 맞아서 거부될 수 있다.

**그래서 일반적으로는 Step 2의 "그냥 복사" 방법을 추천한다.** patch 방법은 "원본 + 수정의 차이" 를 한 눈에 보고 싶거나, 다른 버전에 손으로 옮길 때 참고용으로 쓴다.

---

## 7. 자주 헷갈리는 포인트

### Q1. 그냥 우리 프로젝트 안에 mujoco_warp 폴더를 두면 안 되나?

원리상은 가능하지만, 그러면 PyPI 의 mujoco_warp 와 분리돼서 버전 관리가 꼬인다. 표준적인 방식은 **(1) PyPI 에서 설치하고 (2) 그 위에 우리 변경분만 따로 보관해서 덮어쓰기** 다. 우리가 하는 것도 이 방식.

### Q2. 두 사본 (vendor/ 와 site-packages/) 중 어느 게 "진짜" 실행되나?

**site-packages 안의 파일이 실제로 import 되어 실행된다.** vendor/ 안의 파일은 "백업 / 복원용 사본" 일 뿐이고, 학습/평가 코드는 vendor/ 를 import 하지 않는다.

그래서 vendor/ 만 수정해 두고 site-packages 에 반영하지 않으면 **수정이 적용되지 않은 채로 학습이 돈다.** 항상 둘이 같아야 한다.

### Q3. 수정 내용을 더 바꾸고 싶으면?

1. **site-packages 안의 파일을 직접 편집** 한다 (실제 실행되는 본체).
2. 동작이 확인되면 그 파일을 **vendor/ 로 복사해서 백업** 한다:
   ```bash
   cp "$DEST/derivative.py" "$PATCH/_src/derivative.py"
   cp "$DEST/forward.py"    "$PATCH/_src/forward.py"
   ```
3. patch 파일도 갱신하려면 `diff` 로 다시 만든다:
   ```bash
   diff -urN "$DEST/derivative.py.original_vanilla" "$DEST/derivative.py" >  new.patch
   diff -urN "$DEST/forward.py.original_vanilla"    "$DEST/forward.py"    >> new.patch
   ```

요점: **편집은 site-packages → 백업은 vendor/** 순서.

### Q4. mujoco_warp 버전이 올라가면?

PyPI 에 3.7.0 이 나오고 그게 설치되면, 우리 patch 는 라인이 안 맞아서 거부될 가능성이 높다. 그 경우 새 버전에 변경 사항을 **손으로 다시 옮겨야** 한다. 옮긴 결과를 `vendor/mujoco_warp_3.7.0_patch/` 같은 새 폴더에 보관하면 된다.

지금은 3.6.0 에 고정해 둔 상태가 가장 안전.

### Q5. 다른 사람이 이 프로젝트를 clone 했을 때 무엇을 해야 하나?

```
clone 직후 상태:
  - vendor/mujoco_warp_3.6.0_patch/   ← 같이 받아짐 (git에 포함)
  - site-packages/mujoco_warp/         ← 아직 깨끗한 원본 (PyPI에서 받은 그대로)
```

그래서 clone 한 다음에는 반드시 **Step 2 (사본 복사)** 를 한 번 실행해야 한다. 그렇지 않으면 우리 수정이 적용되지 않은 채로 코드가 돈다.

---

## 8. 한 번에 보는 체크리스트

새 환경을 세팅할 때 순서:

```bash
# (1) conda 환경 만들고 활성화
conda create -n mjlab python=3.11
conda activate mjlab

# (2) 의존성 설치 (mujoco_warp 도 이 단계에서 PyPI 원본이 설치됨)
cd /home/rbdo/unitree_rl_mjlab
pip install -e .

# (3) ★ 우리 GPU 패치 적용 (이걸 빼먹으면 학습이 부정확하게 돈다)
PATCH=/home/rbdo/unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch
DEST=$(python -c "import mujoco_warp, os; print(os.path.dirname(mujoco_warp.__file__))")/_src
cp "$PATCH/_src/derivative.py" "$DEST/derivative.py"
cp "$PATCH/_src/forward.py"    "$DEST/forward.py"

# (4) 검증
md5sum "$PATCH/_src/derivative.py" "$DEST/derivative.py"
md5sum "$PATCH/_src/forward.py"    "$DEST/forward.py"
python -c "import mujoco_warp; print('OK')"
```

`pip install` 류의 명령을 다시 칠 일이 생기면, 그때마다 **(3) 단계를 다시 실행**한다고 생각하면 된다.

---

## 9. 더 깊이 알고 싶으면

이 문서는 "구조와 원리"에만 집중했다. 다음 문서들에 더 자세한 내용이 있다:

- `docs/code_modifications_explained.md` — 같은 주제의 좀 더 빠른 정리 (C 콜백 파일까지 포함)
- `vendor/mujoco_warp_3.6.0_patch/mujoco_warp_3.6.0_coupling.patch` — 실제 수정 내용 (diff 형식)
- `src/assets/robots/unitree_go2/mj_native_electric_actuator.py` — 이 패치를 호출하는 파이썬 쪽 코드 (특히 `use_coupled=True` 경로)
- `src/assets/robots/unitree_go2/coupled_ode_solver.py` — 커플드 ODE 솔버 본체
