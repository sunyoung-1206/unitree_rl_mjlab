# 백업 작업 기록 (2026-04-27)

## 1. 목적
`experiment/fault-coupled-ode` 작업 도중 GitHub에 스냅샷을 남기되,
**워킹트리의 `M` / `??` 표시는 그대로 유지**(추후 정리 후 재백업 예정)
하는 백업을 수행.

## 2. 작업 시점 환경
- 원격: `git@github.com:sunyoung-1206/unitree_rl_mjlab.git`
- 현재 브랜치: `experiment/fault-coupled-ode` (origin에 push 안 됨)
- HEAD: `d6871fd` — `origin/main`과 동일 커밋 (`git log origin/main..HEAD` 비어 있음)
- 따라서 GitHub와의 모든 차이는 **워킹트리 안**에만 존재

### 차이 분포
| 분류 | 개수/용량 | 비고 |
|---|---|---|
| 수정 파일 (M) | 7개 | scripts/, src/assets/, src/tasks/ 하위 |
| untracked 디렉토리 | `docs/` 36K, `results/` 355M, `solver_comparison/` 48M, `.claude/` 12K | results/가 백업 용량의 88% |
| untracked notebooks | 3개 | method_a_gpu_flow / solver_flow_explained / v2 |
| untracked scripts | 23개, 약 280K | analyze_*, eval_*, phase*, plot_* 등 |
| **총 백업 대상** | **약 404MB** | |
| .gitignore 자동 제외 | logs/ 2.9G, wandb/ 643M, .claude/settings.local.json | 푸시 대상 아님 |
| 100MB 초과 단일 파일 | 없음 | 최대 ~9.7MB (mp4) |

## 3. 백업 방법: 분리 index를 통한 무영향 push
`.git/index` 와 워킹트리를 건드리지 않고 origin에만 새 브랜치를 추가하기 위해
임시 index 파일을 분리해서 사용.

```bash
TMP_INDEX=$(mktemp /tmp/backup_index.XXXXXX)
export GIT_INDEX_FILE="$TMP_INDEX"

git read-tree HEAD                                            # 임시 index에 HEAD 트리 로드
git add -A                                                    # 워킹트리 변경분 staging (gitignore 자동 적용)
TREE=$(git write-tree)
PARENT=$(git rev-parse HEAD)
COMMIT=$(git commit-tree "$TREE" -p "$PARENT" -m "snapshot ...")
git push origin "${COMMIT}:refs/heads/backup/<name>"          # 원격 ref만 직접 업데이트

unset GIT_INDEX_FILE
rm -f "$TMP_INDEX"
```

핵심 효과:
- 실제 `.git/index` — 미수정
- 워킹트리 파일 — 미수정
- 로컬 브랜치/HEAD — 미수정 (로컬에 `backup/...` 브랜치도 안 만들어짐)
- 원격에만 새 브랜치 추가됨
- 푸시 직후 `git status` 출력 100% 동일

## 4. 1차 백업
- 브랜치: `backup/snapshot-2026-04-27`
- 커밋: `e97b4ce1`
- 변경: 451 files, +53,314 / −37 (origin/main 대비)
- 범위: `/home/rbdo/unitree_rl_mjlab/` 워킹트리 전체

## 5. 발견: 레포 외부 mjwarp 수정
1차 백업 후 추가 위험 발견.

### 상황
- `mujoco-warp` 3.6.0이 conda env `mjlab`의 site-packages에 **PyPI 휠로 정식 설치** (editable 아님)
- 그 안에서 두 파일을 직접 편집한 상태 (mtime 2026-04-27):

| 파일 | 수정 내용 |
|---|---|
| `_src/forward.py` (1103L) | L151 vanilla filterexact 분기, **L683 "Coupled filterexact: standard filter + Ke mismatch correction"** |
| `_src/derivative.py` (320L) | L70~95 `schur = -one_minus_beta * Kt_gr * Ke_gr / R_val` 항을 `vel`에 더함 |

= memory의 *Cross-Jacobian / coupled filterexact* 구현체.

### 위험
이 수정은 git 추적 밖이라 다음 중 어느 거든 일어나면 **즉시 소실**:
- `pip install --upgrade mujoco-warp`
- `pip install --force-reinstall mujoco-warp`
- conda env 재생성
- 새 머신 셋업

1차 백업 브랜치는 `unitree_rl_mjlab/`만 담았기에 이 수정은 포함하지 못함.

## 6. 2차 백업: vendor 패치 추가
응급 보호로 두 파일을 레포 안에 보관 + 원본 대비 diff 패치 생성.

### 추가된 파일
```
unitree_rl_mjlab/vendor/mujoco_warp_3.6.0_patch/
├── _src/
│   ├── forward.py            (33KB, 수정본 전체)
│   └── derivative.py         (9.5KB, 수정본 전체)
└── mujoco_warp_3.6.0_coupling.patch   (5.9KB, unified diff)
```

### 패치 통계
- 9 hunks, +59 / −4 lines
- 원본은 `pip download mujoco-warp==3.6.0 --no-deps`로 확보
- `patch --dry-run -p1` 통과 검증 완료

### 추가 백업 브랜치
- 브랜치: `backup/snapshot-2026-04-27-mjwarp`
- 커밋: `f3be7312`
- 범위: 1차 백업 내용 + `vendor/mujoco_warp_3.6.0_patch/`

## 7. 원격 브랜치 현황
| 브랜치 | 커밋 | 상태 |
|---|---|---|
| `backup/snapshot-2026-04-27` | `e97b4ce1` | mjwarp 패치 미포함 |
| `backup/snapshot-2026-04-27-mjwarp` | `f3be7312` | **완전판 (권장)** |

## 8. 새 머신/새 env에서 복원

```bash
# 1. 레포 클론 + 백업 브랜치 체크아웃
git clone git@github.com:sunyoung-1206/unitree_rl_mjlab.git
cd unitree_rl_mjlab
git checkout backup/snapshot-2026-04-27-mjwarp

# 2. (mjwarp 환경 셋업 — pip install mujoco-warp==3.6.0 등)

# 3. mjwarp 패치 적용 (둘 중 택1)
PKG=$(python -c "import mujoco_warp, os; print(os.path.dirname(mujoco_warp.__file__))")

# (a) 패치 적용
cd "$PKG" && patch -p1 < <repo>/vendor/mujoco_warp_3.6.0_patch/mujoco_warp_3.6.0_coupling.patch

# (b) 또는 파일 통째로 덮어쓰기
cp -r <repo>/vendor/mujoco_warp_3.6.0_patch/_src/* "$PKG/_src/"
```

## 9. 알려진 한계
- 패치는 **mujoco-warp 3.6.0 휠에 묶여 있음**. 새 버전으로 올리면 hunk가 깨질 수 있음.
- 장기적 안정화 옵션:
  1. `mujoco_warp` 소스 클론 + `pip install -e .` → 수정을 그 git에 commit (env 재생성 안전)
  2. `sunyoung-1206/mujoco_warp` 포크 → `unitree_rl_mjlab/setup.py`가 fork commit hash를 참조하게 변경 (새 머신에도 자동 재현)

## 10. 작업 후 워킹트리 상태
- 시작 시점의 `M` 7개 + 모든 `??` 그대로 유지
- 신규 untracked: `vendor/` 디렉토리 하나만 추가됨 (의도된 결과)
- `.git/index` / HEAD / 로컬 브랜치 모두 미변동
