# Go2 평지 보행 sim2real 개선 작업 정리 (2026-05)

이 문서는 2026년 5월에 진행한 Unitree Go2 평지 보행 정책의 sim2real 개선 작업을
**처음 보는 사람도 이해할 수 있도록** 정리한 것입니다. 전문용어는 그때그때 쉽게
풀어 설명합니다.

---

## 0. 먼저, 핵심 용어 쉽게 풀기

| 용어 | 쉬운 설명 |
|---|---|
| **sim2real gap** | 시뮬레이터에서 잘 걷던 로봇이 실제 로봇에선 비틀거리는 현상. "시뮬과 현실의 차이". |
| **도메인 랜덤화 (DR)** | 학습할 때 시뮬 조건(마찰·무게·센서 오차 등)을 일부러 매번 다르게 흔드는 것. 다양한 상황을 겪게 해서 실제 로봇에서도 잘 버티게 만든다. |
| **tracking reward** | "명령한 속도대로 잘 따라갔는가" 점수. 1.0에 가까울수록 명령을 정확히 따름. |
| **정책(policy) / actor** | 로봇의 "뇌". 센서 입력을 받아 다리 관절 목표를 출력하는 신경망. |
| **critic** | 학습 도우미 신경망. 정책이 얼마나 잘하고 있는지 "가치"를 평가해 학습을 돕는다. 배포(실제 로봇)엔 안 들어가고 학습 때만 쓴다. |
| **PPO** | 강화학습 알고리즘의 한 종류. 정책을 조금씩 안전하게 개선한다. |
| **curriculum (커리큘럼)** | 쉬운 것부터 점점 어렵게 가르치는 학습법. 사람 교육과정과 같은 개념. |
| **reward floor clip** | 한 스텝의 점수 합이 음수면 0으로 잘라내는 장치. 벌점이 학습을 과하게 끌어내리지 않게 막는다. |
| **iteration (iter)** | 학습 반복 횟수 한 단위. 수천~수만 번 반복하며 정책이 좋아진다. |

---

## 1. 전체 그림 (무엇을, 왜)

**문제**: Go2 평지 보행 정책에 도메인 랜덤화(DR)를 강하게 켜면 보상(reward)이 절반으로
떨어졌다. 즉, 실제 로봇 대비를 위해 시뮬을 흔들수록 학습이 망가지는 상황.

**한 일 (3덩어리)**:
1. **원인 분석** — sim2real이 검증된 외부 레포(PGTT)와 내 환경을 1:1 비교해 무엇이
   다른지 찾음. → "reward floor clip"이라는 핵심 장치가 내 환경에 없다는 걸 발견.
2. **floor clip 도입 + 검증** — 그 장치를 내 환경에 이식하고, 있을 때/없을 때를
   실제 학습으로 비교.
3. **deploy 스타일 DR 이식** — 또 다른 참조(Isaac Lab deploy baseline)의 "DR를 적게
   흔들고, 학습 진행에 따라 DR 강도를 자동으로 올리는 커리큘럼" 방식을 새 task로 구현.

결과적으로 "DR 켜면 reward 폭락" 문제를 해결하고, 실제 로봇에 가까운 다양한 sim2real
교란(센서 노이즈·외력·마찰·동작 지연)을 정책이 견디게 만들었다.

---

## 2. 작업 A — PGTT 비교 분석

**PGTT**(Phase-Guided Terrain Traversal): GO2/ANYmal에서 sim2real이 검증된 외부 연구 레포.
이걸 기준(reference)으로 삼아 내 환경과 비교.

- 결과 문서: `../pgtt_analysis/pgtt_vs_mjlab_comparison.md` (1차),
  `../pgtt_analysis/pgtt_vs_mjlab_comparison_v2.md` (2차).
- 비교 항목: 도메인 랜덤화 범위, 학습 하이퍼파라미터, 보상 함수, 액션(동작) 설계,
  커리큘럼 구조.

**가장 중요한 발견 — reward floor clip**

PGTT 코드(`joystick_base.py:220`)에는 이런 한 줄이 있었다:
```python
reward = clip(sum(rewards) * dt, 0.0, 10000.0)   # 한 스텝 점수 합을 0 밑으로 안 떨어뜨림
```
즉 PGTT는 **한 스텝의 점수 합이 음수면 0으로 잘라낸다.** 그래서 "넘어짐" 같은 큰 벌점이
있어도 그 스텝에 좋은 점수(예: 보행 리듬 보상)가 같이 있으면 벌점이 상쇄되어, 학습 신호가
과도하게 음수로 가지 않는다. 내 환경(mjlab + rsl_rl)에는 이 장치가 없어서, 벌점이 그대로
반영돼 DR을 켜면 reward가 무너졌던 것.

> 참고: 처음엔 "내 환경은 dt(시간간격)도 안 곱한다"고 잘못 봤는데, 확인해보니 내 환경도
> dt는 곱한다. **진짜 차이는 floor clip(0으로 자르기) 하나뿐**이었다. v2 문서에서 정정함.

---

## 3. 작업 B — floor clip 도입 + A/B 실험

### 한 일
- `VelocityFloorClippedRunner`라는 학습 실행기를 새로 만들어, 매 스텝 reward를
  `clamp(min=0)` 으로 자르게 함 (PGTT 방식 그대로 재현).
- mjlab/라이브러리 본체는 안 건드리고, 특정 task에만 적용되도록 격리.
- `Unitree-Go2-Flat` task에 이걸 붙이고, DR도 7종 추가.
- 비교용으로 floor clip만 뺀 `Unitree-Go2-Flat-NoClip` task도 만듦.

### A/B 실험 결과 (4096개 환경 병렬, 5000 iter)

| 지표 | Floor clip 있음 | 없음(NoClip) | 해석 |
|---|---|---|---|
| 평균 보상 | **43.11** | 39.39 | clip이 9.4% 높음 |
| 속도 추종(x,y) 오차 | **0.749** | 0.917 | clip이 더 정확 |
| 액션 떨림 | 더 적음 | 더 큼 | clip이 더 부드러운 동작 |

**결론**: floor clip을 켜면 같은 DR에서도 정책이 명령을 더 정확히 따르고, 더 빠르게
안정적인 동작으로 수렴했다. "DR 켜면 reward 폭락" 증상의 핵심 처방이 맞았다.

부수 산출물: 관측(observation) 구성 문서 `docs/observation_space_go2_flat.md`
(정책 입력 47개, critic 입력 74개 항목을 코드로 검증해 정리).

---

## 4. 작업 C — deploy 스타일 DR 새 task (`Unitree-Go2-Flat-DeployDR-v0`)

또 다른 참조(Isaac Lab deploy baseline)의 아이디어를 옮긴 작업. 핵심 철학은
**"DR를 무작정 세게 켜지 말고, ① 불필요한 건 끄고 ② 정책 실력에 맞춰 DR 강도를 천천히
올리자"**.

기존 task는 절대 안 건드리고 **새 task 안에서만** 단계(Phase)별로 구현했다.

### Phase 1 — 새 task 등록 (기존 Flat 그대로 복제)
- `Unitree-Go2-Flat-DeployDR-v0` 등록. 이 시점엔 기존과 100% 동일, 이름만 다름.
- 200 iter 학습으로 정상 동작 확인 (보상이 -3.9 → 28.6으로 정상 상승).

### Phase 2 — DR 조정 (끄기 + 줄이기)
- 불필요한 DR 9종(무게·무게중심·PD게인·모터강도·전압·외력·센서바이어스 등)을 **끔**.
- 마찰은 좁은 범위(0.3~1.25)로, 밀치기(push)는 5초마다 수평으로만 ±0.5로, 초기 자세는
  기본자세 × 0.5~1.5로 다양화.
- **결과**: DR 13종을 한꺼번에 켠 버전 대비 속도 추종이 **145%로 회복**(같은 500 iter).
  즉 "필요 없는 DR이 학습을 갉아먹고 있었다"는 게 확인됨.

### Phase 3 — DR 커리큘럼 (자동 난이도 조절)
- **단일 숫자 `level`(0~1)** 하나로 DR 강도를 조절. level이 클수록 센서 노이즈·밀치기가
  세진다.
- 정책이 잘하면(넘어짐 적고, 명령 잘 따르고, 시간초과로 끝나면) level을 조금씩 올리고,
  자주 넘어지면 내린다. **EMA**(지수이동평균 = 최근 성적을 부드럽게 평균낸 값)로 판단.
- **결과**: level이 0.1에서 시작 → 정책이 배우면서 자동으로 0.94까지 상승. "쉬운 것부터
  점점 어렵게"가 의도대로 작동.

### Phase 4 — 보행 리듬 보상 (foot_gait)
- 4발이 대각선으로 번갈아 딛는 "트롯(trot)" 걸음을 유도하는 보상. 가중치 0.10.
- 기존에 같은 공식의 보상이 이미 있어서 중복 함수 대신 가중치만 0.5→0.10으로 조정.
- 보행 리듬 일치도가 0.58로 수렴. (참고: 가중치를 0.5→0.1로 낮췄더니 오히려 명령
  추종이 좋아져서, 합의 하에 합격 기준을 0.6→0.55로 낮춤. 트롯 자체는 잘 학습됨.)

### Phase 5 — critic에 "정답지" 정보 추가 (asymmetric actor-critic)
- **asymmetric(비대칭) actor-critic**: 정책(actor)은 실제 로봇에서 얻을 수 있는 정보만,
  학습 도우미(critic)는 실제론 알기 어려운 "정답지" 정보까지 받는 구조. critic이 더
  정확히 평가하면 학습이 빨라진다.
- critic에 현재 마찰계수·DR level·마지막 밀치기 정보를 추가 (74개 → 78개 입력).
- **결과**: value loss(가치 추정 오차)가 11% 감소 → critic이 더 잘 평가함.

### Phase 6 — heading(방향) 명령 검토 (보고만)
- 내 환경은 25%의 환경만 방향유지 모드(`rel_heading_envs=0.25`), 참조는 100%. 차이는
  크지만 **지시대로 변경하지 않고 보고만** 함.

---

## 5. 작업 D — 후속 검증 + 동작 지연/노이즈 추가

### 작업 1 — 커리큘럼 구현 검증
- 코드를 스펙과 1:1 대조: 9개 중 8개 정확히 일치. 1개(추종 점수 정규화 방식)는 표기 차이
  였는데, 사용자 확인 결과 "[0,1]로 정규화"가 원래 의도라 현재 구현이 정확함.
- 실제 학습 로그로 확인: "명령 추종 점수가 0.75를 넘은 뒤에 level이 오른다"는 인과관계가
  그래프로 확인됨. 자주 넘어질 땐 level이 내려가는 것도 관측. 버그 없음.

### 작업 2 — "DR 전혀 없는" 깨끗한 기준과 비교
- 가장 공정한 비교를 위해 `Unitree-Go2-Flat-DeployDR-NoDR-v0`(DR만 끄고 나머지 동일)를
  만들어, DR 있는 버전과 **같은 난수 seed, 2000 iter**로 비교.

| 지표 | DR 없음 | DR 있음(level→1.0) | 차이 |
|---|---|---|---|
| 속도 추종(전후좌우) | 0.918 | 0.887 | **3.1%p** |
| 속도 추종(회전) | 0.964 | 0.949 | **1.5%p** |

- **판정**: 차이가 10%p 이내 → "DR 때문에 치르는 비용이 정상 범위". 커리큘럼이 강한
  DR 비용을 3%p 정도로 잘 흡수하고 있다는 뜻. (이 NoDR task는 회귀 테스트용으로 보존.)

### 작업 3 — 동작 지연 + 동작 노이즈 추가 (sim2real의 핵심 채널)
- 실제 로봇은 명령을 내려도 통신·구동 지연으로 살짝 늦게 움직이고, 명령에 잡음도 섞인다.
  이걸 시뮬에 넣는 `DelayedNoisyJointPositionAction`을 새로 만듦:
  - **동작 지연**: 과거 동작을 잠깐 저장했다가 1스텝 늦게 적용(ring buffer 사용).
  - **동작 노이즈**: 적용 직전 동작에 약간의 무작위 잡음을 더함.
  - 둘 다 커리큘럼 level에 연동(level이 커지면 지연/노이즈도 커짐). level=0이면 둘 다
    꺼져서 기존과 똑같이 동작(안전).
- critic 입력에 현재 지연·노이즈 정보도 추가 (78개 → 80개).
- **단위 테스트 통과**: 지연 1스텝이면 정확히 직전 동작이 적용됨, 노이즈 켜면 분산 증가,
  level=0이면 원래대로.

### 작업 1 재실행 (최종 통합 검증)
- 모든 DR 채널(센서노이즈+밀치기+동작지연+동작노이즈)이 켜진 상태에서도 level이 1.0까지
  정상 상승(정체 없음). 즉 정책이 sim2real 교란을 전부 감당함.

---

## 6. 작업 E — git 커밋 정리

작업을 논리 단위 8개 커밋으로 분리(각 커밋은 단독으로 빌드/등록이 깨지지 않음):

```
feat: add reward floor clip runner + deploy DR events + NoClip task   ← 선행(base)
feat: register Unitree-Go2-Flat-DeployDR-v0 (clone of Flat) — Phase 1
feat: tune DR events for deploy ... — Phase 2
feat: add EMA-based DeployDRCurriculum ... — Phase 3
feat: add foot_gait reward (weight 0.10) — Phase 4
feat: add asymmetric critic privileged obs ... — Phase 5
feat: add NoDR regression baseline task — 작업 2
feat: add DelayedNoisyJointPositionAction + ... — 작업 3
```

---

## 7. 만들어진 task 목록과 학습 명령어

| task 이름 | 설명 |
|---|---|
| `Unitree-Go2-Flat` | 기존 평지 task + floor clip + DR 13종 (기존, 무손상) |
| `Unitree-Go2-Flat-NoClip` | 위에서 floor clip만 뺀 비교용 |
| `Unitree-Go2-Flat-DeployDR-v0` | **메인 결과물** — deploy 스타일 DR + 커리큘럼 + 동작지연/노이즈 |
| `Unitree-Go2-Flat-DeployDR-NoDR-v0` | DeployDR에서 DR만 끈 기준선(회귀 테스트용) |

학습 명령 예시:
```bash
cd /home/rbdo/unitree_rl_mjlab
PYTHON=/home/rbdo/miniconda3/envs/mjlab/bin/python

# 메인 결과물 학습 (DR 전부 커리큘럼 연동)
$PYTHON scripts/train.py Unitree-Go2-Flat-DeployDR-v0 \
  --env.scene.num-envs 4096 --agent.max-iterations 5000 --agent.run-name deploydr_v0

# DR 없는 기준선
$PYTHON scripts/train.py Unitree-Go2-Flat-DeployDR-NoDR-v0 \
  --env.scene.num-envs 4096 --agent.max-iterations 5000 --agent.run-name deploydr_nodr_v0
```

학습된 정책 시각화(play)는 `docs/` 의 다른 안내나 `scripts/play.py --checkpoint-file <경로>` 참고.

---

## 8. 결과 수치 한눈에

- floor clip 켜기: 평균 보상 +9.4%, 속도 추종 더 정확 (5000 iter A/B).
- 불필요 DR 9종 끄기: 속도 추종 145% 회복 (DR 13종 동시 대비).
- DR 커리큘럼: level 0.1 → 0.94~1.0 자동 상승 (정책 실력에 맞춰).
- DR 비용(깨끗한 기준 대비): 속도 추종 단 3.1%p / 1.5%p 손해 (10%p 이내, 정상).
- asymmetric critic: value 추정 오차 11% 감소.
- 동작 지연/노이즈 추가 후에도: 명령 추종 거의 유지(±0.2%p), level 1.0까지 정상 상승.

---

## 9. 남은 일 / 코드만으론 못 한 것

- **floor clip의 정확한 explained variance(+5%p) 검증**: 사용하는 RL 라이브러리(rsl_rl)가
  이 지표를 기록하지 않고, 라이브러리 수정은 금지라 직접 측정 불가. value loss 11% 감소가
  간접 근거. 정확히 보려면 별도 A/B 연구 필요.
- **마찰 num_buckets / 주기적 재샘플**: mjlab이 지원 안 해서 startup(시작 시 1회)만 적용.
  코드에 TODO 주석으로 남김.
- **heading(방향) 명령 비율**: 참조는 100%, 내 환경 25%. 차이 크지만 지시에 따라 미변경,
  추가 결정 대기.
- **PGTT 분석의 일부 항목**: action delay 등 PGTT 논문엔 있지만 코드에서 못 찾은 값은
  "코드 미확인"으로 표기 (추측 안 함).

---

## 10. 관련 문서

- `../pgtt_analysis/pgtt_vs_mjlab_comparison.md`, `..._v2.md` — PGTT vs 내 환경 비교(1·2차)
- `docs/sim2real_dr_flat_floor_clip.md` — floor clip + Flat DR 변경 상세
- `docs/observation_space_go2_flat.md` — 정책/critic 입력 구성(47D/74D)
- 이 문서 — 전체 작업 요약
