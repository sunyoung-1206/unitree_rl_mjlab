# Action delay jitter 대응 — 구현 핸드오프 (Claude Code용)

> 작성: 2026-05-21 · 대상: 다음 세션의 Claude Code(또는 사람) · 상태: **미구현 / 설계 확정 전**
> 목적: 현재 "고정 1-step 지연"을 **간헐적·확률적 지연(jitter)** 으로 확장하는 작업의 배경·방법·체크리스트를 한 곳에.

---

## 0. TL;DR

현재 action delay 는 커리큘럼이 정한 값으로 **모든 step·모든 env 가 항상 동일하게** 지연된다(결정론적 상수).
실제 로봇처럼 "정상 → 지연 → 정상 → 지연"이 무작위로 반복되는 **jitter** 는 학습 분포에 없다.
→ 정책이 *일정한* 지연엔 강해도 *변동* 자체엔 노출된 적이 없다.

대응 3안: **(a) step·env별 i.i.d. 랜덤 지연**, **(b) 시간상관(Markov/hold) 지연**, **(c) 연속(sub-step) 보간 지연**.
권장: **(a) 먼저** (ring buffer 가 이미 per-env 인덱싱을 지원해 변경 최소) → gap 남으면 **(b)**.

핵심 공통 변경: `_delay_steps` 를 매 step 갱신되는 `[num_envs]` 텐서로 운용하고, **critic 의 지연 관찰을 스칼라 → per-env 텐서**로 바꿔 *그 순간 실제 지연*을 넣어야 한다(지금은 스칼라라 jitter 와 불일치).

---

## 1. 현재 구현 (출발점)

### 코드 위치
- 액션 텀: `src/tasks/velocity/mdp/actions.py` — `DelayedNoisyJointPositionAction`, `DelayedNoisyJointPositionActionCfg`
- 커리큘럼: `src/tasks/velocity/mdp/curriculums.py` — `DeployDRCurriculum._apply_level` (≈ L194–221)
- critic obs: `src/tasks/velocity/mdp/observations.py:91` `deploy_delay_steps`, `:97` `deploy_action_noise_std`
- 태스크 배선: `src/tasks/velocity/config/go2/__init__.py` (≈ L250–269) — `delay_max_steps=1`, `noise_std_max=0.1`, critic 78D→80D

### 동작 (요약)
- ring buffer `_action_buf` shape `[num_envs, delay_max_steps+1, action_dim]`, index 0 = 최신 raw action.
- 적용 순서: **delay(과거 raw 꺼냄) → noise(가우시안) → PD 변환(`*scale + offset`)**.
- 지연 선택: `delayed = _action_buf[_env_arange, _delay_steps]`.
  - **이미 per-env fancy indexing** 이다. `_delay_steps` 는 `[num_envs]` long 텐서.
  - 현재는 `set_delay_steps(k)` 가 `_delay_steps.fill_(k)` 로 **모든 env 를 동일 상수**로 채울 뿐.
- 커리큘럼 종속(`_apply_level`): `delay_steps = round(delay_max_steps * level)`. `delay_max_steps=1`, `step_dt=0.02s` 이므로
  - level < 0.5 → 0 (지연 없음), level ≥ 0.5 → 1 (= **20 ms** 지연). 계단식.
- `last_action` 관찰은 `_raw_actions`(노이즈·지연 없는 깨끗한 정책 출력)를 본다 → 정책은 지연을 직접 관찰하지 못함.
- critic 만 `_deploy_delay_steps`(현재 **스칼라 float**, 모든 env 공통)를 privileged 로 본다.
- reset: `_action_buf[idx] = 0.0` (raw 0 = default pose → 첫 step 튐 없음).

### 한계 (이 작업의 동기)
1. **시간 불변**: 한 에피소드 동안 지연이 0 또는 1 로 고정 → jitter 없음.
2. **env 동일**: 모든 env 가 같은 지연 → 분포 다양성 부족.
3. **상한 1 step**: 20 ms 격자만, 그 이하/이상·변동 표현 불가.
4. **critic 불일치 위험**: 지연을 per-env/per-step 으로 만들면 스칼라 critic obs 가 *실제 지연*과 어긋난다.

---

## 2. 문제 정의 — 간헐적·확률적 지연 (jitter)

실로봇 통신/구동 지연 `d_t` 는 상수가 아니라 시간에 따라 변하는 확률과정이다. 모델링 목표:

- 에피소드/스텝에 따라 `d_t ∈ {0,1,…,D_max}` (단위: control step, 1 step = `step_dt` = 20 ms)가 변동.
- 커리큘럼 `level ∈ [0,1]` 에 종속: level↑ → 변동 폭/최대 지연/지연 확률↑ (level=0 이면 지연 0, 기존과 동일).
- 학습 안정성: 분포는 무작위지만 reset 직후 튐이 없어야 하고, critic 은 *그 순간 실제 지연*을 정확히 알아야 한다(비대칭 구조 유지).

---

## 3. 대응 방법

### (a) step·env별 i.i.d. 랜덤 지연 — **권장 1순위**

**모델.** 매 step 각 env 독립으로 `d_t ~ Uniform{0,…,D_cur}`, 여기서 `D_cur = round(D_max * level)`.

**코드 스케치** (`process_actions` 내부, delay 선택 직전):
```python
# 매 step 새로 샘플 (env별 독립). D_cur 는 curriculum 이 set 한 현재 최대 지연.
if self._cur_max_delay > 0:
    self._delay_steps = torch.randint(
        0, self._cur_max_delay + 1, (self.num_envs,),
        device=self.device, dtype=torch.long,
    )
else:
    self._delay_steps.zero_()
delayed = self._action_buf[self._env_arange, self._delay_steps]
```
- 커리큘럼은 `set_delay_steps(k)`(고정) 대신 `set_max_delay(D_cur)` 를 호출하도록 변경.
- `delay_max_steps` 를 2~3 으로 키우고 buffer 도 `D_max+1` 로 (이미 `+1` 구조라 cfg 값만 키우면 됨).

**장점**: ring buffer 가 이미 per-env 인덱싱 → **변경 최소**. 다양성 즉시 확보.
**단점**: 시간 상관(자기상관) 없음 → 진짜 통신 버스트보다 *white jitter* 에 가까움.
**비용**: 소 (액션 텀 + 커리큘럼 setter + critic obs per-env 화).

### (b) 시간상관(Markov / hold) 지연 — gap 남을 때 2순위

**모델.** 지연 상태가 step 간 유지되다가 확률 `p_change` 로만 바뀐다(덩어리진 정상/지연 구간):
```
flip_t ~ Bernoulli(p_change)
d_t = new ~ Uniform{0,…,D_cur}  if flip_t else d_{t-1}
```

**코드 스케치**:
```python
flip = torch.rand(self.num_envs, device=self.device) < self._p_change
new  = torch.randint(0, self._cur_max_delay + 1, (self.num_envs,), device=self.device)
self._delay_steps = torch.where(flip, new, self._delay_steps)
# reset 시 _delay_steps[env_ids] 도 재초기화 필요.
```
- `p_change`, `D_cur` 모두 level 종속 가능. `p_change` 작을수록 지연 구간이 길고 현실적.

**장점**: "정상 ↔ 지연" 이 구간으로 반복 → 질문 시나리오에 가장 부합.
**단점**: 상태 보존 → reset 처리/critic 동기화 주의. 하이퍼파라미터(`p_change`) 1개 추가.
**비용**: 중.

### (c) 연속(sub-step) 보간 지연 — 미세 지연이 중요할 때

**모델.** 정수 step 격자(20 ms) 보다 고운 지연 `τ = (k+α)·step_dt`, `α∈[0,1)` 를 두 인덱스 선형보간:
```python
i  = self._delay_int            # [num_envs] long
a  = self._delay_frac.unsqueeze(-1)  # [num_envs,1] in [0,1)
lo = self._action_buf[self._env_arange, i]
hi = self._action_buf[self._env_arange, torch.clamp(i + 1, max=self.delay_max_steps)]
delayed = (1 - a) * lo + a * hi
```
- jitter 를 `α` 에 주면 5 ms 급 미세 변동도 표현. `i`, `α` 모두 (a)/(b) 방식으로 무작위화 가능.

**장점**: 가장 표현력 높음(연속 latency).
**단점**: 구현 복잡 + 보간이 raw action 을 평균내 약간의 저역통과 효과(주의). 보통 (a)/(b) 로 충분.
**비용**: 중~대.

---

## 4. 공통 변경 사항 (어느 방법이든 필요)

1. **buffer 크기**: `delay_max_steps` 를 1 → 2~3 으로. `_action_buf` 는 `[N, delay_max_steps+1, A]` 이미 +1 구조라 cfg 만 변경.
2. **curriculum API 변경**: `set_delay_steps(k)`(고정 fill) → `set_max_delay(D_cur)`(상한만 set, 실제 표집은 액션 텀이 매 step). `_apply_level` 에서 `D_cur = round(delay_max_steps * level)` 계산해 전달.
3. **critic obs per-env 화 (중요)**: 현재 `_deploy_delay_steps` 는 스칼라(`curriculums.py:220`)이고 `observations.py:91` 이 그걸 read.
   jitter 도입 시 **액션 텀이 매 step 자신이 적용한 `_delay_steps`(`[num_envs]`)를 env 에 노출**하고,
   `deploy_delay_steps` obs 가 per-env 텐서를 반환하도록 바꿔야 critic 이 *그 순간 실제 지연*을 본다.
   (안 바꾸면 critic 이 평균/상한만 보고 실제와 어긋나 가치추정 악화.)
   - 제안: `env._deploy_delay_steps` 를 `[num_envs]` 텐서로 승격, 액션 텀 `process_actions` 끝에서 write.
   - normalize 고려: raw step 수 대신 `d_t / delay_max_steps` 로 [0,1] 정규화 권장.
4. **reset 처리**: (b)/(c) 처럼 상태가 보존되면 `reset(env_ids)` 에서 `_delay_steps`/`_delay_frac` 도 초기화. buffer 0 초기화는 유지.
5. **level=0 안전**: `D_cur=0` 이면 항상 0 지연 → 기존 동작과 비트 동일해야 함(회귀 보장).

---

## 5. 권장 경로

1. **(a) 구현** + `delay_max_steps=2`(40 ms 상한) + critic per-env 화. 단위테스트·짧은 학습(예: 1500 it)으로 회귀/상승 확인.
2. sim2real 평가에서 jitter gap 이 남으면 **(b) 시간상관**으로 승급(`p_change` 도입).
3. (c) 는 미세 latency 가 문제로 드러날 때만.

> (a)→(b) 는 동일 인터페이스(`set_max_delay`, per-env `_delay_steps`) 위에서 표집 규칙만 교체하므로 점진 확장이 쉽다.

---

## 6. 구현 체크리스트 (파일별)

- [ ] `src/tasks/velocity/mdp/actions.py`
  - [ ] `process_actions` 에 매 step 지연 표집 로직((a) 또는 (b)) 추가, delay→noise→PD 순서 유지.
  - [ ] `set_delay_steps` → `set_max_delay(self, d: int)` 로 교체(또는 둘 다 두고 deprecate). `_cur_max_delay` 보관.
  - [ ] `process_actions` 끝에서 `env._deploy_delay_steps = self._delay_steps`(per-env 텐서) write.
  - [ ] (b)/(c) 면 `reset` 에서 상태 초기화.
- [ ] `src/tasks/velocity/mdp/curriculums.py`
  - [ ] `_apply_level` 에서 `D_cur = round(delay_max_steps * level)` → `action_term.set_max_delay(D_cur)`.
  - [ ] (b) 면 `p_change` 도 level 종속으로 set.
  - [ ] `_deploy_delay_steps` 스칼라 노출 제거(액션 텀이 per-env 로 대체).
- [ ] `src/tasks/velocity/mdp/observations.py`
  - [ ] `deploy_delay_steps` 를 per-env 텐서 반환 + 정규화(`/delay_max_steps`)로 변경.
- [ ] `src/tasks/velocity/config/go2/__init__.py`
  - [ ] `DelayedNoisyJointPositionActionCfg(delay_max_steps=2, ...)` 로 상향.
  - [ ] critic obs dim 변화 확인(스칼라→per-env 면 차원 그대로 1 슬롯 평균을 쓸지, env별 1값을 쓸지 결정. 보통 env별 1값이라 차원 유지).

---

## 7. 단위테스트 / acceptance (작업3 패턴 계승)

기존 작업3 테스트: "지연 1 step 이면 정확히 직전 action 적용 / 노이즈 켜면 분산↑ / level=0 이면 원래대로". 확장:

- [ ] **회귀**: `level=0`(D_cur=0) → 출력이 비-jitter 구현과 비트 동일(지연·노이즈 0).
- [ ] **(a) 분포**: D_cur=2 로 고정 후 다수 step 표집 → `_delay_steps` 가 {0,1,2} 를 ~균등하게 포함.
- [ ] **인덱싱 정확성**: 합성 action 시퀀스 주입 → `d_t=k` 일 때 적용값이 `k` step 전 raw 와 일치.
- [ ] **critic 동기화**: `deploy_delay_steps` obs == 그 step 액션 텀이 실제 적용한 `_delay_steps`.
- [ ] **(b)**: `p_change` 작을 때 `_delay_steps` 자기상관(연속 동일값 길이) 이 (a) 보다 김 — run-length 통계로 확인.
- [ ] **reset**: reset 직후 적용값이 default pose 명령(튐 없음).
- [ ] **짧은 학습**: jitter on 상태에서 level 이 정체 없이 상승(작업1 재실행 기준 계승).

---

## 8. 열린 결정 (구현 전 사용자 확인 권장)

1. **방법**: (a) 단독으로 시작? 아니면 처음부터 (b)?
2. **`D_max`(상한)**: 2(40 ms)? 3(60 ms)? 실로봇 측정 latency 분포가 있으면 그에 맞춤.
3. **시간상관 `p_change`**(b 채택 시): 초기값(예: 0.05~0.15)과 level 종속 형태.
4. **critic 정규화**: `d_t` 를 raw step 수 vs `d_t/D_max` 중 무엇으로 넣을지.
5. **noise 와의 결합**: action noise 도 같은 jitter 스케줄에 묶을지(현재 noise 는 매 step 이미 i.i.d. 라 별도 불필요할 가능성 큼).

---

## 참고 (관련 문서/노트북)
- `docs/2026-05_sim2real_deploydr_work_summary.md` — 작업3(동작 지연/노이즈) 원 구현 맥락.
- `deploy_dr_level_curriculum.ipynb` — level 추이/EMA gate/DR 채널 종속 시각화 (gait01 에서 delay 는 it 213 에 ON).
- `gait_weight_comparison.ipynb` — rollout 으로 발 데이터 수집하는 패턴(평가 스크립트 작성 시 재사용 가능).
