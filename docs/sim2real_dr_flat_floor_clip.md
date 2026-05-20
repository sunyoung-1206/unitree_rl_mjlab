# Unitree-Go2-Flat: DR 확장 + reward floor clip 변경 내역

PGTT vs unitree_rl_mjlab sim2real 비교 분석(2026-05-19) 결과 도출된 두 가지 변경을
**`Unitree-Go2-Flat` 태스크에만** 격리해 적용한 패치입니다.

대상 task: `Unitree-Go2-Flat` (builtin PD actuator, 5 ms physics × decimation 4 = 20 ms policy dt).
다른 모든 task (Rough / MethodA-Electric / MethodA-Electric-Sim2Real / Coupled / MethodB / A+ 등)는
**영향받지 않음**.

관련 분석 문서:
- `../../pgtt_analysis/pgtt_vs_mjlab_comparison.md` (1차)
- `../../pgtt_analysis/pgtt_vs_mjlab_comparison_v2.md` (2차, floor clip 발견)

---

## 1. 변경 1 — MA-P2 동등 도메인 랜덤화 7종 추가

**파일:** `src/tasks/velocity/config/go2/__init__.py`

기존 `_go2_flat_pd_cfg(play)` 헬퍼가 `unitree_go2_flat_env_cfg`로부터 cfg를 만들고
timestep/decimation만 5 ms / 4로 override하던 부분에, MA-P2 (`unitree_go2_flat_methoda_electric_sim2real_env_cfg`,
env_cfgs.py:213-312)와 동일한 DR 이벤트 7종을 추가했습니다.

### 적용 항목

| 이벤트 | mode | 범위 | mjlab MA-P2 출처 |
|---|---|---|---|
| `foot_friction` (기존 override) | startup | `(0.2, 1.5)` (기본 `(0.3, 1.2)` 확장) | env_cfgs.py:246 |
| `randomize_V_bus` | reset | `(28.0, 33.6)` V | env_cfgs.py:248-255 |
| `randomize_actuator_gains` | startup | kp/kd × log_uniform(0.8, 1.2) | env_cfgs.py:256-266 |
| `randomize_motor_strength` | startup | scale U(0.9, 1.1) | env_cfgs.py:267-274 |
| `randomize_base_mass` | startup | base_link +U(−1.5, +3.0) kg | env_cfgs.py:275-283 |
| `randomize_link_mass` | startup | hip/thigh/calf × U(0.9, 1.1) | env_cfgs.py:284-292 |
| `joint_pos_bias` | startup | ±0.03 rad | env_cfgs.py:293-300 |
| `external_force_torque` | interval 8~12 s | F ±30 N, τ ±3 N·m on base_link | env_cfgs.py:301-310 |

### Builtin PD 환경에서 no-op되는 이벤트

`randomize_V_bus` 와 `randomize_motor_strength` 는 전기모터 actuator의 `_V_bus` / `_Ktgr` 필드를
참조합니다 (events.py:40-62 / events.py:65-113). Builtin PD actuator에는 해당 필드가
없으므로 두 함수 모두 **안전한 no-op** (early return 또는 조건부 write).

**MA-P2 와 1:1 parity를 유지하기 위해** 두 이벤트를 그대로 등록했습니다. actuator를
나중에 전기모터로 교체하면 추가 코드 변경 없이 활성화됩니다.

### Diff 요약

```diff
 from .env_cfgs import (
   unitree_go2_flat_env_cfg,
   ...
 )
+from mjlab.envs import mdp as envs_mdp
+from mjlab.envs.mdp import dr
+from mjlab.managers.event_manager import EventTermCfg
+from mjlab.managers.scene_entity_config import SceneEntityCfg
+import src.tasks.velocity.mdp as src_mdp

 def _go2_flat_pd_cfg(play: bool = False):
   cfg = unitree_go2_flat_env_cfg(play=play)
   cfg.sim.mujoco.timestep = 0.005
   cfg.decimation = 4
+
+  # ── DR expansions (mirrored from MethodA-Electric Sim2Real) ──
+  cfg.events["foot_friction"].params["ranges"] = (0.2, 1.5)
+  cfg.events["randomize_V_bus"] = EventTermCfg(mode="reset", ...)
+  cfg.events["randomize_actuator_gains"] = EventTermCfg(mode="startup", ...)
+  cfg.events["randomize_motor_strength"] = EventTermCfg(mode="startup", ...)
+  cfg.events["randomize_base_mass"] = EventTermCfg(mode="startup", ...)
+  cfg.events["randomize_link_mass"] = EventTermCfg(mode="startup", ...)
+  cfg.events["joint_pos_bias"] = EventTermCfg(mode="startup", ...)
+  cfg.events["external_force_torque"] = EventTermCfg(mode="interval", ...)
+
   return cfg
```

### 격리 설계 — 왜 `env_cfgs.py:unitree_go2_flat_env_cfg`가 아닌 `__init__.py:_go2_flat_pd_cfg`인가

`unitree_go2_flat_env_cfg` 는 다음 함수들의 base로 호출됩니다 (env_cfgs.py):

- `unitree_go2_flat_coupled_electric_env_cfg` (L153)
- `unitree_go2_flat_methoda_electric_env_cfg` (L186) — **MA-P1, Phase 1 의도가 "DR 최소"**
- `unitree_go2_flat_aplus_tloop_electric_env_cfg` (L353)
- `unitree_go2_flat_methodb_electric_env_cfg` (L378)

만약 `unitree_go2_flat_env_cfg` 자체를 수정했다면 위 4개 task가 모두 같은 DR을 상속받게 되어
MethodA-Electric의 **2-phase curriculum (Phase 1 = DR 최소 → Phase 2 = DR 확장)** 의도가 깨집니다.
따라서 task 등록의 wrapper인 `_go2_flat_pd_cfg`에서만 DR을 덧붙여 다른 파생 task를 보호했습니다.

---

## 2. 변경 2 — PGTT 식 reward floor clip runner 추가

**파일:** `src/tasks/velocity/rl/runner.py`

PGTT는 `joystick_base.py:220` 에서 `reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)`
로 매 step의 reward 합을 **0으로 하한 클리핑** 합니다. 그 결과:

- termination=-1, lin_vel_z=-2 등 모든 페널티는 같은 step의 positive reward에 흡수되거나
  floor clip으로 사라짐 → PPO advantage에 음의 신호가 흐르지 않음.
- 강한 페널티 (PGTT termination=-1, mjlab is_terminated=-200)가 "공격적 학습 신호" 가 아니라
  **soft constraint** 로 동작.

mjlab은 동일한 dt scaling을 하지만 (RewardManager `scale_by_dt=True` 기본,
`reward_manager.py:117`) **floor clip은 없음** (`reward_manager.py:130` 의 `return self._reward_buf`
는 raw sum). 따라서 같은 weight를 그대로 두면 mjlab 쪽이 훨씬 강한 페널티로 동작합니다.

### 구현 — `VelocityFloorClippedRunner`

```python
class VelocityFloorClippedRunner(VelocityOnPolicyRunner):
  # PGTT joystick_base.py:220 mirror — reward = clip(sum(rewards) * dt, 0, 1e4).
  # Per-term `_episode_sums` (logging) stay unclipped; only the aggregate
  # returned to PPO is floored at 0 so negative-dominant steps don't bleed into
  # the advantage estimate.
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    rm = self.env.unwrapped.reward_manager
    _orig_compute = rm.compute

    def _floor_clipped_compute(dt):
      return torch.clamp(_orig_compute(dt), min=0.0, max=10000.0)

    rm.compute = _floor_clipped_compute
```

### 설계 결정

| 옵션 | 채택? | 이유 |
|---|---|---|
| `mjlab/managers/reward_manager.py:130` 직접 수정 | ❌ | site-packages 수정 — 다른 task / 다른 프로젝트에도 영향 |
| `env_cfg`에 새 플래그 추가 | ❌ | `ManagerBasedRlEnvCfg`는 mjlab 소유 dataclass — 수정 불가 |
| Env 클래스 서브클래싱 | ❌ | `register_mjlab_task`가 env_cls 인자를 받지 않음 (`registry.py:22-27`) |
| Runner 서브클래스에서 monkey-patch | ✅ | `runner_cls` 인자가 task별 등록 가능 — 격리 완벽 |

`mjlab/envs/manager_based_rl_env.py:388` 의 `self.reward_buf = self.reward_manager.compute(dt=self.step_dt)`
호출은 그대로 작동하되, `compute` 가 wrap된 버전으로 교체되어 clipped tensor 반환.

### 보존되는 동작

- `RewardManager._episode_sums[name]` (line 128): per-term 누적 — **unclipped 유지**. TensorBoard /
  WandB의 `Episode_Reward/{name}` 메트릭은 변경 없음. 어떤 항이 floor clip에 흡수되었는지 진단 가능.
- `RewardManager._step_reward` (line 129): per-term 순간값 — **unclipped 유지**.
- PPO advantage 계산용 aggregate (`env.reward_buf`) 만 clipped.

### Diff 요약

```diff
 # src/tasks/velocity/rl/runner.py
 class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
   ...

+class VelocityFloorClippedRunner(VelocityOnPolicyRunner):
+  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
+    super().__init__(env, train_cfg, log_dir, device)
+    rm = self.env.unwrapped.reward_manager
+    _orig_compute = rm.compute
+    def _floor_clipped_compute(dt):
+      return torch.clamp(_orig_compute(dt), min=0.0, max=10000.0)
+    rm.compute = _floor_clipped_compute
```

```diff
 # src/tasks/velocity/rl/__init__.py
 from .runner import (
+  VelocityFloorClippedRunner as VelocityFloorClippedRunner,
   VelocityOnPolicyRunner as VelocityOnPolicyRunner,
 )
```

```diff
 # src/tasks/velocity/config/go2/__init__.py
-from src.tasks.velocity.rl import VelocityOnPolicyRunner
+from src.tasks.velocity.rl import VelocityFloorClippedRunner, VelocityOnPolicyRunner

 register_mjlab_task(
   task_id="Unitree-Go2-Flat",
   env_cfg=_go2_flat_pd_cfg(),
   play_env_cfg=_go2_flat_pd_cfg(play=True),
   rl_cfg=unitree_go2_ppo_runner_cfg(),
-  runner_cls=VelocityOnPolicyRunner,
+  # PGTT-style reward floor clip at min=0 per step. Isolated to this task.
+  runner_cls=VelocityFloorClippedRunner,
 )
```

---

## 3. 영향 범위 검증

`/home/rbdo/miniconda3/envs/mjlab/bin/python` 으로 확인 (2026-05-19):

### DR 격리

```
flat_env_cfg (base) events    : ['base_com', 'encoder_bias', 'foot_friction',
                                  'push_robot', 'reset_base', 'reset_robot_joints']
Unitree-Go2-Flat task events  : (+7) base + V_bus + actuator_gains + motor_strength +
                                  base_mass + link_mass + joint_pos_bias + external_force_torque
MA-P1 events                  : (변경 없음) base 그대로 6개
MA-P2 events                  : (변경 없음) base + 7 그대로
Coupled-Electric events       : (변경 없음) base 그대로 6개
MethodB events                : (변경 없음) base 그대로 6개
```

→ MA-P1 의 "Phase 1 = DR 최소" 의도 보존됨.

### Runner 격리

```
Unitree-Go2-Flat   runner_cls : VelocityFloorClippedRunner
Unitree-Go2-Rough  runner_cls : VelocityOnPolicyRunner
(나머지 task)                  : VelocityOnPolicyRunner
```

### Floor clip 로직

```python
# 단위 테스트
fake_compute = lambda dt: torch.tensor([-2.0, -0.5, 1.5, -10.0])
wrapped      = lambda dt: torch.clamp(fake_compute(dt), 0.0, 10000.0)
wrapped(0.02)   # → tensor([0.0, 0.0, 1.5, 0.0])  ✓
```

---

## 4. 변경된 파일 목록

| 파일 | 변경 |
|---|---|
| `src/tasks/velocity/config/go2/__init__.py` | (1) 신규 import 5종, (2) `_go2_flat_pd_cfg`에 DR 7개 + foot_friction 범위 확장, (3) `Unitree-Go2-Flat` runner_cls 교체 |
| `src/tasks/velocity/rl/runner.py` | `VelocityFloorClippedRunner` 클래스 신설 |
| `src/tasks/velocity/rl/__init__.py` | `VelocityFloorClippedRunner` 재내보내기 |

수정하지 않은 파일 (격리 위해 의도적으로 유지):
- `src/tasks/velocity/velocity_env_cfg.py`
- `src/tasks/velocity/config/go2/env_cfgs.py`
- `src/tasks/velocity/config/go2/rl_cfg.py`
- mjlab site-packages 일체

---

## 5. 다른 task에 같은 변경을 적용하려면

### DR 확장만 (예: MA-P2에도 추가 — 이미 갖고 있음)

해당 없음. MA-P2 는 같은 7개 DR을 이미 자체 함수 `unitree_go2_flat_methoda_electric_sim2real_env_cfg`
에 갖고 있음.

### Floor clip만 (예: MA-P2에 floor clip 도입)

`__init__.py` 의 해당 task 등록에 `runner_cls=VelocityFloorClippedRunner` 한 줄 추가하면 됨.

```python
register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodA-Electric-Sim2Real",
  env_cfg=...,
  play_env_cfg=...,
  rl_cfg=unitree_go2_methoda_electric_ppo_runner_cfg(...),
  runner_cls=VelocityFloorClippedRunner,   # ← 추가
)
```

---

## 6. 학습 가이드

`Unitree-Go2-Flat` 학습 시 두 변경이 결합돼 작동:

- 강한 DR (11종 이벤트, base 6 + 추가 5 + no-op 2)이 첫 step부터 활성
- floor clip으로 학습 초반 fall이 잦더라도 `is_terminated=-200` 페널티가 PPO advantage에
  음수로 흐르지 않음 → 학습 안정성

기대 효과 (PGTT vs mjlab 비교 §F 가설):
- DR 강화 시 reward가 절반으로 떨어지던 증상 완화
- 학습 초반 `Episode_Reward/is_terminated` 가 음수로 누적되는 것은 변함없음 (per-term 로깅은 unclipped) →
  텐서보드에서 어떤 페널티가 floor에 흡수되었는지 진단 가능.

학습 명령 예시 (4096 envs, 5000 iter, tensorboard):
```bash
python scripts/train.py Unitree-Go2-Flat \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 5000 \
  --agent.logger tensorboard
```
