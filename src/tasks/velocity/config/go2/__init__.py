from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import register_mjlab_task
import src.tasks.velocity.mdp as src_mdp
from src.tasks.velocity.rl import VelocityFloorClippedRunner, VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_flat_coupled_electric_env_cfg,
  unitree_go2_flat_aplus_tloop_electric_env_cfg,
  unitree_go2_flat_methoda_electric_env_cfg,
  unitree_go2_flat_methoda_electric_sim2real_env_cfg,
  unitree_go2_flat_methoda_electric_playpd_env_cfg,
  unitree_go2_flat_methodb_electric_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .rl_cfg import (
  unitree_go2_methoda_electric_ppo_runner_cfg,
  unitree_go2_methodb_electric_ppo_runner_cfg,
  unitree_go2_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

def _go2_flat_pd_cfg(play: bool = False):
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.sim.mujoco.timestep = 0.005  # 5ms
  cfg.decimation = 4               # policy dt = 20ms (5ms × 4)

  # ── DR expansions (mirrored from MethodA-Electric Sim2Real) ──
  # randomize_V_bus / randomize_motor_strength no-op on builtin PD (they check
  # for _V_bus / _Ktgr fields). Kept for 1:1 parity with MA-P2 so the event
  # set matches if the actuator is later swapped to electric.
  cfg.events["foot_friction"].params["ranges"] = (0.2, 1.5)
  cfg.events["randomize_V_bus"] = EventTermCfg(
    mode="reset",
    func=src_mdp.randomize_V_bus,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "voltage_range": (28.0, 33.6),
    },
  )
  cfg.events["randomize_actuator_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "kp_range": (0.8, 1.2),
      "kd_range": (0.8, 1.2),
      "distribution": "log_uniform",
      "operation": "scale",
    },
  )
  cfg.events["randomize_motor_strength"] = EventTermCfg(
    mode="startup",
    func=src_mdp.randomize_motor_strength,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "scale_range": (0.9, 1.1),
    },
  )
  cfg.events["randomize_base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "ranges": (-1.5, 3.0),
      "operation": "add",
    },
  )
  cfg.events["randomize_link_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=".*(hip|thigh|calf).*"),
      "ranges": (0.9, 1.1),
      "operation": "scale",
    },
  )
  cfg.events["joint_pos_bias"] = EventTermCfg(
    mode="startup",
    func=src_mdp.joint_pos_bias,
    params={
      "asset_cfg": SceneEntityCfg("robot"),
      "bias_range": (-0.03, 0.03),
    },
  )
  cfg.events["external_force_torque"] = EventTermCfg(
    mode="interval",
    interval_range_s=(8.0, 12.0),
    func=envs_mdp.apply_external_force_torque,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
      "force_range": (-30.0, 30.0),
      "torque_range": (-3.0, 3.0),
    },
  )

  return cfg

register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=_go2_flat_pd_cfg(),
  play_env_cfg=_go2_flat_pd_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  # PGTT-style reward floor clip at min=0 per step. Isolated to this task.
  runner_cls=VelocityFloorClippedRunner,
)

# A/B 비교 변종: env / DR / PPO config 모두 Unitree-Go2-Flat 과 동일하되
# floor clip만 제거 (VelocityOnPolicyRunner 사용). PGTT 식 floor clip의
# 효과 자체를 검증하기 위한 ablation.
register_mjlab_task(
  task_id="Unitree-Go2-Flat-NoClip",
  env_cfg=_go2_flat_pd_cfg(),
  play_env_cfg=_go2_flat_pd_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


# ─────────────────────────────────────────────────────────────────────────────
# Deploy-style DR 실험 task. Isaac Lab deploy baseline의 DR/reward/curriculum
# 의도를 mjlab API로 이식. 기존 task 는 건드리지 않고 이 task 안에서만 변경한다.
#
# Phase 1: DR/reward/obs 모두 기존 Flat(_go2_flat_pd_cfg)과 100% 동일.
#          단 floor clip 은 미적용 (참조 레포 미사용 + 사용자 지시) →
#          VelocityOnPolicyRunner 사용.
# ─────────────────────────────────────────────────────────────────────────────
def _go2_flat_deploydr_cfg(play: bool = False):
  cfg = _go2_flat_pd_cfg(play=play)

  # ── Phase 2: deploy-style DR (끄기 + 줄이기) ──────────────────────────────
  # 참조 deploy baseline 은 mass / COM / PD gain / 외력 / encoder bias 를 흔들지
  # 않는다. 아래 9개 DR 이벤트를 완전히 제거 (DR 강도 ↓ → tracking reward 회복).
  for _ev in (
    "randomize_base_mass",
    "randomize_link_mass",
    "base_com",
    "randomize_actuator_gains",
    "randomize_motor_strength",
    "randomize_V_bus",
    "external_force_torque",
    "encoder_bias",
    "joint_pos_bias",
  ):
    cfg.events.pop(_ev, None)

  # push_robot: 5초 고정 간격, 수평 linear 만 ±0.5 (z / 각속도 제거).
  # play 모드에서는 rough cfg 가 push_robot 을 이미 제거하므로 가드.
  # func 을 logged 버전으로 교체 → Phase 5 critic 의 push_history_xy obs 가 읽음.
  push = cfg.events.get("push_robot")
  if push is not None:
    push.func = src_mdp.push_by_setting_velocity_logged
    push.interval_range_s = (5.0, 5.0)
    push.params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}

  # foot_friction: range (0.3, 1.25) 로 통일.
  # TODO(mjlab): geom_friction 은 num_buckets 미지원 → startup 분포 안정화 불가.
  # TODO(mjlab): per-env 200-reset 마다 재샘플하는 hybrid 메커니즘 미지원 →
  #              현재 startup-only. interval mode 커스텀 event 로 추후 구현 가능.
  cfg.events["foot_friction"].params["ranges"] = (0.3, 1.25)

  # reset_base: 위치 / 요 고정, 초기 속도만 6축 ±0.5 (deploy 의도: 자세는
  # 일정하게 출발하되 초기 속도 외란으로 강건성 학습).
  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {
    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0),
  }
  reset_base.params["velocity_range"] = {
    "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
  }

  # reset_robot_joints: 초기 자세를 default × U(0.5, 1.5) 로 scale (deploy 의도).
  # mjlab 기본은 offset 만 제공 → src_mdp.reset_joints_by_scale 로 교체.
  rj = cfg.events["reset_robot_joints"]
  rj.func = src_mdp.reset_joints_by_scale
  rj.params = {
    "position_range": (0.5, 1.5),
    "velocity_range": (0.0, 0.0),
    "asset_cfg": rj.params["asset_cfg"],
  }

  # ── Phase 3: DR curriculum (단일 스칼라 level, reward EMA 기반 자동 조정) ──
  # level 이 obs noise scale + push velocity range 에 곱해진다 (level=0.1 에서 시작,
  # DR 거의 꺼진 상태).
  cfg.curriculum["deploy_dr"] = CurriculumTermCfg(
    func=src_mdp.DeployDRCurriculum,
    params={},  # 기본값은 DeployDRCurriculum.__init__ 의 스펙 그대로.
  )

  # ── Phase 4: foot_gait reward ─────────────────────────────────────────────
  # 기존 mdp.feet_gait 가 deploy 의 foot_gait 공식과 동일하다:
  #   phase=(ep_len*dt)%0.6/0.6, leg_phase=(phase+offset)%1, stance=leg_phase<0.56,
  #   reward=mean(stance==contact)*(cmd>thr).  offset [0,0.5,0.5,0] = 대각 trot.
  # → 중복 함수 추가 대신 deploy task 의 weight 만 0.5 → 0.10 으로 조정.
  cfg.rewards["foot_gait"].weight = 0.10

  # ── Phase 5: asymmetric critic — privileged obs 3종 추가 ──────────────────
  # critic group(enable_corruption=False)에만 추가. actor 는 47D 그대로 유지.
  # 기존 critic 은 base_lin_vel/foot_height/foot_air_time/foot_contact/
  # foot_contact_forces 를 이미 가짐 → friction/level/push_history 만 신설.
  critic_terms = cfg.observations["critic"].terms
  # base_lin_vel 은 deploy baseline 처럼 ×2 스케일.
  if "base_lin_vel" in critic_terms:
    critic_terms["base_lin_vel"].scale = 2.0
  critic_terms["foot_friction_coeff"] = ObservationTermCfg(
    func=src_mdp.foot_friction_coeff,
    params={"asset_cfg": SceneEntityCfg("robot", geom_names=(
      "FR_foot_collision", "FL_foot_collision",
      "RR_foot_collision", "RL_foot_collision",
    ))},
  )
  critic_terms["deploy_curriculum_level"] = ObservationTermCfg(
    func=src_mdp.deploy_curriculum_level,
  )
  critic_terms["push_history_xy"] = ObservationTermCfg(
    func=src_mdp.last_push_xy,
  )

  return cfg

register_mjlab_task(
  task_id="Unitree-Go2-Flat-DeployDR-v0",
  env_cfg=_go2_flat_deploydr_cfg(),
  play_env_cfg=_go2_flat_deploydr_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  # floor clip 미적용 (참조 레포 + 사용자 지시).
  runner_cls=VelocityOnPolicyRunner,
)


# No-DR clean baseline: DeployDR-v0 와 reward / foot_gait(0.10) / critic obs /
# reset 설정 전부 동일하되 DR 만 끈 통제 변종. 차이는:
#   - deploy_dr curriculum level 을 0.0 에 고정 (level_max=0) → obs noise 0, push 0
#   - foot_friction 을 nominal 단일값 (1.0, 1.0) 으로 (startup friction DR 제거)
# track_linear 절대 갭 측정 + 회귀 테스트용. 삭제 금지.
def _go2_flat_deploydr_nodr_cfg(play: bool = False):
  cfg = _go2_flat_deploydr_cfg(play=play)
  cfg.curriculum["deploy_dr"].params = {
    "level_init": 0.0,
    "level_min": 0.0,
    "level_max": 0.0,  # level 이 절대 오르지 않음 → DR off 유지.
  }
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)  # nominal 단일값.
  return cfg

register_mjlab_task(
  task_id="Unitree-Go2-Flat-DeployDR-NoDR-v0",
  env_cfg=_go2_flat_deploydr_nodr_cfg(),
  play_env_cfg=_go2_flat_deploydr_nodr_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat-Coupled-Electric",
  env_cfg=unitree_go2_flat_coupled_electric_env_cfg(),
  play_env_cfg=unitree_go2_flat_coupled_electric_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# A+ (coupled) + driver-rate torque-tracking integral loop.
# 관측/액션 공간이 Coupled-Electric 와 동일하므로 같은 PPO 체크포인트 사용 가능.
register_mjlab_task(
  task_id="Unitree-Go2-Flat-Coupled-Tloop-Electric",
  env_cfg=unitree_go2_flat_aplus_tloop_electric_env_cfg(),
  play_env_cfg=unitree_go2_flat_aplus_tloop_electric_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Action type 토글: 여기서 "position" ↔ "velocity" 만 바꾸면 됨.
# env_cfg의 use_velocity_action 과 rl_cfg의 action_type 을 같은 값으로 맞출 것.
_METHODA_ACTION_TYPE = "position"  # or "velocity"
_methoda_use_vel = _METHODA_ACTION_TYPE == "velocity"

# Base electric task: real motor params (Kt/Ke/R/L/V_bus) + obs delay/history=5,
# but no extra sim2real DR events on top of the velocity_env_cfg baseline DR.
# Train this for ~2000 iterations as Phase 1 of the two-phase sim2real curriculum.
register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodA-Electric",
  env_cfg=unitree_go2_flat_methoda_electric_env_cfg(
    use_velocity_action=_methoda_use_vel,
  ),
  play_env_cfg=unitree_go2_flat_methoda_electric_env_cfg(
    play=True, use_velocity_action=_methoda_use_vel,
  ),
  rl_cfg=unitree_go2_methoda_electric_ppo_runner_cfg(
    action_type=_METHODA_ACTION_TYPE,
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# sim2real-DR-v1 expansions added on top of the base electric task: V_bus,
# actuator gains, motor strength, base/link mass, joint_pos_bias, external
# force/torque + widened foot_friction. Phase 2 of the curriculum: resume the
# base task's final checkpoint into this task for another ~2000 iter.
register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodA-Electric-Sim2Real",
  env_cfg=unitree_go2_flat_methoda_electric_sim2real_env_cfg(
    use_velocity_action=_methoda_use_vel,
  ),
  play_env_cfg=unitree_go2_flat_methoda_electric_sim2real_env_cfg(
    play=True, use_velocity_action=_methoda_use_vel,
  ),
  rl_cfg=unitree_go2_methoda_electric_ppo_runner_cfg(
    action_type=_METHODA_ACTION_TYPE,
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# PlayPD: fast visualization task for any MethodA-Electric checkpoint (base or
# Sim2Real). Uses builtin PD actuator + 5 ms physics so play runs in real time,
# keeps obs delay/history=5 and action scaling so checkpoints load.
register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodA-Electric-PlayPD",
  env_cfg=unitree_go2_flat_methoda_electric_playpd_env_cfg(
    use_velocity_action=_methoda_use_vel,
  ),
  play_env_cfg=unitree_go2_flat_methoda_electric_playpd_env_cfg(
    play=True, use_velocity_action=_methoda_use_vel,
  ),
  rl_cfg=unitree_go2_methoda_electric_ppo_runner_cfg(
    action_type=_METHODA_ACTION_TYPE,
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# Method B (ZOH integrator + BE Schur/Force RHS) — GPU-only, mjwarp-patched.
_METHODB_ACTION_TYPE = "position"  # or "velocity"
_methodb_use_vel = _METHODB_ACTION_TYPE == "velocity"

register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodB-Electric",
  env_cfg=unitree_go2_flat_methodb_electric_env_cfg(
    use_velocity_action=_methodb_use_vel,
  ),
  play_env_cfg=unitree_go2_flat_methodb_electric_env_cfg(
    play=True, use_velocity_action=_methodb_use_vel,
  ),
  rl_cfg=unitree_go2_methodb_electric_ppo_runner_cfg(
    action_type=_METHODB_ACTION_TYPE,
  ),
  runner_cls=VelocityOnPolicyRunner,
)
