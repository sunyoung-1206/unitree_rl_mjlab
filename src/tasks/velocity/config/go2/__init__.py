from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task
import src.tasks.velocity.mdp as src_mdp
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_flat_coupled_electric_env_cfg,
  unitree_go2_flat_aplus_tloop_electric_env_cfg,
  unitree_go2_flat_methoda_electric_env_cfg,
  unitree_go2_flat_methoda_electric_obshistory_env_cfg,
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
  # Vanilla DR baseline + builtin PD actuator sim setting (5 ms physics,
  # 20 ms policy). Events / rewards / obs 는 velocity_env_cfg 기본을 그대로 사용.
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.sim.mujoco.timestep = 0.005
  cfg.decimation = 4
  return cfg

register_mjlab_task(
  task_id="Unitree-Go2-Flat",
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
#
# base_cfg_fn: DR/reward/curriculum 로직은 액추에이터 모델과 독립적이므로,
# 기본 builtin-PD(_go2_flat_pd_cfg) 대신 전기모터 기반 env_cfg(예:
# unitree_go2_flat_methoda_electric_env_cfg)를 넘기면 같은 DeployDR 기법을
# 그 solving 방식 위에 그대로 적용할 수 있다.
# ─────────────────────────────────────────────────────────────────────────────
def _go2_flat_deploydr_cfg(play: bool = False, base_cfg_fn=_go2_flat_pd_cfg):
  cfg = base_cfg_fn(play=play)

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

  # ── 배포용 command/termination 설정 ──────────────────────────────────────
  # heading 100%: 모든 비정지 env 가 목표 방향을 바라보도록 wz 자동 계산 (deploy
  # 참조와 일치). init yaw 는 위 reset_base 에서 0 고정.
  twist = cfg.commands["twist"]
  twist.rel_heading_envs = 1.0

  # illegal_contact: base/hip/thigh/calf 중 하나라도 ≥1N 으로 지면 접촉 시 즉시 종료
  # (발 geom 은 sensor 에서 제외됨). 임계 10N → 1N.
  if "illegal_contact" in cfg.terminations:
    cfg.terminations["illegal_contact"].params["force_threshold"] = 1.0

  # fell_over: 방향 무관 단일 70° 대신 roll/pitch 분리 한계 (roll 0.8 / pitch 1.0 rad).
  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=src_mdp.bad_orientation_roll_pitch,
    params={"limit_roll": 0.8, "limit_pitch": 1.0},
  )

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

  # ── 작업 3: action delay + noise term + critic privileged obs 2종 ──────────
  # joint_pos action 을 delay(ring buffer) + gaussian noise 버전으로 교체.
  # delay_steps / noise_std 는 DeployDRCurriculum 이 level 로 런타임 set
  # (level=0 이면 delay=0, noise=0 → 기존 JointPositionAction 과 동일).
  jp = cfg.actions["joint_pos"]
  cfg.actions["joint_pos"] = src_mdp.DelayedNoisyJointPositionActionCfg(
    entity_name=jp.entity_name,
    actuator_names=jp.actuator_names,
    scale=jp.scale,
    use_default_offset=jp.use_default_offset,
    delay_max_steps=1,     # 참조 레포 스펙 1-step.
    noise_std_max=0.1,
  )
  # critic 에 현재 DR 상태 2개 추가 (78D → 80D). actor 는 47D 유지.
  critic_terms["deploy_delay_steps"] = ObservationTermCfg(
    func=src_mdp.deploy_delay_steps,
  )
  critic_terms["deploy_action_noise_std"] = ObservationTermCfg(
    func=src_mdp.deploy_action_noise_std,
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


# foot_gait 가중치 비교 변종: DeployDR-v0 와 모든 게 동일하되 foot_gait weight 만
# 0.10 → 0.50. 보행 리듬(트롯) 강제를 더 강하게 했을 때 배포 성능 비교용.
def _go2_flat_deploydr_gait05_cfg(play: bool = False):
  cfg = _go2_flat_deploydr_cfg(play=play)
  cfg.rewards["foot_gait"].weight = 0.50
  return cfg

register_mjlab_task(
  task_id="Unitree-Go2-Flat-DeployDR-Gait05-v0",
  env_cfg=_go2_flat_deploydr_gait05_cfg(),
  play_env_cfg=_go2_flat_deploydr_gait05_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)


# DeployDR-Gait05 위에 같은 DR 기법을 그대로 두고, 액추에이터 solving만
# builtin PD → Method A 전기모터(method="A", implicit Euler/BE Schur, 바닐라
# obs)로 교체한 변종. base_cfg_fn 하나만 바뀌므로 DR/reward/curriculum 로직은
# _go2_flat_deploydr_cfg 를 100% 재사용한다.
def _go2_flat_methoda_deploydr_gait05_cfg(play: bool = False):
  cfg = _go2_flat_deploydr_cfg(
    play=play, base_cfg_fn=unitree_go2_flat_methoda_electric_env_cfg,
  )
  cfg.rewards["foot_gait"].weight = 0.50
  return cfg

register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodA-Electric-DeployDR-Gait05-v0",
  env_cfg=_go2_flat_methoda_deploydr_gait05_cfg(),
  play_env_cfg=_go2_flat_methoda_deploydr_gait05_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
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

# Vanilla method A+ (filterexact/ZOH-consistent Schur, dynprm[4]=1). Actor obs
# is untouched (history_length=1, no delay) — same tier as MethodB-Electric and
# MethodA-Electric-Vanilla below, so all three methods have one directly-
# comparable baseline task.
register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodAPlus-Electric",
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

# Vanilla method A (BE-consistent Schur, dynprm[4]=0). Actor obs is untouched
# (history_length=1, no delay) — same tier as MethodAPlus-Electric and
# MethodB-Electric, so all three methods have one directly-comparable baseline.
register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodA-Electric",
  env_cfg=unitree_go2_flat_methoda_electric_env_cfg(
    use_velocity_action=_methoda_use_vel,
  ),
  play_env_cfg=unitree_go2_flat_methoda_electric_env_cfg(
    play=True, use_velocity_action=_methoda_use_vel,
  ),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Same as above, plus obs delay/history=5 (base_ang_vel/projected_gravity/
# joint_pos/joint_vel, 0..4 step lag) — no extra sim2real DR events on top of
# the velocity_env_cfg baseline DR. Train this for ~2000 iterations as Phase 1
# of the two-phase sim2real curriculum.
register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodA-Electric-ObsHistory",
  env_cfg=unitree_go2_flat_methoda_electric_obshistory_env_cfg(
    use_velocity_action=_methoda_use_vel,
  ),
  play_env_cfg=unitree_go2_flat_methoda_electric_obshistory_env_cfg(
    play=True, use_velocity_action=_methoda_use_vel,
  ),
  rl_cfg=unitree_go2_methoda_electric_ppo_runner_cfg(
    action_type=_METHODA_ACTION_TYPE,
  ),
  runner_cls=VelocityOnPolicyRunner,
)

# sim2real-DR-v1 expansions added on top of MethodA-Electric-ObsHistory: V_bus,
# actuator gains, motor strength, base/link mass, joint_pos_bias, external
# force/torque + widened foot_friction. Phase 2 of the curriculum: resume
# MethodA-Electric-ObsHistory's final checkpoint into this task for another
# ~2000 iter.
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

# PlayPD: fast visualization task for any MethodA-Electric-ObsHistory checkpoint
# (or Sim2Real, which builds on it). Uses builtin PD actuator + 5 ms physics so
# play runs in real time, keeps obs delay/history=5 and action scaling so
# checkpoints load.
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
