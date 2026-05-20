from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
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
  return cfg

register_mjlab_task(
  task_id="Unitree-Go2-Flat-DeployDR-v0",
  env_cfg=_go2_flat_deploydr_cfg(),
  play_env_cfg=_go2_flat_deploydr_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  # floor clip 미적용 (참조 레포 + 사용자 지시).
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
