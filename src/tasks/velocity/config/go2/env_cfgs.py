"""Unitree Go2 velocity environment configurations."""

from typing import Literal

from src.assets.robots import (
  get_go2_robot_cfg,
)
from src.assets.robots.unitree_go2.go2_constants import (
  get_go2_coupled_electric_robot_cfg,
  get_go2_methoda_robot_cfg,
  get_go2_methodb_robot_cfg,
  get_go2_aplus_tloop_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
import src.tasks.velocity.mdp as src_mdp

from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

TerrainType = Literal["rough", "obstacles"]


def unitree_go2_rough_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500

  cfg.scene.entities = {"robot": get_go2_robot_cfg()}

  # Set raycast sensor frame to Go2 base_link.
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # Grab all collision geoms...
      pattern=r".*_collision\d*$",
      # Except for the foot geoms.
      exclude=tuple(geom_names),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 1.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["std_standing"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.05,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.1,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*(FR|FL|RR|RL)_hip_joint.*": 0.15,
    r".*(FR|FL|RR|RL)_thigh_joint.*": 0.35,
    r".*(FR|FL|RR|RL)_calf_joint.*": 0.5,
  }

  cfg.rewards["foot_gait"].params["offset"] = [0.0, 0.5, 0.5, 0.0]
  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 10.0},
  )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_go2_flat_coupled_electric_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Go2 flat terrain + MuJoCo-native coupled 전기모터 (Method A+: filterexact Schur).

  Physics dt = 0.1ms, policy dt = 20ms (decimation=200).
  """
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_go2_coupled_electric_robot_cfg()}

  cfg.sim.mujoco.timestep = 0.0001   # 0.1ms
  cfg.decimation = 200                # 0.1ms × 200 = 20ms policy dt

  return cfg


def unitree_go2_flat_methoda_electric_env_cfg(
  play: bool = False,
  use_velocity_action: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Go2 flat terrain + Method A electric motor, vanilla observations.

  method="A": BE integrator + BE Schur + BE Force RHS, dynprm[4]=0
  (β_int = β_imp = 1/(1+h/τ) on patched mjwarp). Actor obs stays at the
  velocity_env_cfg default (history_length=1, no delay) — mirrors how
  MethodAPlus-Electric (A+) and MethodB-Electric are left vanilla, so all
  three methods have one directly-comparable baseline.

  Physics dt = 0.1ms, policy dt = 20ms (decimation=200). Real-world motor
  parameters (Kt 0.26 N·m/A, V_bus 30.8 V, R 0.66 Ω, L 83 µH) via
  ``GO2_METHODA_ARTICULATION``.

  For the obs history=5 / per-term delay variant used as Phase 1 of the
  two-phase sim2real curriculum, use
  ``unitree_go2_flat_methoda_electric_obshistory_env_cfg``.

  Args:
    play: play(=eval) 모드 여부.
    use_velocity_action: True 시 JointVelocity 액션으로 교체.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_go2_methoda_robot_cfg()}

  cfg.sim.mujoco.timestep = 0.0001   # 0.1ms
  cfg.decimation = 200                # 0.1ms × 200 = 20ms policy dt

  if use_velocity_action:
    cfg.actions = {
      "joint_vel": JointVelocityActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=5.0,
        use_default_offset=True,
      )
    }

  return cfg


def unitree_go2_flat_methoda_electric_obshistory_env_cfg(
  play: bool = False,
  use_velocity_action: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Go2 flat terrain + Method A electric motor, obs history=5 + per-term delay.

  Same actuator physics as ``unitree_go2_flat_methoda_electric_env_cfg``
  (method="A", BE integrator + BE Schur + BE Force RHS, dynprm[4]=0), plus
  an actor-observation override: history_length=5 with per-term delay 0..4
  steps on base_ang_vel / projected_gravity / joint_pos / joint_vel.

  Baseline domain randomization (foot_friction 0.3..1.2, encoder_bias,
  base_com, push_robot) is inherited from velocity_env_cfg. For the extended
  sim2real DR variant, use
  ``unitree_go2_flat_methoda_electric_sim2real_env_cfg``.

  Args:
    play: play(=eval) 모드 여부.
    use_velocity_action: True 시 JointVelocity 액션으로 교체.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_go2_methoda_robot_cfg()}

  cfg.sim.mujoco.timestep = 0.0001   # 0.1ms
  cfg.decimation = 200                # 0.1ms × 200 = 20ms policy dt

  if use_velocity_action:
    cfg.actions = {
      "joint_vel": JointVelocityActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=5.0,
        use_default_offset=True,
      )
    }

  # ── Observation: actor history + per-term delay ──────────────────────
  actor_group = cfg.observations["actor"]
  actor_group.history_length = 5
  for term_name in ("base_ang_vel", "projected_gravity", "joint_pos", "joint_vel"):
    term = actor_group.terms[term_name]
    term.delay_min_lag = 0
    term.delay_max_lag = 4

  return cfg


def unitree_go2_flat_methoda_electric_sim2real_env_cfg(
  play: bool = False,
  use_velocity_action: bool = False,
) -> ManagerBasedRlEnvCfg:
  """MethodA Electric + sim2real-DR-v1 expansions.

  Adds seven new DR event terms on top of the base electric cfg and widens
  ``foot_friction`` (0.3, 1.2) → (0.2, 1.5). The base cfg already carries the
  real motor parameters and obs delay/history.

  Added DR events:
    * randomize_V_bus              reset    U(28.0, 33.6) V
    * randomize_actuator_gains     startup  kp & kd log_uniform(0.8, 1.2)
    * randomize_motor_strength     startup  Kt = Ke shared U(0.9, 1.1)
                                            → gainprm[0], dynprm[1], dynprm[3]
    * randomize_base_mass          startup  base_link mass + U(-1.5, 3.0) kg
    * randomize_link_mass          startup  hip/thigh/calf × U(0.9, 1.1)
    * joint_pos_bias               startup  +U(-0.03, 0.03) rad on
                                            default_joint_pos + action _offset
    * external_force_torque        interval 8~12 s, F ±30 N, τ ±3 N·m on base

  Intended for the "Phase 2" half of a two-phase sim2real curriculum: train
  ``Unitree-Go2-Flat-MethodA-Electric-ObsHistory`` (no extra DR) for ~2000
  iter first, then resume into this task for another 2000 iter.

  Args:
    play: play(=eval) 모드 여부.
    use_velocity_action: True 시 JointVelocity 액션으로 교체.
  """
  cfg = unitree_go2_flat_methoda_electric_obshistory_env_cfg(
    play=play, use_velocity_action=use_velocity_action,
  )

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


def unitree_go2_flat_methoda_electric_playpd_env_cfg(
  play: bool = False,
  use_velocity_action: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Fast play env for the MethodA-Electric-ObsHistory trained policy.

  Same obs/action space as the training task (so the trained checkpoint loads
  with ``strict=True``), but swaps the electric-motor actuator for builtin PD
  with dt = 5 ms / decimation = 4 — real-time visualization on a single GPU.

  Visual fidelity trade-off:
    * No current dynamics / V_bus saturation / motor transient.
    * No domain randomization (clean baseline).
    * Same PD gains (kp 20/20/40, kd 1/1/2) so joint behaviour is close to the
      training environment in steady state.

  Intended for ``scripts/play.py`` only — do not train with this task.
  """
  cfg = unitree_go2_flat_methoda_electric_obshistory_env_cfg(
    play=play, use_velocity_action=use_velocity_action,
  )
  cfg.scene.entities = {"robot": get_go2_robot_cfg()}
  cfg.sim.mujoco.timestep = 0.005   # 5 ms physics
  cfg.decimation = 4                # 20 ms policy
  return cfg


def unitree_go2_flat_aplus_tloop_electric_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Go2 flat terrain + A+ (coupled) + driver-rate torque-tracking integral loop.

  Coupled (A+) 와 동일한 시간 위계 (physics 0.1ms, policy 20ms, decimation=200,
  pd_substeps=50 → driver/PD/적분 갱신 주기 5ms).

  use_torque_loop=True 가 켜져 있어 감자(demag) 고장 시 적분기가 I_cmd 를 자동
  보정. healthy 에서는 적분기 ≈ 0 으로 머무르므로 nominal 동작과 거의 동일하다.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_go2_aplus_tloop_robot_cfg()}

  cfg.sim.mujoco.timestep = 0.0001   # 0.1ms
  cfg.decimation = 200                # 0.1ms × 200 = 20ms policy dt

  return cfg


def unitree_go2_flat_methodb_electric_env_cfg(
  play: bool = False,
  use_velocity_action: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Go2 flat terrain + Method B (ZOH integrator + BE Schur + BE Force RHS).

  GPU 전용 (patched mjwarp). dynprm[4] = 2.
  Integrator 는 1차 선형 ODE 의 정확 해(ZOH), implicit damping/Jacobian 은 BE 형식.

  Physics dt = 0.1ms, policy dt = 20ms (decimation=200).
  PD/tau target 재계산 주기 = 5ms → policy dt 안에서 4회 업데이트 (pd_substeps=50).

  Args:
    play: play(=eval) 모드 여부.
    use_velocity_action: True 시 JointVelocity 액션으로 교체.
  """
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.scene.entities = {"robot": get_go2_methodb_robot_cfg()}

  cfg.sim.mujoco.timestep = 0.0001   # 0.1ms
  cfg.decimation = 200                # 0.1ms × 200 = 20ms policy dt

  if use_velocity_action:
    cfg.actions = {
      "joint_vel": JointVelocityActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale=5.0,
        use_default_offset=True,
      )
    }

  return cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Unitree Go2 flat terrain velocity configuration."""
  cfg = unitree_go2_rough_env_cfg(play=play)

  cfg.sim.njmax = 300
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 64
  cfg.sim.nconmax = None

  # Switch to flat terrain.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # Remove raycast sensor and height scan (no terrain to scan).
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]

  # Disable terrain curriculum (not present in play mode since rough clears all).
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-0.5, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.5, 0.5)
    twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)

  return cfg
