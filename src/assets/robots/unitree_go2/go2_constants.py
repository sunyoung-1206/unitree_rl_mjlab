"""Unitree Go2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src.assets.robots.unitree_go2.mj_native_electric_actuator import NativeElectricActuatorCfg

##
# MJCF and assets.
##

GO2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg() -> EntityCfg:
  """Get a fresh Go2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ARTICULATION,
  )

##
# Electric motor configs (mujoco_warp filterexact_coupled / GPU-only)
##

_COUPLED_SUBSTEPS = 200  # decimation=200 (0.1ms × 200 = 20ms policy dt)
_PD_RECOMPUTE = 50       # PD 재계산 주기: 5ms / 0.1ms = 50 physics steps

# Shared real-world motor physical parameters for the Unitree Go2.
# Sourced from the Go2 motor datasheet — used by every Method variant
# (Coupled/A+, A, B, A+ tloop) so the underlying physics stays consistent.
# `method` is intentionally NOT included here so each variant can override.
_GO2_MOTOR_PHYS = dict(
  Kt=0.26,       # [N·m/A]
  Ke=0.26,       # [V·s/rad_motor], physically Ke == Kt
  R=0.66,        # [Ω] (= 1.5 × R_line, Delta wiring)
  L=83e-6,       # [H] (= 0.5 × L_line, Delta wiring) — τ_e = L/R ≈ 126 µs
  gear_ratio=6.33,
  V_bus=30.8,    # [V] (mid-point of 28–33.6 V, randomized by sim2real DR)
  substeps=_COUPLED_SUBSTEPS,
  pd_substeps=_PD_RECOMPUTE,
  use_coupled=True,
)


# Coupled electric motor (default method=A+/ZOH on patched mjwarp).
GO2_COUPLED_ELECTRIC_HIP = NativeElectricActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=20.0, damping=1.0, effort_limit=23.5, saturation_effort=23.5,
  velocity_limit=30.0, armature=0.01,
  **_GO2_MOTOR_PHYS,
)
GO2_COUPLED_ELECTRIC_THIGH = NativeElectricActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=20.0, damping=1.0, effort_limit=23.5, saturation_effort=23.5,
  velocity_limit=30.0, armature=0.01,
  **_GO2_MOTOR_PHYS,
)
GO2_COUPLED_ELECTRIC_CALF = NativeElectricActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=40.0, damping=2.0, effort_limit=45.0, saturation_effort=45.0,
  velocity_limit=30.0, armature=0.02,
  **_GO2_MOTOR_PHYS,
)

GO2_COUPLED_ELECTRIC_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_COUPLED_ELECTRIC_HIP,
    GO2_COUPLED_ELECTRIC_THIGH,
    GO2_COUPLED_ELECTRIC_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


# Method A (BE consistent: integrator/Schur/Force 전부 β_be = 1/(1+h/τ)).
_MA_MOTOR = dict(**_GO2_MOTOR_PHYS, method="A")
GO2_METHODA_HIP = NativeElectricActuatorCfg(
  target_names_expr=(".*hip_.*",), stiffness=20.0, damping=1.0,
  effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0, armature=0.01, **_MA_MOTOR)
GO2_METHODA_THIGH = NativeElectricActuatorCfg(
  target_names_expr=(".*thigh_.*",), stiffness=20.0, damping=1.0,
  effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0, armature=0.01, **_MA_MOTOR)
GO2_METHODA_CALF = NativeElectricActuatorCfg(
  target_names_expr=(".*calf_.*",), stiffness=40.0, damping=2.0,
  effort_limit=45.0, saturation_effort=45.0, velocity_limit=30.0, armature=0.02, **_MA_MOTOR)
GO2_METHODA_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_METHODA_HIP, GO2_METHODA_THIGH, GO2_METHODA_CALF),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_methoda_robot_cfg() -> EntityCfg:
  """Go2 Method A (BE integrator + BE Schur + BE Force RHS)."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_METHODA_ARTICULATION,
  )


# Method B (ZOH integrator + BE Schur/Force RHS). GPU 전용 (mjwarp patched).
_MB_MOTOR = dict(**_GO2_MOTOR_PHYS, method="B")
GO2_METHODB_HIP = NativeElectricActuatorCfg(
  target_names_expr=(".*hip_.*",), stiffness=20.0, damping=1.0,
  effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0, armature=0.01, **_MB_MOTOR)
GO2_METHODB_THIGH = NativeElectricActuatorCfg(
  target_names_expr=(".*thigh_.*",), stiffness=20.0, damping=1.0,
  effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0, armature=0.01, **_MB_MOTOR)
GO2_METHODB_CALF = NativeElectricActuatorCfg(
  target_names_expr=(".*calf_.*",), stiffness=40.0, damping=2.0,
  effort_limit=45.0, saturation_effort=45.0, velocity_limit=30.0, armature=0.02, **_MB_MOTOR)
GO2_METHODB_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_METHODB_HIP, GO2_METHODB_THIGH, GO2_METHODB_CALF),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_methodb_robot_cfg() -> EntityCfg:
  """Go2 Method B (ZOH integrator + BE Schur + BE Force RHS, GPU-only)."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_METHODB_ARTICULATION,
  )


# A+ (coupled) + torque-tracking integral loop in driver.
# Cfg는 _MA_MOTOR/Coupled와 동일, use_torque_loop만 추가로 켜짐.
# Ki=200, integral_max=0.5 — α∈[0.05, 1.0] 전 영역 robustness 검증 결과 (RR_calf 기준):
#   * ringing/posture/cap usage 모두 K_i=50 대비 우월
#   * gait period 600 ms 유지 (K_i≥400 에선 보행 망가짐)
#   * cap usage ≤ 58 % at α=0.2 (K_i=100 의 97% → K_i=200 의 58% 마진 확보)
_APLUS_TLOOP_MOTOR = dict(
  **_GO2_MOTOR_PHYS,
  method="A+",
  use_torque_loop=True, Ki=200.0, integral_max=0.5,
)
GO2_APLUS_TLOOP_HIP = NativeElectricActuatorCfg(
  target_names_expr=(".*hip_.*",), stiffness=20.0, damping=1.0,
  effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0,
  armature=0.01, **_APLUS_TLOOP_MOTOR)
GO2_APLUS_TLOOP_THIGH = NativeElectricActuatorCfg(
  target_names_expr=(".*thigh_.*",), stiffness=20.0, damping=1.0,
  effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0,
  armature=0.01, **_APLUS_TLOOP_MOTOR)
GO2_APLUS_TLOOP_CALF = NativeElectricActuatorCfg(
  target_names_expr=(".*calf_.*",), stiffness=40.0, damping=2.0,
  effort_limit=45.0, saturation_effort=45.0, velocity_limit=30.0,
  armature=0.02, **_APLUS_TLOOP_MOTOR)
GO2_APLUS_TLOOP_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(GO2_APLUS_TLOOP_HIP, GO2_APLUS_TLOOP_THIGH, GO2_APLUS_TLOOP_CALF),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_aplus_tloop_robot_cfg() -> EntityCfg:
  """Go2 A+ (coupled) + driver-rate torque-tracking integral loop.

  use_torque_loop=True 만 추가로 켠 변형. demag 고장 발생 시 적분기가
  I_cmd 를 자동 상승시켜 τ_actual ≈ τ_cmd 를 유지 → q 추종 오차가 q 가
  아닌 전류 신호로 전가됨 (진단 feature).
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_APLUS_TLOOP_ARTICULATION,
  )


def get_go2_coupled_electric_robot_cfg() -> EntityCfg:
  """Go2 coupled 전기모터 버전 (Schur complement back-EMF).

  filterexact_coupled dyntype 사용:
  implicit solver에 dt²·Kt·Ke·gr²/(L·(1+dt/τ))·JᵀJ 항이 추가되어
  전류 추적 정확도가 크게 향상됨 (Phase 3: 최대 19x).
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_COUPLED_ELECTRIC_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())

  viewer.launch(robot.spec.compile())
