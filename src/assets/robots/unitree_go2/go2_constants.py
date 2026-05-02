"""Unitree Go2 constants."""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg
from src.assets.robots.unitree_go2.electric_actuator import ElectricMotorActuatorCfg
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
# Electric motor actuator config (PD + 전류-토크 ODE)
# play_mujoco.py ElectricMotorState와 동일한 파라미터
##

GO2_ELECTRIC_HIP = ElectricMotorActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  saturation_effort=23.5,
  velocity_limit=30.0,
  armature=0.01,
  Kt=0.128,
  Ke=0.128,
  R=0.3,
  L=1e-4,
  gear_ratio=6.33,
)

GO2_ELECTRIC_THIGH = ElectricMotorActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  saturation_effort=23.5,
  velocity_limit=30.0,
  armature=0.01,
  Kt=0.128,
  Ke=0.128,
  R=0.3,
  L=1e-4,
  gear_ratio=6.33,
)

GO2_ELECTRIC_CALF = ElectricMotorActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45.0,
  saturation_effort=45.0,
  velocity_limit=30.0,
  armature=0.02,
  Kt=0.128,
  Ke=0.128,
  R=0.3,
  L=1e-4,
  gear_ratio=6.33,
)

GO2_ELECTRIC_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_ELECTRIC_HIP,
    GO2_ELECTRIC_THIGH,
    GO2_ELECTRIC_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)

##
# Native electric motor actuator config (MuJoCo act/act_dot 통합)
# 전류 I를 MuJoCo state (d->act)에 직접 통합
##

# substeps=50: policy 5ms / dt_sub 0.1ms = 50 서브스텝
# 첫 서브스텝에서 PD → τ_des → I_des 캐시 (ZOH)
# 이후 49 서브스텝에서 최신 ω로 전압만 갱신

_SUBSTEPS = 50  # decimation과 동일하게 설정 (0.1ms × 50 = 5ms policy dt)

GO2_NATIVE_ELECTRIC_HIP = NativeElectricActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  saturation_effort=23.5,
  velocity_limit=30.0,
  armature=0.01,
  Kt=0.128,
  Ke=0.128,
  R=0.3,
  L=1e-4,
  gear_ratio=6.33,
  substeps=_SUBSTEPS,
)

GO2_NATIVE_ELECTRIC_THIGH = NativeElectricActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  saturation_effort=23.5,
  velocity_limit=30.0,
  armature=0.01,
  Kt=0.128,
  Ke=0.128,
  R=0.3,
  L=1e-4,
  gear_ratio=6.33,
  substeps=_SUBSTEPS,
)

GO2_NATIVE_ELECTRIC_CALF = NativeElectricActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45.0,
  saturation_effort=45.0,
  velocity_limit=30.0,
  armature=0.02,
  Kt=0.128,
  Ke=0.128,
  R=0.3,
  L=1e-4,
  gear_ratio=6.33,
  substeps=_SUBSTEPS,
)

GO2_NATIVE_ELECTRIC_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_NATIVE_ELECTRIC_HIP,
    GO2_NATIVE_ELECTRIC_THIGH,
    GO2_NATIVE_ELECTRIC_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
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

def get_go2_electric_robot_cfg() -> EntityCfg:
  """Go2 전기모터 ODE 액추에이터 버전.

  기본 PD 학습 정책과 동일한 구조이지만, 액추에이터가
  전류-토크 방정식(dI/dt)을 통해 토크를 출력함.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_ELECTRIC_ARTICULATION,
  )

def get_go2_native_electric_robot_cfg() -> EntityCfg:
  """Go2 MuJoCo-native 전기모터 버전.

  전류 I를 MuJoCo의 act state에 통합.
  MuJoCo implicit solver가 전기-기계 연립 시스템의
  ∂(act_dot)/∂(act) = -R/L 를 해석적으로 처리.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_NATIVE_ELECTRIC_ARTICULATION,
  )


##
# Coupled electric motor actuator config (Python callback: dyntype=user)
# callback에서 dI/dt = (V - R·I - Ke·gr·ω) / L 을 직접 계산
# filterexact와 달리 back-EMF가 act_dot에 포함되어 물리적으로 정확
# 단, mujoco_warp (GPU) 미지원 → standard mujoco (CPU) 전용
##

_COUPLED_SUBSTEPS = 200  # decimation=200 (0.1ms × 200 = 20ms policy dt)
_PD_RECOMPUTE = 50       # PD 재계산 주기: 5ms / 0.1ms = 50 physics steps

GO2_COUPLED_ELECTRIC_HIP = NativeElectricActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=20.0, damping=1.0, effort_limit=23.5, saturation_effort=23.5,
  velocity_limit=30.0, armature=0.01,
  Kt=0.128, Ke=0.128, R=0.3, L=1e-4, gear_ratio=6.33,
  substeps=_COUPLED_SUBSTEPS, pd_substeps=_PD_RECOMPUTE, use_coupled=True,
)

GO2_COUPLED_ELECTRIC_THIGH = NativeElectricActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=20.0, damping=1.0, effort_limit=23.5, saturation_effort=23.5,
  velocity_limit=30.0, armature=0.01,
  Kt=0.128, Ke=0.128, R=0.3, L=1e-4, gear_ratio=6.33,
  substeps=_COUPLED_SUBSTEPS, pd_substeps=_PD_RECOMPUTE, use_coupled=True,
)

GO2_COUPLED_ELECTRIC_CALF = NativeElectricActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=40.0, damping=2.0, effort_limit=45.0, saturation_effort=45.0,
  velocity_limit=30.0, armature=0.02,
  Kt=0.128, Ke=0.128, R=0.3, L=1e-4, gear_ratio=6.33,
  substeps=_COUPLED_SUBSTEPS, pd_substeps=_PD_RECOMPUTE, use_coupled=True,
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
_MA_MOTOR = dict(Kt=0.128, Ke=0.128, R=0.3, L=1e-4, gear_ratio=6.33,
                 substeps=_COUPLED_SUBSTEPS, pd_substeps=_PD_RECOMPUTE,
                 use_coupled=True, method="A")
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
_MB_MOTOR = dict(Kt=0.128, Ke=0.128, R=0.3, L=1e-4, gear_ratio=6.33,
                 substeps=_COUPLED_SUBSTEPS, pd_substeps=_PD_RECOMPUTE,
                 use_coupled=True, method="B")
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
# Ki=50, integral_max=0.5 (검증 스크립트 기본값과 동일).
_APLUS_TLOOP_MOTOR = dict(
  Kt=0.128, Ke=0.128, R=0.3, L=1e-4, gear_ratio=6.33,
  substeps=_COUPLED_SUBSTEPS, pd_substeps=_PD_RECOMPUTE,
  use_coupled=True, method="A+",
  use_torque_loop=True, Ki=50.0, integral_max=0.5,
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
