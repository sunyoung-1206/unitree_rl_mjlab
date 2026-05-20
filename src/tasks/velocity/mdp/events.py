"""Custom DR event terms for sim2real-DR-v1.

Three terms not covered by mjlab's built-in DR module:
  - randomize_V_bus            : per-env bus voltage sampled each episode.
  - randomize_motor_strength   : per-env Kt/Ke scale on actuator_gainprm + dynprm.
  - joint_pos_bias             : per-env mechanical zero offset, propagated to both
                                 entity.data.default_joint_pos and the
                                 JointPositionAction offset for consistency.

Pattern follows ``encoder_bias`` (mjlab/envs/mdp/dr/joint.py): sample uniformly per
env-per-joint and assign to the relevant tensor / model field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


__all__ = (
  "randomize_V_bus",
  "randomize_motor_strength",
  "joint_pos_bias",
  "reset_joints_by_scale",
  "push_by_setting_velocity_logged",
)


def _resolve_env_ids(env: "ManagerBasedRlEnv", env_ids: torch.Tensor | None) -> torch.Tensor:
  if env_ids is None:
    return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  return env_ids.to(env.device, dtype=torch.long)


def randomize_V_bus(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  voltage_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Sample per-env bus voltage and write to each NativeElectricActuator._V_bus.

  Args:
    voltage_range: (min, max) bus voltage in volts. Sampled uniformly.
  """
  asset = env.scene[asset_cfg.name]
  env_ids = _resolve_env_ids(env, env_ids)
  n = len(env_ids)
  samples = sample_uniform(
    torch.tensor(voltage_range[0], device=env.device),
    torch.tensor(voltage_range[1], device=env.device),
    (n, 1),
    env.device,
  )
  for actuator in asset.actuators:
    if getattr(actuator, "_V_bus", None) is not None:
      actuator._V_bus[env_ids] = samples


def randomize_motor_strength(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  scale_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Scale Kt and Ke by the same per-env factor (Kt=Ke physically).

  Writes:
    - sim.model.actuator_gainprm[:, ctrl_ids, 0]   (Kt·gr)
    - sim.model.actuator_dynprm[:, ctrl_ids, 1]    (Ke·gr, plant)
    - sim.model.actuator_dynprm[:, ctrl_ids, 3]    (Ke·gr, nominal)

  All three are scaled by the SAME factor so the vendor mjwarp patch's
  demagnetization detector (which compares dynprm[1] vs dynprm[3]) stays inactive.
  """
  asset = env.scene[asset_cfg.name]
  env_ids = _resolve_env_ids(env, env_ids)
  n = len(env_ids)

  # Collect all ctrl ids from electric actuators.
  ctrl_id_list: list[int] = []
  for actuator in asset.actuators:
    if getattr(actuator, "_Ktgr", None) is None:
      continue
    ctrl_id_list.extend(actuator.global_ctrl_ids.cpu().tolist())
  if not ctrl_id_list:
    return
  ctrl_ids = torch.tensor(ctrl_id_list, device=env.device, dtype=torch.long)

  samples = sample_uniform(
    torch.tensor(scale_range[0], device=env.device),
    torch.tensor(scale_range[1], device=env.device),
    (n, len(ctrl_ids)),
    env.device,
  )

  # Multiply nominal Kt·gr / Ke·gr by per-env factor.
  default_gainprm = env.sim.get_default_field("actuator_gainprm")
  default_dynprm = env.sim.get_default_field("actuator_dynprm")
  env.sim.model.actuator_gainprm[env_ids[:, None], ctrl_ids, 0] = (
    default_gainprm[ctrl_ids, 0] * samples
  )
  env.sim.model.actuator_dynprm[env_ids[:, None], ctrl_ids, 1] = (
    default_dynprm[ctrl_ids, 1] * samples
  )
  env.sim.model.actuator_dynprm[env_ids[:, None], ctrl_ids, 3] = (
    default_dynprm[ctrl_ids, 3] * samples
  )


def joint_pos_bias(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  bias_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
  action_term_name: str = "joint_pos",
) -> None:
  """Per-env mechanical zero-position bias applied consistently to both:
    - asset.data.default_joint_pos
    - JointPositionAction._offset (clones default_joint_pos at build time).

  Mode should be ``"startup"``. The two tensors must stay in sync so that reset
  pose, action default offset, and the policy's "zero action" target all agree.
  """
  asset = env.scene[asset_cfg.name]
  env_ids = _resolve_env_ids(env, env_ids)
  n_envs = len(env_ids)

  num_joints = asset.num_joints
  bias = sample_uniform(
    torch.tensor(bias_range[0], device=env.device),
    torch.tensor(bias_range[1], device=env.device),
    (n_envs, num_joints),
    env.device,
  )

  # 1) entity default joint pos (read by reset_joints_by_offset and at action build).
  asset.data.default_joint_pos[env_ids] += bias

  # 2) action term offset — captured at __init__ before startup events run.
  try:
    term = env.action_manager.get_term(action_term_name)
  except (KeyError, AttributeError):
    return
  offset = getattr(term, "_offset", None)
  if offset is None:
    return
  target_ids = getattr(term, "_target_ids", None)
  if target_ids is None:
    offset[env_ids] += bias
  else:
    offset[env_ids] += bias[:, target_ids]


def reset_joints_by_scale(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  position_range: tuple[float, float],
  velocity_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """Reset joints to ``default * U(position_range)`` (scale 의미).

  mjlab 기본 ``reset_joints_by_offset`` 은 더하기(offset)만 지원한다. deploy
  baseline 은 곱하기(scale, 예: (0.5, 1.5))로 초기 자세를 흔든다 — 그 의도를
  이식. soft joint limit 으로 clamp.
  """
  env_ids = _resolve_env_ids(env, env_ids)
  asset = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  default_joint_vel = asset.data.default_joint_vel
  soft_limits = asset.data.soft_joint_pos_limits

  joint_ids = asset_cfg.joint_ids
  joint_pos = default_joint_pos[env_ids][:, joint_ids].clone()
  joint_pos *= sample_uniform(
    torch.tensor(position_range[0], device=env.device),
    torch.tensor(position_range[1], device=env.device),
    joint_pos.shape,
    env.device,
  )
  limits = soft_limits[env_ids][:, joint_ids]
  joint_pos = joint_pos.clamp_(limits[..., 0], limits[..., 1])

  joint_vel = default_joint_vel[env_ids][:, joint_ids].clone()
  joint_vel += sample_uniform(
    torch.tensor(velocity_range[0], device=env.device),
    torch.tensor(velocity_range[1], device=env.device),
    joint_vel.shape,
    env.device,
  )

  if isinstance(joint_ids, list):
    joint_ids = torch.tensor(joint_ids, device=env.device)
  asset.write_joint_state_to_sim(
    joint_pos.view(len(env_ids), -1),
    joint_vel.view(len(env_ids), -1),
    env_ids=env_ids,
    joint_ids=joint_ids,
  )


def push_by_setting_velocity_logged(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  velocity_range: dict,
  asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
  """mjlab push_by_setting_velocity 미러 + 적용한 (x, y) 외란을 env._last_push_xy 에
  기록 (Phase 5 critic 의 push_history_xy obs 용)."""
  asset = env.scene[asset_cfg.name]
  vel_w = asset.data.root_link_vel_w[env_ids]
  range_list = [
    velocity_range.get(k, (0.0, 0.0)) for k in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=env.device)
  sample = sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=env.device)
  vel_w += sample
  asset.write_root_link_velocity_to_sim(vel_w, env_ids=env_ids)

  if getattr(env, "_last_push_xy", None) is None:
    env._last_push_xy = torch.zeros((env.num_envs, 2), device=env.device)
  env._last_push_xy[env_ids] = sample[:, :2]
