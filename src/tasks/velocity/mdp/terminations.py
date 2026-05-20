from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def bad_orientation_roll_pitch(
  env: ManagerBasedRlEnv,
  limit_roll: float,
  limit_pitch: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Roll / pitch 를 분리해서 임계 초과 시 종료.

  mjlab 기본 bad_orientation 은 방향 무관 단일 각도(acos(-g_z))만 본다. deploy 는
  좌우 전복(roll)과 앞뒤 숙임(pitch)에 다른 한계를 두고 싶을 때가 많다.

  projected_gravity_b g = R^T·[0,0,-1] (직립 시 [0,0,-1]). ZYX(yaw,pitch,roll) 기준
  g = [sinθ, -cosθ·sinφ, -cosθ·cosφ] 이므로:
    pitch θ = atan2(g_x, sqrt(g_y²+g_z²)),  roll φ = atan2(-g_y, -g_z).
  |roll|>limit_roll OR |pitch|>limit_pitch 면 종료.
  """
  asset = env.scene[asset_cfg.name]
  g = asset.data.projected_gravity_b  # [B, 3]
  roll = torch.atan2(-g[:, 1], -g[:, 2])
  pitch = torch.atan2(g[:, 0], torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2))
  return (roll.abs() > limit_roll) | (pitch.abs() > limit_pitch)