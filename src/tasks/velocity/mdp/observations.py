from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase


# ── Phase 5: critic 전용 privileged observations (deploy baseline) ────────────
def foot_friction_coeff(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """현재 env 의 발 마찰계수 (tangential, axis 0) 평균. [B, 1].

  geom_friction 은 [B, num_geoms, 3] (per-env). asset_cfg.geom_ids 는 global geom
  index 로 해석됨. critic 이 DR 로 흔들린 마찰을 알 수 있게 제공.
  """
  gf = env.sim.model.geom_friction
  geom_ids = asset_cfg.geom_ids
  return gf[:, geom_ids, 0].mean(dim=1, keepdim=True)


def deploy_curriculum_level(env: ManagerBasedRlEnv) -> torch.Tensor:
  """DeployDRCurriculum 의 현재 level ∈ [0,1]. [B, 1].

  curriculum 이 env._deploy_dr_level 에 기록 (없으면 0). DR 강도를 critic 에 노출.
  """
  lvl = float(getattr(env, "_deploy_dr_level", 0.0))
  return torch.full((env.num_envs, 1), lvl, device=env.device)


def last_push_xy(env: ManagerBasedRlEnv) -> torch.Tensor:
  """마지막 push 이벤트의 수평 속도 외란 (x, y). [B, 2].

  push_by_setting_velocity_logged 가 env._last_push_xy 에 기록 (없으면 0).
  """
  buf = getattr(env, "_last_push_xy", None)
  if buf is None:
    return torch.zeros((env.num_envs, 2), device=env.device)
  return buf


def deploy_delay_steps(env: ManagerBasedRlEnv) -> torch.Tensor:
  """현재 적용 중인 action delay step. [B, 1]. curriculum 이 env 에 기록."""
  v = float(getattr(env, "_deploy_delay_steps", 0.0))
  return torch.full((env.num_envs, 1), v, device=env.device)


def deploy_action_noise_std(env: ManagerBasedRlEnv) -> torch.Tensor:
  """현재 action noise std. [B, 1]. curriculum 이 env 에 기록."""
  v = float(getattr(env, "_deploy_action_noise_std", 0.0))
  return torch.full((env.num_envs, 1), v, device=env.device)

