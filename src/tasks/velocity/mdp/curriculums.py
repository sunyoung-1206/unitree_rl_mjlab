from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .velocity_command import UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_SCENE_CFG = SceneEntityCfg("robot")


class VelocityStage(TypedDict):
  step: int
  lin_vel_x: tuple[float, float] | None
  lin_vel_y: tuple[float, float] | None
  ang_vel_z: tuple[float, float] | None


class RewardWeightStage(TypedDict):
  step: int
  weight: float


def terrain_levels_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_SCENE_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]

  terrain = env.scene.terrain
  assert terrain is not None
  terrain_generator = terrain.cfg.terrain_generator
  assert terrain_generator is not None

  command = env.command_manager.get_command(command_name)
  assert command is not None

  # Compute the distance the robot walked.
  distance = torch.norm(
    asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
  )

  # Robots that walked far enough progress to harder terrains.
  move_up = distance > terrain_generator.size[0] / 2

  # Robots that walked less than half of their required distance go to simpler
  # terrains.
  move_down = (
    distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
  )
  move_down *= ~move_up

  # Update terrain levels.
  terrain.update_env_origins(env_ids, move_up, move_down)

  return torch.mean(terrain.terrain_levels.float())


def commands_vel(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  velocity_stages: list[VelocityStage],
) -> dict[str, torch.Tensor]:
  del env_ids  # Unused.
  command_term = env.command_manager.get_term(command_name)
  assert command_term is not None
  cfg = cast(UniformVelocityCommandCfg, command_term.cfg)
  for stage in velocity_stages:
    if env.common_step_counter > stage["step"]:
      if "lin_vel_x" in stage and stage["lin_vel_x"] is not None:
        cfg.ranges.lin_vel_x = stage["lin_vel_x"]
      if "lin_vel_y" in stage and stage["lin_vel_y"] is not None:
        cfg.ranges.lin_vel_y = stage["lin_vel_y"]
      if "ang_vel_z" in stage and stage["ang_vel_z"] is not None:
        cfg.ranges.ang_vel_z = stage["ang_vel_z"]
  return {
    # "lin_vel_x_min": torch.tensor(cfg.ranges.lin_vel_x[0]),
    # "lin_vel_x_max": torch.tensor(cfg.ranges.lin_vel_x[1]),
    # "lin_vel_y_min": torch.tensor(cfg.ranges.lin_vel_y[0]),
    # "lin_vel_y_max": torch.tensor(cfg.ranges.lin_vel_y[1]),
    # "ang_vel_z_min": torch.tensor(cfg.ranges.ang_vel_z[0]),
    # "ang_vel_z_max": torch.tensor(cfg.ranges.ang_vel_z[1]),
  }


def reward_weight(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  reward_name: str,
  weight_stages: list[RewardWeightStage],
) -> torch.Tensor:
  """Update a reward term's weight based on training step stages."""
  del env_ids  # Unused.
  reward_term_cfg = env.reward_manager.get_term_cfg(reward_name)
  for stage in weight_stages:
    if env.common_step_counter > stage["step"]:
      reward_term_cfg.weight = stage["weight"]
  return torch.tensor([reward_term_cfg.weight])


class DeployDRCurriculum:
  """단일 스칼라 level ∈ [level_min, level_max] 로 DR 강도를 reward EMA 에 따라
  자동 조정 (deploy baseline 의 reward-driven curriculum 이식).

  level 이 다음에 곱해진다:
    - obs noise scale     : base_noise × (max_obs_noise_scale × level)
    - push velocity range : ±(max_push × level)  [x, y 수평]

  level up   : timeout_ema ≥ timeout_up AND tracking_ema ≥ tracking_up AND
               fall_ema ≤ fall_up  가 up_count reset 연속.
  level down : fall_ema ≥ fall_down 이 down_count reset 연속.
  step       : up +up_step, down -down_step (비대칭). 변경 후 cooldown reset 동안 동결.

  CurrTerm 으로 등록하면 reset 마다 __call__(env, env_ids) 가 호출되고,
  반환 dict 가 Curriculum/deploy_dr/{level,tracking_ema,fall_rate_ema,
  timeout_rate_ema} 로 로깅된다.
  """

  def __init__(self, cfg, env: "ManagerBasedRlEnv"):
    self._env = env
    p = dict(getattr(cfg, "params", {}) or {})
    self.alpha = p.get("ema_alpha", 0.03)
    self.level = p.get("level_init", 0.1)
    self.level_min = p.get("level_min", 0.0)
    self.level_max = p.get("level_max", 1.0)
    self.up_step = p.get("up_step", 0.01)
    self.down_step = p.get("down_step", 0.03)
    self.up_count = p.get("up_count", 4)
    self.down_count = p.get("down_count", 2)
    self.cooldown_n = p.get("cooldown", 5)
    self.timeout_up = p.get("timeout_up", 0.80)
    self.tracking_up = p.get("tracking_up", 0.75)
    self.fall_up = p.get("fall_up", 0.15)
    self.fall_down = p.get("fall_down", 0.25)
    self.max_obs_noise_scale = p.get("max_obs_noise_scale", 1.0)
    self.max_push = p.get("max_push", 0.5)
    self.tracking_terms = tuple(
      p.get("tracking_terms", ("track_linear_velocity", "track_angular_velocity"))
    )
    self.push_term = p.get("push_term", "push_robot")

    # EMA / streak 상태.
    self.tracking_ema = 0.0
    self.timeout_ema = 0.0
    self.fall_ema = 0.0
    self._up_streak = 0
    self._down_streak = 0
    self._cooldown = 0

    # base obs noise 스냅샷 (actor group 만 corruption 적용됨). 스케일 전 원본 보존.
    self._noise_terms = []  # (term_cfg, operation, base_min, base_max)
    om = env.observation_manager
    actor_cfgs = getattr(om, "_group_obs_term_cfgs", {}).get("actor", [])
    for tc in actor_cfgs:
      nz = getattr(tc, "noise", None)
      if nz is not None and hasattr(nz, "n_min") and isinstance(nz.n_min, (int, float)):
        self._noise_terms.append((tc, nz.operation, float(nz.n_min), float(nz.n_max)))

    # base push event cfg (interval mode). 없으면 None.
    self._push_cfg = None
    try:
      self._push_cfg = env.event_manager.get_term_cfg(self.push_term)
    except Exception:
      self._push_cfg = None

    # 초기 level 즉시 적용 (학습 시작 시 DR 거의 꺼진 상태).
    self._apply_level()

  def reset(self, env_ids=None):
    # level / EMA 는 학습 전체에 걸쳐 누적되는 전역 상태 → reset 시 초기화하지 않음.
    del env_ids

  def _apply_level(self):
    from mjlab.utils.noise import UniformNoiseCfg

    # apply() 가 _tensor_cache 로 n_min/n_max 를 캐싱하므로, 단순 mutation 은
    # 무효 → 매 적용마다 noise 객체를 새로 만들어 캐시를 비운다.
    s = self.max_obs_noise_scale * self.level
    for tc, op, bmin, bmax in self._noise_terms:
      lo, hi = bmin * s, bmax * s
      # level≈0 이면 lo==hi==0 → UniformNoiseCfg 가 n_min<n_max 위배로 에러.
      # 이 경우 noise 자체를 끈다 (None). level 이 다시 오르면 복원됨.
      tc.noise = UniformNoiseCfg(operation=op, n_min=lo, n_max=hi) if lo < hi else None
    if self._push_cfg is not None:
      pv = self.max_push * self.level
      self._push_cfg.params["velocity_range"] = {"x": (-pv, pv), "y": (-pv, pv)}
    # Phase 5 critic obs (deploy_curriculum_level) 가 읽도록 env 에 노출.
    self._env._deploy_dr_level = self.level

  def _state_dict(self):
    return {
      "level": torch.tensor(self.level),
      "tracking_ema": torch.tensor(self.tracking_ema),
      "fall_rate_ema": torch.tensor(self.fall_ema),
      "timeout_rate_ema": torch.tensor(self.timeout_ema),
    }

  def __call__(self, env: "ManagerBasedRlEnv", env_ids, **kwargs):
    del kwargs
    # 전체 reset(slice) 이거나 빈 env_ids 면 metric 갱신 없이 현재 상태만 보고.
    if not isinstance(env_ids, torch.Tensor) or env_ids.numel() == 0:
      return self._state_dict()

    n = env_ids.numel()
    dt = env.step_dt
    ep_len = env.episode_length_buf[env_ids].clamp(min=1).float()

    # tracking: 각 tracking reward 의 episode_sum 을 (weight × dt × ep_len) 로
    # 정규화 → step 평균 exp 값 [0,1]. lin/ang 평균.
    rm = env.reward_manager
    track = torch.zeros(n, device=env.device)
    cnt = 0
    for name in self.tracking_terms:
      try:
        w = rm.get_term_cfg(name).weight
        ep_sum = rm._episode_sums[name][env_ids]
        track = track + ep_sum / (w * dt * ep_len)
        cnt += 1
      except (KeyError, ValueError, AttributeError):
        pass
    if cnt > 0:
      track = track / cnt
    batch_track = float(track.mean().item())
    batch_timeout = float(env.termination_manager.time_outs[env_ids].float().mean().item())
    batch_fall = float(env.termination_manager.terminated[env_ids].float().mean().item())

    a = self.alpha
    self.tracking_ema = (1 - a) * self.tracking_ema + a * batch_track
    self.timeout_ema = (1 - a) * self.timeout_ema + a * batch_timeout
    self.fall_ema = (1 - a) * self.fall_ema + a * batch_fall

    changed = False
    if self._cooldown > 0:
      self._cooldown -= 1
    else:
      up_ok = (
        self.timeout_ema >= self.timeout_up
        and self.tracking_ema >= self.tracking_up
        and self.fall_ema <= self.fall_up
      )
      down_ok = self.fall_ema >= self.fall_down
      if up_ok:
        self._up_streak += 1
        self._down_streak = 0
        if self._up_streak >= self.up_count:
          self.level = min(self.level_max, self.level + self.up_step)
          self._up_streak = 0
          self._cooldown = self.cooldown_n
          changed = True
      elif down_ok:
        self._down_streak += 1
        self._up_streak = 0
        if self._down_streak >= self.down_count:
          self.level = max(self.level_min, self.level - self.down_step)
          self._down_streak = 0
          self._cooldown = self.cooldown_n
          changed = True
      else:
        self._up_streak = 0
        self._down_streak = 0

    if changed:
      self._apply_level()

    return self._state_dict()
