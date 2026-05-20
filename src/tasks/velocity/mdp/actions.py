"""Delayed + noisy joint position action (deploy baseline 의
``DelayedNoisyJointPositionActionCfg`` 이식).

mjlab ``JointPositionAction`` 을 상속해 두 sim2real 채널을 추가:
  - action delay : raw action 을 길이 (delay_max+1) ring buffer 에 저장하고 매 step
                   ``delay_steps`` 만큼 과거 action 을 적용 (env 별 정수 텐서).
  - action noise : 적용 직전 (delay 꺼낸) action 에 gaussian(0, noise_std) 추가.

적용 순서: delay(과거 raw 꺼냄) → noise → PD 변환(scale/offset). 참조 레포와 동일.

delay_steps / noise_std 는 DeployDRCurriculum 이 level 에 따라 런타임에 set 한다
(set_delay_steps / set_noise_std). level=0 이면 delay=0, noise=0 → 기본
JointPositionAction 과 수치적으로 동일 (NoDR 회귀).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions.actions import (
  JointPositionAction,
  JointPositionActionCfg,
)

__all__ = (
  "DelayedNoisyJointPositionAction",
  "DelayedNoisyJointPositionActionCfg",
)


class DelayedNoisyJointPositionAction(JointPositionAction):
  def __init__(self, cfg, env):
    super().__init__(cfg=cfg, env=env)
    self.delay_max_steps = int(cfg.delay_max_steps)
    self.noise_std_max = float(cfg.noise_std_max)

    # ring buffer of RAW actions. raw 0 == default pose (processed = offset =
    # default_joint_pos) 이므로 0 초기화는 "default action 으로 채움" 과 동일 →
    # 첫 step 튐 없음.
    self._action_buf = torch.zeros(
      self.num_envs, self.delay_max_steps + 1, self.action_dim, device=self.device
    )
    # env 별 delay step (정수). curriculum 이 set, 기본 0.
    self._delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._noise_std = 0.0
    self._env_arange = torch.arange(self.num_envs, device=self.device)

  # curriculum 연동용 setter (obs noise / push range set 과 동일 패턴).
  def set_delay_steps(self, steps: int) -> None:
    self._delay_steps.fill_(max(0, min(self.delay_max_steps, int(steps))))

  def set_noise_std(self, std: float) -> None:
    self._noise_std = max(0.0, float(std))

  def process_actions(self, actions: torch.Tensor) -> None:
    # last_action obs 는 정책 raw 출력을 보도록 유지.
    self._raw_actions[:] = actions
    # ring buffer: 최신을 index 0 에, 과거를 뒤로 밀기.
    self._action_buf = torch.roll(self._action_buf, shifts=1, dims=1)
    self._action_buf[:, 0] = actions
    # delay: env 별 delay_steps 만큼 과거 raw 꺼냄 (0 이면 방금 넣은 최신).
    delayed = self._action_buf[self._env_arange, self._delay_steps]
    # noise: PD 변환 전에 추가.
    if self._noise_std > 0.0:
      delayed = delayed + torch.randn_like(delayed) * self._noise_std
    # PD 변환.
    self._processed_actions = delayed * self._scale + self._offset

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    idx = slice(None) if env_ids is None else env_ids
    self._action_buf[idx] = 0.0  # raw 0 = default pose.


@dataclass(kw_only=True)
class DelayedNoisyJointPositionActionCfg(JointPositionActionCfg):
  delay_max_steps: int = 1
  """최대 action delay (step). 참조 레포 스펙 1-step."""
  noise_std_max: float = 0.1
  """최대 action noise std (raw action 공간)."""

  def build(self, env) -> DelayedNoisyJointPositionAction:
    return DelayedNoisyJointPositionAction(self, env)
