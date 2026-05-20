import os

import torch
import wandb

from mjlab.entity import Entity
from mjlab.envs.mdp.actions import JointPositionAction, JointVelocityAction
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner


def _get_metadata_any_joint_action(env, run_path: str) -> dict:
  """exporter_utils.get_base_metadata의 velocity-action 호환 버전.

  상위 util은 action term 이름을 'joint_pos'로 하드코딩하고 JointPositionAction만
  허용한다. velocity action 학습에서도 동일 메타데이터(스케일/강성/감쇠 등)를
  export하기 위해 joint_pos / joint_vel 어느 쪽이든 처리.
  """
  terms = dict(env.action_manager._terms)  # dict[name, BaseAction]
  joint_action = None
  for name in ("joint_pos", "joint_vel"):
    if name in terms and isinstance(
      terms[name], (JointPositionAction, JointVelocityAction)
    ):
      joint_action = terms[name]
      break
  if joint_action is None:
    # 등록된 joint action이 없으면 상위 유틸로 AssertionError를 터뜨려서
    # 기존 동작과 동일한 진단 메시지를 유지.
    return get_base_metadata(env, run_path)

  robot: Entity = env.scene["robot"]
  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names
    if jname in joint_name_to_ctrl_id
  ]
  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
  scale = joint_action._scale
  return {
    "run_path": run_path,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": env.observation_manager.active_terms["actor"],
    "action_type": "joint_vel"
    if isinstance(joint_action, JointVelocityAction)
    else "joint_pos",
    "action_scale": scale[0].cpu().tolist()
    if isinstance(scale, torch.Tensor)
    else scale,
  }


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = _get_metadata_any_joint_action(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class VelocityFloorClippedRunner(VelocityOnPolicyRunner):
  # PGTT joystick_base.py:220 mirror — reward = clip(sum(rewards) * dt, 0, 1e4).
  # Per-term `_episode_sums` (logging) stay unclipped; only the aggregate
  # returned to PPO is floored at 0 so negative-dominant steps don't bleed into
  # the advantage estimate.
  def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
    super().__init__(env, train_cfg, log_dir, device)
    rm = self.env.unwrapped.reward_manager
    _orig_compute = rm.compute

    def _floor_clipped_compute(dt):
      return torch.clamp(_orig_compute(dt), min=0.0, max=10000.0)

    rm.compute = _floor_clipped_compute
