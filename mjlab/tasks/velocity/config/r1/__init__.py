from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_r1_flat_env_cfg,
  unitree_r1_rough_env_cfg,
)
from .rl_cfg import unitree_r1_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Unitree-R1",
  env_cfg=unitree_r1_rough_env_cfg(),
  play_env_cfg=unitree_r1_rough_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Unitree-R1",
  env_cfg=unitree_r1_flat_env_cfg(),
  play_env_cfg=unitree_r1_flat_env_cfg(play=True),
  rl_cfg=unitree_r1_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
