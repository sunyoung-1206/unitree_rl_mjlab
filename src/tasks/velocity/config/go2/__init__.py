from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_go2_flat_env_cfg,
  unitree_go2_flat_electric_env_cfg,
  unitree_go2_flat_native_electric_env_cfg,
  unitree_go2_flat_coupled_electric_env_cfg,
  unitree_go2_flat_methoda_electric_env_cfg,
  unitree_go2_rough_env_cfg,
)
from .rl_cfg import (
  unitree_go2_methoda_electric_ppo_runner_cfg,
  unitree_go2_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-Go2-Rough",
  env_cfg=unitree_go2_rough_env_cfg(),
  play_env_cfg=unitree_go2_rough_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

def _go2_flat_pd_cfg(play: bool = False):
  cfg = unitree_go2_flat_env_cfg(play=play)
  cfg.sim.mujoco.timestep = 0.005  # 5ms
  cfg.decimation = 4               # policy dt = 20ms (5ms × 4)
  return cfg

register_mjlab_task(
  task_id="Unitree-Go2-Flat",
  env_cfg=_go2_flat_pd_cfg(),
  play_env_cfg=_go2_flat_pd_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat-Electric",
  env_cfg=unitree_go2_flat_electric_env_cfg(),
  play_env_cfg=unitree_go2_flat_electric_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat-Native-Electric",
  env_cfg=unitree_go2_flat_native_electric_env_cfg(),
  play_env_cfg=unitree_go2_flat_native_electric_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2-Flat-Coupled-Electric",
  env_cfg=unitree_go2_flat_coupled_electric_env_cfg(),
  play_env_cfg=unitree_go2_flat_coupled_electric_env_cfg(play=True),
  rl_cfg=unitree_go2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

# Action type 토글: 여기서 "position" ↔ "velocity" 만 바꾸면 됨.
# env_cfg의 use_velocity_action 과 rl_cfg의 action_type 을 같은 값으로 맞출 것.
_METHODA_ACTION_TYPE = "position"  # or "velocity"
_methoda_use_vel = _METHODA_ACTION_TYPE == "velocity"

register_mjlab_task(
  task_id="Unitree-Go2-Flat-MethodA-Electric",
  env_cfg=unitree_go2_flat_methoda_electric_env_cfg(
    use_velocity_action=_methoda_use_vel,
  ),
  play_env_cfg=unitree_go2_flat_methoda_electric_env_cfg(
    play=True, use_velocity_action=_methoda_use_vel,
  ),
  rl_cfg=unitree_go2_methoda_electric_ppo_runner_cfg(
    action_type=_METHODA_ACTION_TYPE,
  ),
  runner_cls=VelocityOnPolicyRunner,
)
