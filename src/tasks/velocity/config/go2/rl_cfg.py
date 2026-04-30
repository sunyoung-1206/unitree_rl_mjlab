"""RL configuration for Unitree Go2 velocity task."""

from dataclasses import replace
from typing import Literal

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_go2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree Go2 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2_velocity",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_go2_methoda_electric_ppo_runner_cfg(
  action_type: Literal["position", "velocity"] = "position",
) -> RslRlOnPolicyRunnerCfg:
  """MethodA-Electric 전용 러너 설정.

  실험 로그 경로가 action type과 주요 타이밍 하이퍼를 한눈에 보여주도록
  experiment_name / run_name 을 인코딩.

    experiment_name: "go2_methoda_electric"
    run_name       : "act-{pos|vel}_pdt20ms_phyDt0p1ms_tauDec4"

  decimation=200 (0.1ms × 200 = 20ms policy dt)이지만,
  tau 목표 업데이트 주기는 pd_substeps=50 → 20ms 안에서 4회 (= 표기상 tauDec4).
  """
  cfg = unitree_go2_ppo_runner_cfg()
  act_tag = "vel" if action_type == "velocity" else "pos"
  return replace(
    cfg,
    experiment_name="go2_methoda_electric",
    run_name=f"act-{act_tag}_pdt20ms_phyDt0p1ms_tauDec4",
  )


def unitree_go2_methodb_electric_ppo_runner_cfg(
  action_type: Literal["position", "velocity"] = "position",
) -> RslRlOnPolicyRunnerCfg:
  """MethodB-Electric 전용 러너 설정.

    experiment_name: "go2_methodb_electric"
    run_name       : "act-{pos|vel}_pdt20ms_phyDt0p1ms_tauDec4"
  """
  cfg = unitree_go2_ppo_runner_cfg()
  act_tag = "vel" if action_type == "velocity" else "pos"
  return replace(
    cfg,
    experiment_name="go2_methodb_electric",
    run_name=f"act-{act_tag}_pdt20ms_phyDt0p1ms_tauDec4",
  )
