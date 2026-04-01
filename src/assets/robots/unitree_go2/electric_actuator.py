"""Electric motor actuator for Unitree Go2.

전류 제어 (current-controlled) 모터 모델:
  실제 전동기에서 드라이브 전자회로는 전압 피드포워드로 back-EMF를 보상한다:
    V_cmd = R × I_des + Ke × ω   (내부 보상)
  따라서 전기 ODE는 back-EMF 항이 상쇄되어 단순화된다:
    dI/dt = (V_cmd - R×I - Ke×ω) / L
           = (R×I_des + Ke×ω - R×I - Ke×ω) / L
           = (I_des - I) × R/L
           = (I_des - I) / τ_e

흐름:
  [physics step, 5ms마다] compute(cmd) 호출 (action_manager가 매 step 호출)
    PD + velocity saturation → τ_des
    I_des = τ_des / (Kt × gr)   ← 토크 역산
    sub-step 적분 (dt_sub ≈ 0.1ms):
      dI/dt = (I_des - I) / τ_e   ← 같은 step 안에서 즉시 적분
    return τ = Kt × I_new × gr   ← 갱신된 I로 출력 (지연 없음)

  [physics step 이후] update(dt) 호출
    ODE는 compute()에서 완료됨. 별도 처리 없음.

시정수:
  τ_e = L / R = 1e-4 / 0.3 ≈ 0.33 ms
  Euler 안정 조건: dt_sub < τ_e
  physics 주기(5ms) → n_sub = 45 → dt_sub ≈ 0.11 ms  (τ_e/3, 안전)

효과:
  I_des → ODE → I_new → tau_out 이 한 step 안에서 완결.
  1-step(5ms) 지연 제거: tau_applied ≈ tau_des (같은 step 기준).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


@dataclass(kw_only=True)
class ElectricMotorActuatorCfg(DcMotorActuatorCfg):
    """PD + 전류-토크 ODE 액추에이터 설정.

    DcMotorActuatorCfg (stiffness, damping, effort_limit,
    saturation_effort, velocity_limit)를 상속하고 전기모터 파라미터 추가.
    """

    Kt: float = 0.128       # 토크 상수      [N·m/A]
    Ke: float = 0.128       # 역기전력 상수  [V·s/rad]
    R: float  = 0.3         # 권선 저항      [Ω]
    L: float  = 1e-4        # 권선 인덕턴스  [H]
    gear_ratio: float = 6.33  # 감속비

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> ElectricMotorActuator:
        return ElectricMotorActuator(self, entity, target_ids, target_names)


class ElectricMotorActuator(DcMotorActuator):
    """PD + 전류-토크 ODE 액추에이터.

    내부 상태:
      I      : 전류 [num_envs, num_joints]
      _I_des : 목표 전류 (policy step마다 갱신)
    """

    def __init__(
        self,
        cfg: ElectricMotorActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self.I:      torch.Tensor | None = None
        self._I_des: torch.Tensor | None = None
        self._n_sub: int   = 50
        self._dt_sub: float = 0.0

        # 로깅 상태 (start_logging() 호출 시 활성화)
        self._log: dict[str, list] | None = None
        self._log_physics_step: int = 0

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)

        cfg = self.cfg
        assert isinstance(cfg, ElectricMotorActuatorCfg)

        num_envs   = data.nworld
        num_joints = len(self._target_names)

        self.I      = torch.zeros(num_envs, num_joints, dtype=torch.float, device=device)
        self._I_des = torch.zeros(num_envs, num_joints, dtype=torch.float, device=device)

        # physics step dt = timestep × 1 (update는 매 step마다 호출)
        # MjModel에서 실제 timestep 읽어서 n_sub 결정
        physics_dt = float(mj_model.opt.timestep)   # 보통 0.005s
        tau_e      = cfg.L / cfg.R                   # ≈ 0.33ms

        # Euler 안정 조건: dt_sub < τ_e
        # safety factor 1/3 → dt_sub ≈ τ_e/3
        self._n_sub  = max(10, int(physics_dt / (tau_e / 3.0)))
        self._dt_sub = physics_dt / self._n_sub

    # ── 로깅 API ─────────────────────────────────────────────────────────────

    def start_logging(self) -> None:
        """physics step마다 전류/토크 데이터 수집 시작 (env 0만 기록)."""
        self._log = {
            "physics_step": [],   # 절대 physics step 번호
            "pos_target":   [],   # [num_joints] 관절 위치 목표
            "pos":          [],   # [num_joints] 현재 관절 위치
            "vel":          [],   # [num_joints] 현재 관절 속도
            "tau_des":      [],   # [num_joints] PD가 원하는 토크
            "tau_applied":  [],   # [num_joints] 실제 인가 토크 (I 기반, 1-step 지연)
            "I_des":        [],   # [num_joints] 목표 전류
            "I_before":     [],   # [num_joints] ODE 적분 전 전류
            "I_after":      [],   # [num_joints] ODE 적분 후 전류
            "I_substep":    [],   # [n_sub, num_joints] sub-step별 전류 (과도상태 시각화)
            "tau_substep":  [],   # [n_sub, num_joints] sub-step별 토크 = Kt * I_sub * gr
        }
        self._log_physics_step = 0

    def get_log(self) -> dict[str, list]:
        """수집된 로그 반환 (start_logging() 없이 호출하면 빈 dict)."""
        return self._log or {}

    def stop_logging(self) -> None:
        """로깅 중단 (로그 데이터는 get_log()로 계속 접근 가능)."""
        pass  # _log를 유지한 채 새 데이터만 추가하지 않으려면 별도 플래그 사용

    # ── physics step마다 호출 (action_manager가 매 step 호출) ────────────────

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        """PD → I_des → ODE 적분 → 갱신된 I로 τ 출력 (지연 없음).

        순서:
          1. PD로 tau_des 계산
          2. I_des = tau_des / (Kt × gr)
          3. ODE sub-step 적분: I → I_new  (지연 제거 핵심)
          4. tau_out = Kt × I_new × gr
        """
        assert self.I      is not None
        assert self._I_des is not None
        cfg = self.cfg
        assert isinstance(cfg, ElectricMotorActuatorCfg)

        # 1. PD + velocity saturation 토크
        tau_des = super().compute(cmd)   # [num_envs, num_joints]

        # 2. 토크 역산 → 목표 전류
        self._I_des.copy_(tau_des / (cfg.Kt * cfg.gear_ratio))

        # 로깅: ODE 적분 전 전류 (env 0만)
        if self._log is not None:
            self._log["physics_step"].append(self._log_physics_step)
            self._log["pos_target"].append(cmd.position_target[0].cpu().numpy().copy())
            self._log["pos"].append(cmd.pos[0].cpu().numpy().copy())
            self._log["vel"].append(cmd.vel[0].cpu().numpy().copy())
            self._log["tau_des"].append(tau_des[0].cpu().numpy().copy())
            self._log["I_des"].append(self._I_des[0].cpu().numpy().copy())
            self._log["I_before"].append(self.I[0].cpu().numpy().copy())

        # 3. ODE sub-step 적분: I_des → I  (같은 step 안에서 갱신)
        tau_e = cfg.L / cfg.R   # ≈ 0.33ms
        I = self.I
        log_substeps = self._log is not None
        substep_buf_I:   list = []
        substep_buf_tau: list = []
        for _ in range(self._n_sub):
            dI = (self._I_des - I) / tau_e
            I  = I + dI * self._dt_sub
            if log_substeps:
                I_sub = I[0].cpu().numpy().copy()
                substep_buf_I.append(I_sub)
                substep_buf_tau.append(
                    np.clip(I_sub * cfg.Kt * cfg.gear_ratio, -cfg.effort_limit, cfg.effort_limit)
                )

        I_max = cfg.effort_limit / (cfg.Kt * cfg.gear_ratio)
        self.I = torch.clamp(I, -I_max, I_max)

        # 4. 갱신된 I로 출력 토크 계산 (지연 없음)
        tau_out = torch.clamp(cfg.Kt * self.I * cfg.gear_ratio, -cfg.effort_limit, cfg.effort_limit)

        # 로깅: ODE 적분 후 전류 및 실제 인가 토크
        if self._log is not None:
            self._log["tau_applied"].append(tau_out[0].cpu().numpy().copy())
            self._log["I_after"].append(self.I[0].cpu().numpy().copy())
            self._log["I_substep"].append(np.stack(substep_buf_I))      # (n_sub, num_joints)
            self._log["tau_substep"].append(np.stack(substep_buf_tau))  # (n_sub, num_joints)
            self._log_physics_step += 1

        return tau_out

    # ── physics step마다 호출 ────────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """ODE는 compute()에서 완료됨. 별도 처리 없음."""
        pass

    # ── episode 리셋 ─────────────────────────────────────────────────────────

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        assert self.I      is not None
        assert self._I_des is not None
        if env_ids is None:
            self.I[:]      = 0.0
            self._I_des[:] = 0.0
        else:
            self.I[env_ids]      = 0.0
            self._I_des[env_ids] = 0.0
