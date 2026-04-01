"""Electric motor actuator for Unitree Go2.

전기-기계 연립 ODE (Backward Implicit Euler):
  상태 벡터: x = [I, ω]ᵀ

  L·dI/dt  =  V_cmd - R·I - Ke·gr·ω        (전기)
  J·dω/dt  =  Kt·gr·I - τ_load             (기계, MuJoCo가 담당)

  시스템 행렬 S = M - dt·A_f:
    S = [[L + dt·R,    dt·Ke·gr  ],
         [-dt·Kt·gr,   J         ]]
  det(S) = (L+dt·R)·J + dt²·Ke·Kt·gr²

흐름 (~0.11ms 마다):
  1. PD → τ_des → I_des = τ_des / (Kt·gr)
  2. V_cmd = R·I_des  (back-EMF 미보상 → ω가 dI/dt에 명시적으로 등장)
  3. CoupledElecMechSolver.step() → [I_new, ω_approx] 동시 계산
     - ω_approx: solver 내부의 ω 추정 (τ_load 근사값으로 계산)
     - I_new:    τ_out에 사용
  4. τ_out = Kt·I_new·gr → MuJoCo에 인가 → MuJoCo가 실제 ω 갱신

커플링 경로:
  I_new ──→ τ_out ──→ MuJoCo ──→ ω_new ──→ 다음 step의 V_cmd·back-EMF

시정수:
  τ_e = L/R ≈ 0.33ms
  physics timestep ≈ 0.111ms (τ_e/3),  decimation=180 → policy dt=20ms
  Backward Euler: 안정 조건 무조건 만족
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg

from src.assets.robots.unitree_go2.coupled_ode_solver import CoupledElecMechSolver

if TYPE_CHECKING:
    from mjlab.entity import Entity


@dataclass(kw_only=True)
class ElectricMotorActuatorCfg(DcMotorActuatorCfg):
    """PD + 전기-기계 연립 ODE 액추에이터 설정.

    DcMotorActuatorCfg (stiffness, damping, effort_limit,
    saturation_effort, velocity_limit)를 상속하고 전기·기계 파라미터 추가.
    """

    Kt: float = 0.128          # 토크 상수      [N·m/A]
    Ke: float = 0.128          # 역기전력 상수  [V·s/rad_motor]
    R:  float = 0.3            # 권선 저항      [Ω]
    L:  float = 1e-4           # 권선 인덕턴스  [H]
    gear_ratio: float = 6.33   # 감속비
    J:  float = 0.01           # joint-space 유효 관성 모멘트 [kg·m²]
                               # (회전자 관성 × gr² + 링크 관성 추정치)

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> ElectricMotorActuator:
        return ElectricMotorActuator(self, entity, target_ids, target_names)


class ElectricMotorActuator(DcMotorActuator):
    """PD + 전기-기계 연립 ODE 액추에이터.

    내부 상태:
      I      : 전류 [num_envs, num_joints]
      _I_des : 목표 전류
      _solver: CoupledElecMechSolver (backward Euler 2×2 시스템)
    """

    def __init__(
        self,
        cfg: ElectricMotorActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self.I:       torch.Tensor | None = None
        self._I_des:  torch.Tensor | None = None
        self._solver: CoupledElecMechSolver | None = None

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

        dt    = float(mj_model.opt.timestep)
        tau_e = cfg.L / cfg.R
        assert dt < tau_e, (
            f"physics timestep {dt*1e3:.3f}ms ≥ τ_e {tau_e*1e3:.3f}ms: "
            "Backward Euler도 정확도 저하. timestep을 줄이세요."
        )

        self._solver = CoupledElecMechSolver(
            L  = cfg.L,
            R  = cfg.R,
            Ke = cfg.Ke,
            Kt = cfg.Kt,
            J  = cfg.J,
            gr = cfg.gear_ratio,
            dt = dt,
        )

    # ── 로깅 API ─────────────────────────────────────────────────────────────

    def start_logging(self) -> None:
        """physics step마다 전류/토크 데이터 수집 시작 (env 0만 기록)."""
        self._log = {
            "physics_step": [],
            "pos_target":   [],
            "pos":          [],
            "vel":          [],   # ω_joint (MuJoCo 갱신값)
            "tau_des":      [],
            "tau_applied":  [],
            "I_des":        [],
            "I_before":     [],
            "I_after":      [],
            "back_emf":     [],   # Ke·gr·ω_joint (전류 억제 항)
            "omega_approx": [],   # solver 내부 ω 추정 (τ_load ≈ 0 기준)
        }
        self._log_physics_step = 0

    def get_log(self) -> dict[str, list]:
        return self._log or {}

    def stop_logging(self) -> None:
        pass

    # ── physics step마다 호출 ────────────────────────────────────────────────

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        """PD → I_des → CoupledElecMechSolver → τ 출력.

        핵심:
          solver.step()이 2×2 시스템 행렬 S를 역산하여
          I와 ω를 동시에 업데이트한다.
          I_new는 τ_out에 사용, ω_approx는 로깅용.
          실제 ω는 MuJoCo가 τ_out으로부터 업데이트한다.
        """
        assert self.I      is not None
        assert self._I_des is not None
        assert self._solver is not None
        cfg = self.cfg
        assert isinstance(cfg, ElectricMotorActuatorCfg)

        # 1. PD + velocity saturation 토크
        tau_des = super().compute(cmd)   # [num_envs, num_joints]

        # 2. 토크 역산 → 목표 전류
        self._I_des.copy_(tau_des / (cfg.Kt * cfg.gear_ratio))

        # 3. Backward Euler 연립 ODE 1-step
        #    V_cmd = R·I_des  (back-EMF 미보상 → ω가 시스템 행렬에 명시적으로 반영)
        #    τ_load ≈ 0 (MuJoCo가 실제 기계 ODE를 담당하므로 solver의 ω는 근사)
        omega_joint = cmd.vel                              # MuJoCo 갱신 ω [num_envs, num_joints]
        V_cmd       = cfg.R * self._I_des
        tau_load    = torch.zeros_like(omega_joint)        # 근사: MuJoCo에 위임

        I_new, omega_approx = self._solver.step(
            I        = self.I,
            omega    = omega_joint,
            V_cmd    = V_cmd,
            tau_load = tau_load,
        )

        I_max  = cfg.effort_limit / (cfg.Kt * cfg.gear_ratio)
        I_new  = torch.clamp(I_new, -I_max, I_max)

        # 로깅
        if self._log is not None:
            back_emf = cfg.Ke * cfg.gear_ratio * omega_joint
            self._log["physics_step"].append(self._log_physics_step)
            self._log["pos_target"].append(cmd.position_target[0].cpu().numpy().copy())
            self._log["pos"].append(cmd.pos[0].cpu().numpy().copy())
            self._log["vel"].append(omega_joint[0].cpu().numpy().copy())
            self._log["tau_des"].append(tau_des[0].cpu().numpy().copy())
            self._log["I_des"].append(self._I_des[0].cpu().numpy().copy())
            self._log["I_before"].append(self.I[0].cpu().numpy().copy())
            self._log["back_emf"].append(back_emf[0].cpu().numpy().copy())

        self.I = I_new

        # 4. 출력 토크
        tau_out = torch.clamp(
            cfg.Kt * self.I * cfg.gear_ratio,
            -cfg.effort_limit,
            cfg.effort_limit,
        )

        if self._log is not None:
            self._log["tau_applied"].append(tau_out[0].cpu().numpy().copy())
            self._log["I_after"].append(self.I[0].cpu().numpy().copy())
            self._log["omega_approx"].append(omega_approx[0].cpu().numpy().copy())
            self._log_physics_step += 1

        return tau_out

    # ── physics step 이후 호출 ───────────────────────────────────────────────

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
