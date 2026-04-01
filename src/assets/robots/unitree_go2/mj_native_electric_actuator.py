"""MuJoCo-native electric motor actuator via act/act_dot coupling.

MuJoCo의 activation state (d->act)를 전류 I로 사용하여,
전기 ODE를 MuJoCo 엔진 내부에서 기계 ODE와 동시에 적분한다.

┌─────────────────────────────────────────────────────────────────────┐
│  시스템 모델                                                       │
│                                                                     │
│  전기계: L·dI/dt = V − R·I − Ke·gr·ω                              │
│  기계계: τ_actuator = Kt·gr·I                                      │
│                                                                     │
│  MuJoCo state 매핑:                                                │
│    d->act[i]  = 전류 I  [A]                                        │
│    d->ctrl[i] = 제어 입력 (filterexact: 등가전류, user: 전압)       │
│    force      = gainprm[0] × act = Kt·gr × I                      │
│                                                                     │
│  ∂(act_dot)/∂(act) = −R/L  →  MuJoCo implicit 행렬에 포함          │
│  ∂(act_dot)/∂(qvel) = −Ke·gr/L  →  ctrl에 반영 (1-step lag)       │
└─────────────────────────────────────────────────────────────────────┘

신호 흐름 (매 physics step):

  Policy → τ_des
    │
    ├─ ① 전류 변환:  I_des = τ_des / (Kt·gr)
    │
    ├─ ② 가상 전압 제어기:  V = R·I_des + Ke·gr·ω   ← 역기전력 피드포워드
    │                        (I_des 추종에 필요한 물리 전압)
    │
    ├─ ③ filterexact 매핑:  ctrl = (V − Ke·gr·ω) / R
    │                             = I_des
    │
    └─ ④ MuJoCo 물리:
         dI/dt = (ctrl·R − R·I) / L           [filterexact 전개]
               = (V − Ke·gr·ω − R·I) / L      [ctrl 대입]
               = (R·I_des + Ke·gr·ω − Ke·gr·ω − R·I) / L
               = R·(I_des − I) / L             [back-EMF 상쇄 후]

         → I(t) = I_des·(1 − e^{−t/τ_e})   (τ_e = L/R ≈ 0.33ms)
         → τ = Kt·gr·I

  역기전력의 역할:
    • 전압 계산 (②): V에 Ke·gr·ω를 더해 back-EMF 보상
    • 물리 방정식 (④): 원래 전기 ODE에 −Ke·gr·ω 항 존재
    • 가상 제어기가 보상하므로 I → I_des 수렴이 ω에 무관
    • 전압 포화 시 (V > V_bus) 보상 불가 → 고속에서 자연스러운 토크 저하

=== Approach A: dyntype=filterexact (기본, 권장) ===

  MuJoCo 내장 1차 필터 (정확한 지수 적분):
    act_dot = (ctrl − act) / τ_e          where τ_e = L/R

  장점:
    • ∂act_dot/∂act = −R/L 해석적으로 implicit 행렬에 포함
    • mujoco_warp (GPU) 호환
    • C callback 불필요

=== Approach B: dyntype=user + callback (대안) ===

  callback returns: dI/dt = (V − R·I − Ke·gr·ω) / L

  한계:
    • MuJoCo가 user dynamics Jacobian을 0으로 가정
    • mujoco_warp 비호환
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

from mjlab.actuator.actuator import ActuatorCmd
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


# ═══════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════


@dataclass(kw_only=True)
class NativeElectricActuatorCfg(DcMotorActuatorCfg):
    """MuJoCo-native 전기모터 액추에이터 설정.

    MuJoCo의 act/act_dot 메커니즘으로 전류 I를 MuJoCo state에 통합.
    DcMotorActuatorCfg (PD + DC motor saturation)를 상속.
    """

    Kt: float = 0.128          # 토크 상수      [N·m/A]
    Ke: float = 0.128          # 역기전력 상수  [V·s/rad_motor]
    R:  float = 0.3            # 권선 저항      [Ω]
    L:  float = 1e-4           # 권선 인덕턴스  [H]
    gear_ratio: float = 6.33   # 감속비
    V_bus: float = float("inf")
    """버스 전압 한계 [V].  가상 제어기 출력 V를 ±V_bus로 클램프.
    inf이면 무제한 (I가 I_des를 항상 추종).
    유한값이면 고속에서 back-EMF가 V_bus를 초과 → 자연스러운 토크 저하."""

    substeps: int = 1
    """Physics sub-steps per policy step (= decimation).
    1이면 서브스테핑 없음 (매 호출마다 PD 재계산).
    >1이면 ZOH: 첫 서브스텝에서 τ_des → I_des 캐시,
    이후 서브스텝에서는 최신 ω로 전압만 갱신."""

    use_callback: bool = False
    """True: dyntype=user (Python callback, standard mujoco only).
    False: dyntype=filterexact (recommended, mujoco_warp compatible)."""

    def build(
        self, entity: Entity, target_ids: list[int], target_names: list[str]
    ) -> NativeElectricActuator:
        return NativeElectricActuator(self, entity, target_ids, target_names)


# ═══════════════════════════════════════════════════════════════════════
#  User callback (Approach B)
# ═══════════════════════════════════════════════════════════════════════

_callback_installed: bool = False


def _act_dyn_callback(
    model: mujoco.MjModel, data: mujoco.MjData, actuator_id: int
) -> float:
    """mjcb_act_dyn callback: dI/dt for dyntype=user actuator.

    MuJoCo가 fwd_actuation() 중 호출.
    모터 파라미터는 dynprm에 저장:
      dynprm[0]=R, [1]=L, [2]=Ke, [3]=gr
    """
    if model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_USER:
        return 0.0

    prm = model.actuator_dynprm[actuator_id]
    R, L, Ke, gr = prm[0], prm[1], prm[2], prm[3]

    act_adr = model.actuator_actadr[actuator_id]
    I = data.act[act_adr]
    V_cmd = data.ctrl[actuator_id]

    joint_id = model.actuator_trnid[actuator_id, 0]
    dof_adr = model.jnt_dofadr[joint_id]
    omega = data.qvel[dof_adr]

    # L·dI/dt = V_cmd − R·I − Ke·gr·ω
    return (V_cmd - R * I - Ke * gr * omega) / L


def install_act_dyn_callback() -> None:
    """mjcb_act_dyn 콜백 등록 (standard mujoco, 1회만)."""
    global _callback_installed
    if not _callback_installed:
        mujoco.set_mjcb_act_dyn(_act_dyn_callback)
        _callback_installed = True


def uninstall_act_dyn_callback() -> None:
    """mjcb_act_dyn 콜백 해제."""
    global _callback_installed
    mujoco.set_mjcb_act_dyn(None)
    _callback_installed = False


# ═══════════════════════════════════════════════════════════════════════
#  Actuator
# ═══════════════════════════════════════════════════════════════════════


class NativeElectricActuator(DcMotorActuator):
    """MuJoCo-native 전기모터 액추에이터.

    MuJoCo의 act 필드를 전류 I로 사용하여 엔진 내부에서 전기 ODE를 적분.

    vs ElectricMotorActuator (기존):
      기존: Python에서 Backward Euler ODE → τ_out → data.ctrl (motor)
      신규: data.ctrl = 제어입력 → MuJoCo act filter/callback → force = Kt·gr·I

    핵심 차이:
      • I가 MuJoCo state (d->act)의 일부 → 체크포인트/리셋 자동 처리
      • filterexact: MuJoCo implicit solver에 ∂act_dot/∂act = −R/L 포함
      • force 계산이 MuJoCo 내부에서 처리 (gain × act)
    """

    def __init__(
        self,
        cfg: NativeElectricActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._act_adr: torch.Tensor | None = None
        self._mjw_data: mjwarp.Data | None = None

        # Precompute constants
        self._Ktgr = cfg.Kt * cfg.gear_ratio     # Kt·gr  [N·m/A → joint torque per amp]
        self._Kegr = cfg.Ke * cfg.gear_ratio      # Ke·gr  [V per joint rad/s]

        # Sub-stepping ZOH state
        self._sub_idx: int = 0                    # 현재 서브스텝 인덱스
        self._I_des_hold: torch.Tensor | None = None   # ZOH 캐시: I_des
        self._tau_des_hold: torch.Tensor | None = None  # ZOH 캐시: τ_des (로깅용)

        # Logging
        self._log: dict[str, list] | None = None
        self._log_step: int = 0

    # ── MjSpec 수정: actuator 타입 설정 ──────────────────────────────

    def edit_spec(
        self, spec: mujoco.MjSpec, target_names: list[str]
    ) -> None:
        """<general> actuator를 dyntype=filterexact 또는 user로 생성.

        기존 edit_spec() (motor 생성)을 완전히 대체.

        XML 등가:
          <general name="FR_hip" joint="FR_hip_joint"
            dyntype="filterexact" dynprm="3.333e-4"     (또는 user)
            gaintype="fixed" gainprm="0.81024"
            biastype="none"
            actrange="-29 29" actlimited="true"
            forcerange="-23.5 23.5" forcelimited="true"/>
        """
        cfg = self.cfg
        assert isinstance(cfg, NativeElectricActuatorCfg)

        I_max = cfg.effort_limit / self._Ktgr
        tau_e = cfg.L / cfg.R

        for target_name in target_names:
            act = spec.add_actuator(
                name=target_name, target=target_name
            )
            act.trntype = mujoco.mjtTrn.mjTRN_JOINT

            # ── Dynamics ─────────────────────────────────────────
            if cfg.use_callback:
                # Approach B: dyntype=user
                # dynprm에 모터 파라미터 저장 → callback이 읽음
                act.dyntype = mujoco.mjtDyn.mjDYN_USER
                act.dynprm[0] = cfg.R
                act.dynprm[1] = cfg.L
                act.dynprm[2] = cfg.Ke
                act.dynprm[3] = cfg.gear_ratio
                # ctrl = V_cmd (전압)
                V_max = cfg.R * I_max + cfg.Ke * cfg.gear_ratio * cfg.velocity_limit
                act.ctrllimited = True
                act.ctrlrange[:] = np.array([-V_max, V_max])
            else:
                # Approach A: dyntype=filterexact (권장)
                # act_dot = (ctrl − act) / τ_e
                # MuJoCo가 ∂act_dot/∂act = −1/τ_e = −R/L 해석적으로 포함
                act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
                act.dynprm[0] = tau_e
                # ctrl = 목표 전류 (PD + back-EMF)
                act.ctrllimited = True
                act.ctrlrange[:] = np.array([-I_max * 2, I_max * 2])

            # ── Gain: force = Kt·gr × I ─────────────────────────
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm[0] = self._Ktgr

            # ── No bias ──────────────────────────────────────────
            act.biastype = mujoco.mjtBias.mjBIAS_NONE

            # ── Limits ───────────────────────────────────────────
            act.actlimited = True
            act.actrange[:] = np.array([-I_max, I_max])
            act.forcelimited = True
            act.forcerange[:] = np.array(
                [-cfg.effort_limit, cfg.effort_limit]
            )

            # ── Joint properties ─────────────────────────────────
            joint = spec.joint(target_name)
            joint.armature = cfg.armature
            if cfg.frictionloss > 0:
                joint.frictionloss = cfg.frictionloss

            self._mjs_actuators.append(act)

    # ── 초기화 ───────────────────────────────────────────────────────

    def initialize(
        self,
        mj_model: mujoco.MjModel,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)

        cfg = self.cfg
        assert isinstance(cfg, NativeElectricActuatorCfg)

        # act 주소 매핑 (ctrl_id → act_adr)
        global_ids = self._global_ctrl_ids.cpu().tolist()
        act_addrs = [mj_model.actuator_actadr[gid] for gid in global_ids]
        self._act_adr = torch.tensor(
            act_addrs, dtype=torch.long, device=device
        )

        # mjwarp.Data 참조 저장 (logging용)
        self._mjw_data = data

        # user callback 등록 (Approach B)
        if cfg.use_callback:
            install_act_dyn_callback()

        # 검증
        dt = float(mj_model.opt.timestep)
        tau_e = cfg.L / cfg.R
        na = mj_model.na
        assert na > 0, (
            f"na={na}: dyntype이 올바르게 설정되지 않았습니다. "
            "edit_spec()에서 dyntype=filterexact 또는 user를 확인하세요."
        )
        if not cfg.use_callback:
            assert dt < tau_e * 3, (
                f"dt={dt*1e3:.3f}ms > 3·τ_e={tau_e*3*1e3:.3f}ms: "
                "filterexact 정확도 저하. timestep을 줄이세요."
            )

    # ── Logging ──────────────────────────────────────────────────────

    def start_logging(self) -> None:
        self._log = {
            "physics_step": [],
            "pos_target": [],
            "pos": [],
            "vel": [],
            "tau_des": [],
            "tau_applied": [],
            "I": [],
            "ctrl": [],
            "V": [],
            "back_emf": [],
        }
        self._log_step = 0

    def get_log(self) -> dict[str, list]:
        return self._log or {}

    def stop_logging(self) -> None:
        pass

    # ── compute: 매 physics sub-step 마다 호출 ────────────────────────

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        """서브스테핑 ZOH + 가상 전압 제어기 → filterexact ctrl.

        mjlab decimation 루프가 매 physics sub-step마다 이 메서드를 호출.

        ┌───────────────────────────────────────────────────────────┐
        │  Policy (5ms, 200Hz)                                     │
        │    │                                                      │
        │    ▼ position_target (ZOH)                               │
        │  ┌─────────────────────────────────────────────────────┐ │
        │  │  Sub-step 0 (첫 호출):                              │ │
        │  │    τ_des = PD(target, q₀, v₀)  ← 1회 계산, 캐시   │ │
        │  │    I_des = τ_des / (Kt·gr)     ← 캐시              │ │
        │  │                                                      │ │
        │  │  Sub-step k (매 호출):                              │ │
        │  │    ω_k ← MuJoCo 최신 각속도                        │ │
        │  │    V_k = R·I_des + Ke·gr·ω_k  ← back-EMF 갱신    │ │
        │  │    V_k = clamp(V_k, ±V_bus)                        │ │
        │  │    ctrl_k = (V_k − Ke·gr·ω_k) / R                 │ │
        │  │    → mj_step()                                      │ │
        │  │                                                      │ │
        │  │  Sub-step 49 (마지막):                              │ │
        │  │    → (q₅₀, v₅₀) Policy에 반환                      │ │
        │  └─────────────────────────────────────────────────────┘ │
        └───────────────────────────────────────────────────────────┘

        substeps=1이면 매 호출마다 PD 재계산 (서브스테핑 비활성).
        """
        cfg = self.cfg
        assert isinstance(cfg, NativeElectricActuatorCfg)

        # ── ①② τ_des → I_des (ZOH: 첫 서브스텝에서만 계산) ─────
        if cfg.substeps <= 1 or self._sub_idx == 0:
            # PD + DC motor saturation → τ_des
            tau_des = super().compute(cmd)
            I_des = tau_des / self._Ktgr
            # 서브스테핑 시 캐시 저장
            if cfg.substeps > 1:
                self._I_des_hold = I_des
                self._tau_des_hold = tau_des
        else:
            # 서브스텝 1..N-1: 캐시된 I_des 사용 (ZOH)
            I_des = self._I_des_hold
            tau_des = self._tau_des_hold

        # ── ③ 가상 전압 제어기 (매 서브스텝: 최신 ω 반영) ───────
        #    V = R·I_des + Ke·gr·ω
        #    - R·I_des : 저항 강하 보상      (ZOH — 고정)
        #    - Ke·gr·ω : 역기전력 보상       (매 서브스텝 갱신)
        omega = cmd.vel                              # [num_envs, num_joints]
        back_emf = self._Kegr * omega                # Ke·gr·ω  [V]
        V = cfg.R * I_des + back_emf                  # [V]

        # 버스 전압 포화
        V = torch.clamp(V, -cfg.V_bus, cfg.V_bus)

        # ── ④ filterexact / callback 입력 변환 ──────────────────
        if cfg.use_callback:
            ctrl = V
        else:
            ctrl = (V - back_emf) / cfg.R

        # ── 서브스텝 카운터 갱신 ────────────────────────────────
        if cfg.substeps > 1:
            self._sub_idx = (self._sub_idx + 1) % cfg.substeps

        # ── 로깅 (env 0, step 전) ───────────────────────────────
        if self._log is not None and self._mjw_data is not None:
            I_current = self._mjw_data.act[0, self._act_adr].cpu().numpy()
            self._log["physics_step"].append(self._log_step)
            self._log["pos_target"].append(
                cmd.position_target[0].cpu().numpy().copy()
            )
            self._log["pos"].append(cmd.pos[0].cpu().numpy().copy())
            self._log["vel"].append(omega[0].cpu().numpy().copy())
            self._log["tau_des"].append(tau_des[0].cpu().numpy().copy())
            self._log["tau_applied"].append(
                (self._Ktgr * I_current).copy()
            )
            self._log["I"].append(I_current.copy())
            self._log["ctrl"].append(ctrl[0].cpu().numpy().copy())
            self._log["V"].append(V[0].cpu().numpy().copy())
            self._log["back_emf"].append(
                back_emf[0].cpu().numpy().copy()
            )
            self._log_step += 1

        return ctrl

    # ── reset ────────────────────────────────────────────────────────

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """에피소드 리셋 시 전류(act) 및 서브스텝 카운터 초기화.

        mjlab은 policy step 경계에서 reset을 호출하므로
        _sub_idx = 0 리셋이 decimation 루프 시작과 일치.
        """
        if self._mjw_data is not None and self._act_adr is not None:
            if env_ids is None:
                self._mjw_data.act[:, self._act_adr] = 0.0
            else:
                # env_ids [N] × _act_adr [J] → broadcast [N, J]
                self._mjw_data.act[env_ids.unsqueeze(-1), self._act_adr] = 0.0
        # 서브스텝 카운터 리셋 → 다음 compute()에서 PD 재계산
        self._sub_idx = 0

    # ── update (no-op) ───────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """MuJoCo가 act 적분을 처리하므로 별도 업데이트 불필요."""
        pass
