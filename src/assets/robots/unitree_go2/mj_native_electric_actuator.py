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
from typing import TYPE_CHECKING, Literal

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch

# Method identifier → dynprm[4] selector for patched mjwarp.
# A   : β_int = β_imp = 1/(1+h/τ)             — fully BE (integrator + Schur + Force RHS).
# A+  : β_int = β_imp = exp(-h/τ)             — fully ZOH.
# B   : β_int = exp(-h/τ), β_imp = 1/(1+h/τ)  — ZOH integrator, BE Schur/Force.
# Note: stock CPU MuJoCo ignores dynprm[4] and always uses ZOH integrator,
# so on CPU all three methods reduce to A+'s integrator with no Schur/Force RHS
# correction. Method B is GPU-only (mjwarp).
_METHOD_TO_DYNPRM4: dict[str, float] = {"A": 0.0, "A+": 1.0, "B": 2.0}

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
    >1이면 ZOH: pd_substeps마다 PD 재계산,
    그 사이 서브스텝에서는 최신 ω로 전압만 갱신."""

    pd_substeps: int = 0
    """PD 재계산 주기 (physics step 단위). 0이면 policy step에서만 PD 계산.
    예: substeps=200, pd_substeps=50 → policy 20ms, PD 5ms, physics 0.1ms."""

    use_callback: bool = False
    """True: dyntype=user (Python callback, standard mujoco only).
    False: dyntype=filterexact (recommended, mujoco_warp compatible)."""

    use_coupled: bool = False
    """True: dyntype=filterexact_coupled (Schur complement back-EMF coupling).
    implicit solver에 cross-Jacobian 항을 추가하여 전류 추적 정확도 향상.
    use_callback=True와 동시 사용 불가."""

    method: Literal["A", "A+", "B"] = "A+"
    """coupled 모드에서 사용할 적분 method.

    - "A" : integrator/Schur/Force 모두 β_be = 1/(1+h/τ) (BE consistent). dynprm[4]=0.
    - "A+": integrator/Schur/Force 모두 β_exp = exp(-h/τ) (ZOH consistent). dynprm[4]=1.
    - "B" : integrator만 β_exp (ZOH), Schur/Force는 β_be (BE). dynprm[4]=2. **GPU 전용**.

    GPU 전용 의미: stock CPU MuJoCo 는 dynprm[4] 를 무시하고 항상 ZOH integrator 를 쓰고
    Schur/Force RHS 보정 자체가 patched mjwarp 에만 존재한다. 따라서 CPU 에서는 A/A+/B
    가 모두 ZOH integrator + (Schur/Force 보정 없음) 으로 동작한다.

    dynprm[3]은 demag 기준 Ke_nom·gr 저장용으로 별도 사용."""

    # ── Torque-tracking integral loop (driver-rate, 5 ms) ────────────
    use_torque_loop: bool = False
    """드라이버 안에 토크 추종 적분 제어 루프를 활성화.

    False (기본): I_cmd = τ_cmd / (Kt_nominal·gr) — 기존 동작 (회귀 안전).
    True: I_cmd += Ki·integral, where integral accumulates (τ_cmd − τ_actual_prev)·DRIVER_DT.

    감자(demag) 고장으로 Kt_real < Kt_nominal 일 때, 적분기가 I_cmd 를 자동 상승시켜
    τ_actual 이 τ_cmd 를 추종하도록 한다. 그 결과 고장 영향이 q 추종 오차 대신 전류
    신호(I_cmd, I_actual)로 전가되어 진단용 feature 가 된다."""

    Ki: float = 50.0
    """토크 추종 적분 게인 [(A·s) / (N·m·s)]. use_torque_loop=True 일 때만 유효."""

    integral_max: float | None = None
    """적분기 anti-windup 클램프 [A·s].

    None: (I_max − τ_cmd_max/(Kt_nom·gr)) / Ki 로 자동 계산.
    현재 cfg 에선 I_max = effort_limit/(Kt_nom·gr) = τ_cmd_max/(Kt_nom·gr) 이라
    headroom 이 0 으로 수렴 → fallback 값 0.2 를 사용 (TODO: I_max 와 saturation/effort
    분리). 검증 스크립트에서는 명시값을 권장."""

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

        # Torque-tracking integral loop state
        self._integral: torch.Tensor | None = None   # [num_envs, num_joints]
        self._driver_dt: float = 0.005               # 5 ms (= pd_substeps × physics_dt)
        self._integral_max: float = 0.0              # anti-windup clamp [A·s]

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
            elif cfg.use_coupled:
                # Strategy D: dyntype=filterexact with dynprm[1,2] for back-EMF coupling
                # engine detects dynprm[1]>0 && dynprm[2]>0 → activates Schur complement
                # No enum change needed → ABI compatible with pip-installed bindings
                act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
                act.dynprm[0] = tau_e                       # L/R
                act.dynprm[1] = cfg.Ke * cfg.gear_ratio     # Ke_plant·gr (demag script may overwrite)
                act.dynprm[2] = cfg.L                        # L (inductance)
                act.dynprm[3] = cfg.Ke * cfg.gear_ratio     # Ke_nom·gr — NEVER modified
                # dynprm[4]: method selector for patched mjwarp.
                # 0 = A (BE), 1 = A+ (ZOH), 2 = B (ZOH integrator + BE Schur/Force).
                # Stock CPU MuJoCo ignores this slot.
                if cfg.method not in _METHOD_TO_DYNPRM4:
                    raise ValueError(
                        f"NativeElectricActuatorCfg.method must be one of "
                        f"{list(_METHOD_TO_DYNPRM4)}, got {cfg.method!r}"
                    )
                act.dynprm[4] = _METHOD_TO_DYNPRM4[cfg.method]
                # NOTE: patched mjwarp FILTEREXACT kernel computes
                #   act_dot += (dynprm[3] - dynprm[1]) * omega / dynprm[2]
                # Healthy: dynprm[3]==dynprm[1] → correction=0 → vanilla filterexact.
                # Demag:   inject_demagnetization() sets dynprm[1]=Ke_plant·gr, leaves dynprm[3] nominal.
                # ctrl = 목표 전류 (PD + back-EMF), 동일한 변환 로직
                act.ctrllimited = True
                act.ctrlrange[:] = np.array([-I_max * 2, I_max * 2])
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

        # ── Torque-tracking integral loop setup ──────────────────────
        num_envs = data.nworld
        num_joints = len(self._target_names)
        self._integral = torch.zeros(
            num_envs, num_joints, dtype=torch.float, device=device
        )

        # driver_dt = PD 재계산 주기 (5 ms in default Coupled cfg)
        pd_period = cfg.pd_substeps if cfg.pd_substeps > 0 else max(cfg.substeps, 1)
        self._driver_dt = pd_period * dt

        # anti-windup clamp
        I_max = cfg.effort_limit / self._Ktgr
        if cfg.integral_max is not None:
            self._integral_max = float(cfg.integral_max)
        else:
            # User formula: (I_max − τ_cmd_max / Kt_nom·gr) / Ki
            # 현재 cfg 에선 I_max == effort_limit/Kt_nom·gr 이므로 0 → fallback 0.2.
            # TODO: I_max(하드웨어) 와 effort_limit(소프트웨어 클램프) 분리 후 재검토.
            headroom = (I_max - cfg.effort_limit / self._Ktgr) / max(cfg.Ki, 1e-9)
            self._integral_max = headroom if headroom > 0 else 0.2

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
            "I_des": [],
            "integral": [],
            "tau_actual_prev": [],
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

        # ── ①② τ_des → I_des ─────────────────────────────────────
        # PD 재계산 조건:
        #   - substeps <= 1: 매 호출마다
        #   - sub_idx == 0: policy step 경계 (항상)
        #   - pd_substeps > 0 and sub_idx % pd_substeps == 0: PD 주기 경계
        pd_period = cfg.pd_substeps if cfg.pd_substeps > 0 else cfg.substeps
        recompute_pd = (cfg.substeps <= 1
                        or self._sub_idx == 0
                        or (cfg.pd_substeps > 0 and self._sub_idx % pd_period == 0))

        if recompute_pd:
            # PD + DC motor saturation → τ_des
            tau_des = super().compute(cmd)

            if cfg.use_torque_loop:
                # τ_actual_prev: 직전 5 ms 윈도우의 마지막 physics step 인가 토크.
                # data.actuator_force = gain·act = Kt_real·gr × I (post-clamp).
                # 0.1 ms (1 physics step) lag 존재하나 driver_dt(5ms)의 2% → 무시 가능.
                # reset() 에서 명시적으로 0 처리하므로 첫 호출 시 정상.
                tau_actual_prev = self._mjw_data.actuator_force[
                    :, self._global_ctrl_ids
                ]  # [num_envs, num_joints]

                error = tau_des - tau_actual_prev
                self._integral = torch.clamp(
                    self._integral + error * self._driver_dt,
                    min=-self._integral_max,
                    max=self._integral_max,
                )

                I_max_t = cfg.effort_limit / self._Ktgr
                I_des = tau_des / self._Ktgr + cfg.Ki * self._integral
                I_des = torch.clamp(I_des, min=-I_max_t, max=I_max_t)
            else:
                I_des = tau_des / self._Ktgr

            # 캐시 저장
            if cfg.substeps > 1:
                self._I_des_hold = I_des
                self._tau_des_hold = tau_des
        else:
            # PD 주기 사이: 캐시된 I_des 사용 (ZOH)
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
            self._log["I_des"].append(I_des[0].cpu().numpy().copy())
            if cfg.use_torque_loop and self._integral is not None:
                tau_act_prev = (
                    self._mjw_data.actuator_force[0, self._global_ctrl_ids]
                    .cpu().numpy().copy()
                )
                self._log["integral"].append(
                    self._integral[0].cpu().numpy().copy()
                )
                self._log["tau_actual_prev"].append(tau_act_prev)
            self._log_step += 1

        return ctrl

    # ── reset ────────────────────────────────────────────────────────

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        """에피소드 리셋 시 전류(act), 적분기, 서브스텝 카운터 초기화.

        mjlab은 policy step 경계에서 reset을 호출하므로
        _sub_idx = 0 리셋이 decimation 루프 시작과 일치.

        actuator_force 도 함께 0 처리: 다음 mj_step 의 fwd_actuation 에서 어차피
        재계산되지만, 그 전에 적분 루프가 읽으면 stale 값이 들어가므로 명시적 클리어.
        """
        if self._mjw_data is not None and self._act_adr is not None:
            if env_ids is None:
                self._mjw_data.act[:, self._act_adr] = 0.0
                self._mjw_data.actuator_force[:, self._global_ctrl_ids] = 0.0
            else:
                # env_ids [N] × _act_adr [J] → broadcast [N, J]
                ids = env_ids.unsqueeze(-1)
                self._mjw_data.act[ids, self._act_adr] = 0.0
                self._mjw_data.actuator_force[ids, self._global_ctrl_ids] = 0.0
        # 적분기 상태 리셋
        if self._integral is not None:
            if env_ids is None:
                self._integral.zero_()
            else:
                self._integral[env_ids] = 0.0
        # 서브스텝 카운터 리셋 → 다음 compute()에서 PD 재계산
        self._sub_idx = 0

    # ── update (no-op) ───────────────────────────────────────────────

    def update(self, dt: float) -> None:
        """MuJoCo가 act 적분을 처리하므로 별도 업데이트 불필요."""
        pass
