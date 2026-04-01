"""Backward (Implicit) Euler solver for the coupled electrical-mechanical ODE.

수학적 유도
-----------
상태 벡터: x = [I, ω]ᵀ  (전류 [A], joint 각속도 [rad/s])

연립 ODE:
    L · dI/dt  =  V_cmd  -  R·I  -  Ke·gr·ω        … (전기)
    J · dω/dt  =  Kt·gr·I  -  τ_load               … (기계)

질량 행렬·힘 벡터로 표기:
    M · ẋ = f(x)

    M   = [[L,  0 ],        f(x) = [V_cmd - R·I - Ke·gr·ω]
           [0,  J ]]                [Kt·gr·I - τ_load      ]

Jacobian  A_f = ∂f/∂x:
    A_f = [[-R,      -Ke·gr],
           [Kt·gr,    0    ]]

Backward (Implicit) Euler 전개:
    M · (x_{n+1} - xₙ) / dt  =  f(xₙ) + A_f · (x_{n+1} - xₙ)
    ⟹  (M - dt·A_f) · Δx  =  dt · f(xₙ)

시스템 행렬  S = M - dt·A_f:
    S = [[L + dt·R,    dt·Ke·gr  ],
         [-dt·Kt·gr,   J         ]]

2×2 역행렬 (Cramer's rule):
    det(S) = (L + dt·R)·J  +  dt²·Ke·Kt·gr²

    S⁻¹ = (1/det) · [[ J,          -dt·Ke·gr ],
                      [ dt·Kt·gr,   L + dt·R  ]]

b = dt · f(xₙ) 로 놓으면:
    ΔI = ( J · b_I  -  dt·Ke·gr · b_ω ) / det
    Δω = ( dt·Kt·gr · b_I  +  (L+dt·R) · b_ω ) / det

    x_{n+1} = xₙ + Δx

특성
----
- 선형 시스템이므로 implicit Euler = 정확한 backward Euler (linearization 오차 없음)
- det > 0 보장: (L+dt·R)·J > 0, dt²·Ke·Kt·gr² > 0
- explicit Euler 대비 안정 조건 무조건 만족 (unconditionally stable for linear ODE)
- I 업데이트 시 Δω가, ω 업데이트 시 ΔI가 S 안에서 동시에 결정됨 → 진정한 커플링
"""

from __future__ import annotations

import torch


class CoupledElecMechSolver:
    """전기-기계 연립 ODE Backward Euler 솔버.

    파라미터
    --------
    L  : 권선 인덕턴스 [H]
    R  : 권선 저항 [Ω]
    Ke : 역기전력 상수 [V·s/rad_motor]
    Kt : 토크 상수 [N·m/A]
    J  : joint-space 유효 관성 모멘트 [kg·m²]
         (모터 회전자 관성을 기어비로 환산한 값 + 링크 관성)
    gr : 기어비 (감속비)
    dt : 적분 시간 [s]

    내부 상수 (초기화 시 1회 계산)
    --------------------------------
    시스템 행렬 S:
        s00 = L + dt·R        s01 = dt·Ke·gr
        s10 = -dt·Kt·gr       s11 = J
    det = s00·s11 - s01·s10
        = (L + dt·R)·J + dt²·Ke·Kt·gr²
    """

    def __init__(
        self,
        L:  float,
        R:  float,
        Ke: float,
        Kt: float,
        J:  float,
        gr: float,
        dt: float,
    ) -> None:
        self.L  = L
        self.R  = R
        self.Ke = Ke
        self.Kt = Kt
        self.J  = J
        self.gr = gr
        self.dt = dt

        # 시스템 행렬 S의 원소 (스칼라, 상수)
        self._s00 = L + dt * R        # S[0,0]
        self._s01 = dt * Ke * gr      # S[0,1]
        self._s10 = -dt * Kt * gr     # S[1,0]
        self._s11 = J                 # S[1,1]

        # det(S) = (L+dt·R)·J + dt²·Ke·Kt·gr²
        self._det = self._s00 * self._s11 - self._s01 * self._s10

        assert self._det > 0, (
            f"det(S)={self._det:.3e} ≤ 0: 파라미터를 확인하세요. "
            f"(L={L}, R={R}, Ke={Ke}, Kt={Kt}, J={J}, gr={gr}, dt={dt})"
        )

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def step(
        self,
        I:        torch.Tensor,
        omega:    torch.Tensor,
        V_cmd:    torch.Tensor,
        tau_load: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backward Euler 1-step: [I, ω]ₙ → [I, ω]ₙ₊₁.

        S · Δx = b  를 2×2 역행렬로 직접 풀어 I와 ω를 동시에 업데이트한다.

        Args:
            I        : 전류      [..., num_joints]
            omega    : joint 각속도 [..., num_joints]  (MuJoCo qvel 기준)
            V_cmd    : 인가 전압  [..., num_joints]
            tau_load : 부하 토크  [..., num_joints]  (joint space)

        Returns:
            I_new, omega_new  (같은 shape)
        """
        # b = dt · f(xₙ)
        b_I = self.dt * (V_cmd  - self.R * I  - self.Ke * self.gr * omega)
        b_w = self.dt * (self.Kt * self.gr * I  - tau_load)

        # Δx = S⁻¹ · b
        #   S⁻¹ = (1/det) · [[ s11, -s01],
        #                     [-s10,  s00]]
        inv_det = 1.0 / self._det
        dI  = inv_det * ( self._s11 * b_I  - self._s01 * b_w)
        dw  = inv_det * (-self._s10 * b_I  + self._s00 * b_w)

        return I + dI, omega + dw

    # ── 진단 ─────────────────────────────────────────────────────────────────

    def system_matrix(self) -> list[list[float]]:
        """시스템 행렬 S = M - dt·A_f 반환 (디버그용)."""
        return [
            [self._s00, self._s01],
            [self._s10, self._s11],
        ]

    def __repr__(self) -> str:
        S = self.system_matrix()
        return (
            f"CoupledElecMechSolver(\n"
            f"  S = [[{S[0][0]:.4f}, {S[0][1]:.4f}],\n"
            f"       [{S[1][0]:.4f}, {S[1][1]:.4f}]]\n"
            f"  det(S) = {self._det:.4e}\n"
            f"  τ_e = L/R = {self.L/self.R*1e3:.3f} ms\n"
            f"  τ_m = J/(Kt·Ke·gr²/R) = "
            f"{self.J * self.R / (self.Kt * self.Ke * self.gr**2) * 1e3:.3f} ms\n"
            f")"
        )
