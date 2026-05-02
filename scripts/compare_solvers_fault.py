"""Staggered vs Coupled ODE solver 비교 (v2): 공정한 커플링 비교 + 동적 시나리오.

v1 대비 개선:
  1. 기계측 적분 통일: 두 solver 모두 backward Euler로 기계측을 풀고,
     전기-기계 커플링만 simultaneous vs staggered로 다르게 한다.
  2. 동적 입력 시나리오: step input 외에 사인파, 랜덤 지령, 스윙↔스탠스 전환 추가.

사용법:
  python scripts/compare_solvers_fault.py
  # → solver_comparison/ 디렉토리에 PNG + metrics.txt 생성

의존성: numpy, scipy, matplotlib (torch/MuJoCo 불필요)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

matplotlib.use("Agg")

# ═══════════════════════════════════════════════════════════════════════════════
#  Motor Parameters
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class MotorParams:
    """DC 모터 파라미터 (Go2 기본값 + 고장 확장)."""

    Kt: float = 0.128          # 토크 상수 [N·m/A]
    Ke: float = 0.128          # 역기전력 상수 [V·s/rad_motor]
    R: float = 0.3             # 권선 저항 [Ω]
    L: float = 1e-4            # 권선 인덕턴스 [H]
    gear_ratio: float = 6.33   # 감속비
    J: float = 0.01            # joint-space 유효 관성 [kg·m²]
    V_bus: float = float("inf")  # 버스 전압 제한 [V]
    friction: float = 0.0      # 추가 Coulomb 마찰 [N·m]

    @property
    def tau_e(self) -> float:
        return self.L / self.R

    @property
    def gr(self) -> float:
        return self.gear_ratio


FAULT_SCENARIOS: dict[str, MotorParams] = {
    "healthy":        MotorParams(),
    "R_2x":           MotorParams(R=0.6),
    "R_5x":           MotorParams(R=1.5),
    "demag_50pct":    MotorParams(Ke=0.064, Kt=0.064),
    "demag_75pct":    MotorParams(Ke=0.032, Kt=0.032),
    "friction_2Nm":   MotorParams(friction=2.0),
    "friction_5Nm":   MotorParams(friction=5.0),
    "L_10x":          MotorParams(L=1e-3),
    "L_100x":         MotorParams(L=1e-2),
    "V_bus_12V":      MotorParams(V_bus=12.0),
    "combined":       MotorParams(R=0.6, Ke=0.096, Kt=0.096, friction=1.0, V_bus=12.0),
}

# ═══════════════════════════════════════════════════════════════════════════════
#  Simulation Constants
# ═══════════════════════════════════════════════════════════════════════════════

DT_VALUES = [1e-4, 5e-4, 1e-3, 5e-3]  # 0.1ms ~ 5ms
FRICTION_EPS = 0.01    # tanh 마찰 근사 smoothing 계수


# ═══════════════════════════════════════════════════════════════════════════════
#  Input Scenarios — 동적 시나리오 정의
# ═══════════════════════════════════════════════════════════════════════════════


@dataclasses.dataclass
class Scenario:
    """시뮬레이션 입력 시나리오."""
    name: str
    t_end: float                          # 종료 시간 [s]
    I_des_fn: object                      # (t) -> I_des [A]
    tau_load_fn: object                   # (t, omega, params) -> tau_load [N·m]
    description: str = ""


def _make_step_scenario() -> Scenario:
    """Step input: I_des=29A 스텝, 상수 중력 부하."""
    return Scenario(
        name="step",
        t_end=0.01,
        I_des_fn=lambda t: 29.0,
        tau_load_fn=lambda t, omega, p: 5.0 + p.friction * np.sign(omega),
        description="Step I_des=29A, τ_gravity=5N·m",
    )


def _make_sinusoidal_scenario() -> Scenario:
    """사인파 토크 지령: 200Hz (policy rate), 진폭 ±20A."""
    freq = 200.0  # Hz
    return Scenario(
        name="sinusoidal",
        t_end=0.02,  # 4 cycles
        I_des_fn=lambda t: 20.0 * np.sin(2 * np.pi * freq * t),
        tau_load_fn=lambda t, omega, p: 3.0 + p.friction * np.sign(omega),
        description="Sine 200Hz ±20A, τ_gravity=3N·m",
    )


def _make_random_scenario() -> Scenario:
    """랜덤 지령: 5ms마다 I_des 변경 (ZOH), RL 정책 모사."""
    rng = np.random.RandomState(42)
    switch_dt = 0.005  # 5ms policy period
    n_switches = 10
    t_end = switch_dt * n_switches  # 50ms
    values = rng.uniform(-25.0, 25.0, n_switches)

    def I_des_fn(t):
        idx = min(int(t / switch_dt), n_switches - 1)
        return values[idx]

    return Scenario(
        name="random_zoh",
        t_end=t_end,
        I_des_fn=I_des_fn,
        tau_load_fn=lambda t, omega, p: 5.0 + p.friction * np.sign(omega),
        description="Random ZOH ±25A every 5ms, τ_gravity=5N·m",
    )


def _make_swing_stance_scenario() -> Scenario:
    """스윙↔스탠스 전환: 급격한 부하 변화 + ω 부호 반전.

    스탠스(0~5ms): 높은 부하(20N·m), 낮은 ω → 높은 I
    스윙(5~10ms): 낮은 부하(1N·m), 높은 ω → back-EMF 영향 큼
    재접촉(10~15ms): 다시 높은 부하, ω 급감
    """
    def I_des_fn(t):
        phase = t % 0.015
        if phase < 0.005:   # 스탠스: 큰 토크
            return 25.0
        elif phase < 0.010:  # 스윙: 빠른 복귀
            return -15.0
        else:               # 재접촉: 충격 흡수
            return 30.0

    def tau_load_fn(t, omega, p):
        phase = t % 0.015
        if phase < 0.005:   # 스탠스: 무거운 부하
            base = 20.0
        elif phase < 0.010:  # 스윙: 가벼운 부하
            base = 1.0
        else:               # 재접촉: 충격 부하
            base = 25.0
        return base + p.friction * np.sign(omega)

    return Scenario(
        name="swing_stance",
        t_end=0.03,  # 2 full cycles
        I_des_fn=I_des_fn,
        tau_load_fn=tau_load_fn,
        description="Swing↔Stance: τ_load={20,1,25}N·m, I_des={25,-15,30}A",
    )


SCENARIOS = [
    _make_step_scenario(),
    _make_sinusoidal_scenario(),
    _make_random_scenario(),
    _make_swing_stance_scenario(),
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Load Torque (smooth version for RK45)
# ═══════════════════════════════════════════════════════════════════════════════


def smooth_sign(omega: float) -> float:
    """RK45용 부드러운 sign 근사."""
    return np.tanh(omega / FRICTION_EPS)


# ═══════════════════════════════════════════════════════════════════════════════
#  Solver 1: Coupled Backward Euler 2×2 (simultaneous)
# ═══════════════════════════════════════════════════════════════════════════════


def coupled_step(
    I: float, omega: float, V_cmd: float, tau_L: float, p: MotorParams, dt: float
) -> tuple[float, float]:
    """2×2 연립 Backward Euler: I와 ω를 동시에 결정.

    S · [ΔI, Δω]ᵀ = dt · f(xₙ)
    전기-기계 cross-coupling이 시스템 행렬 S에 포함됨.
    """
    L, R, Ke, Kt, gr, J = p.L, p.R, p.Ke, p.Kt, p.gr, p.J

    s00 = L + dt * R
    s01 = dt * Ke * gr
    s10 = -dt * Kt * gr
    s11 = J
    det = s00 * s11 - s01 * s10

    b_I = dt * (V_cmd - R * I - Ke * gr * omega)
    b_w = dt * (Kt * gr * I - tau_L)

    inv_det = 1.0 / det
    dI = inv_det * (s11 * b_I - s01 * b_w)
    dw = inv_det * (-s10 * b_I + s00 * b_w)

    return I + dI, omega + dw


# ═══════════════════════════════════════════════════════════════════════════════
#  Solver 2: Staggered Backward Euler (같은 차수, 커플링만 다름)
# ═══════════════════════════════════════════════════════════════════════════════


def staggered_step(
    I: float, omega: float, V_cmd: float, tau_L: float, p: MotorParams, dt: float
) -> tuple[float, float]:
    """Staggered Backward Euler: 기계측도 backward Euler, 커플링만 분리.

    Step 1 — 전기 (ω 고정, backward Euler):
        (L + dt·R) · I_{n+1} = L·Iₙ + dt·(V_cmd - Ke·gr·ωₙ)
        → I_{n+1} = (L·Iₙ + dt·(V_cmd - Ke·gr·ωₙ)) / (L + dt·R)

    Step 2 — 기계 (새 I 사용, backward Euler):
        J · (ω_{n+1} - ωₙ) = dt · (Kt·gr·I_{n+1} - τ_load)
        → ω_{n+1} = ωₙ + dt · (Kt·gr·I_{n+1} - τ_load) / J

    커플링 차이:
      - coupled: ΔI 계산 시 Δω가 시스템 행렬에 반영됨
      - staggered: ΔI 계산 시 ω = ωₙ 고정 → 1-step lag
      - 기계측은 동일하게 backward Euler (forward Euler 아님)

    Note: 기계측에 backward Euler를 쓰지만, tau_L이 omega에 의존하는 경우
    (Coulomb friction) 여기서는 tau_L을 ωₙ 기준으로 평가한다 (호출 시 이미 계산됨).
    이는 coupled solver와 동일한 처리이므로 공정한 비교가 된다.
    """
    L, R, Ke, Kt, gr, J = p.L, p.R, p.Ke, p.Kt, p.gr, p.J

    # 전기 서브스텝: ω 고정, implicit in I
    I_new = (L * I + dt * (V_cmd - Ke * gr * omega)) / (L + dt * R)

    # 기계 서브스텝: 새 I 사용, implicit in ω (tau_L은 ωₙ 기준으로 이미 계산)
    # J·(ω_{n+1} - ωₙ)/dt = Kt·gr·I_{n+1} - tau_L
    omega_new = omega + dt * (Kt * gr * I_new - tau_L) / J

    return I_new, omega_new


# ═══════════════════════════════════════════════════════════════════════════════
#  Solver 3: RK45 Ground Truth
# ═══════════════════════════════════════════════════════════════════════════════


def rk45_reference(
    p: MotorParams, scenario: Scenario
) -> dict[str, np.ndarray]:
    """scipy RK45로 연립 ODE를 고정밀도로 적분."""
    def rhs(t, y):
        I, omega = y
        I_des = scenario.I_des_fn(t)
        V_cmd_base = p.R * I_des
        V_cmd = np.clip(V_cmd_base, -p.V_bus, p.V_bus)
        # RK45용: 마찰은 smooth 근사
        tau_L_base = scenario.tau_load_fn(t, omega, p)
        # friction을 smooth로 교체 (tau_load_fn이 sign을 쓰므로 여기서 보정)
        tau_L = tau_L_base - p.friction * np.sign(omega) + p.friction * smooth_sign(omega)
        dI_dt = (V_cmd - p.R * I - p.Ke * p.gr * omega) / p.L
        domega_dt = (p.Kt * p.gr * I - tau_L) / p.J
        return [dI_dt, domega_dt]

    n_eval = int(round(scenario.t_end / 1e-5)) + 1
    t_eval = np.linspace(0, scenario.t_end, n_eval)
    sol = solve_ivp(rhs, [0, scenario.t_end], [0.0, 0.0], method="RK45",
                    t_eval=t_eval, rtol=1e-10, atol=1e-12, max_step=1e-5)

    return {
        "t": sol.t,
        "I": sol.y[0],
        "omega": sol.y[1],
        "tau": p.Kt * p.gr * sol.y[0],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Simulation Runner
# ═══════════════════════════════════════════════════════════════════════════════


def run_simulation(
    solver_fn, p: MotorParams, dt: float, scenario: Scenario
) -> dict[str, np.ndarray]:
    """고정 dt solver 실행."""
    n_steps = int(round(scenario.t_end / dt))
    t = np.zeros(n_steps + 1)
    I = np.zeros(n_steps + 1)
    omega = np.zeros(n_steps + 1)

    for k in range(n_steps):
        I_des = scenario.I_des_fn(t[k])
        V_cmd_base = p.R * I_des
        V_cmd = np.clip(V_cmd_base, -p.V_bus, p.V_bus)
        tau_L = scenario.tau_load_fn(t[k], omega[k], p)
        I[k + 1], omega[k + 1] = solver_fn(I[k], omega[k], V_cmd, tau_L, p, dt)
        t[k + 1] = t[k] + dt

    return {
        "t": t,
        "I": I,
        "omega": omega,
        "tau": p.Kt * p.gr * I,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def interpolate_to_ref(ref: dict, res: dict, key: str) -> np.ndarray:
    """res의 시계열을 ref의 시간축에 선형 보간."""
    return np.interp(ref["t"], res["t"], res[key])


def compute_metrics(
    ref: dict, coupled: dict, staggered: dict
) -> dict[str, float]:
    """RK45 대비 오차 메트릭 계산."""
    metrics = {}
    for var in ("I", "omega"):
        c = interpolate_to_ref(ref, coupled, var)
        s = interpolate_to_ref(ref, staggered, var)
        r = ref[var]

        err_c = np.abs(c - r)
        err_s = np.abs(s - r)

        metrics[f"max_{var}_coupled"] = float(np.max(err_c))
        metrics[f"max_{var}_staggered"] = float(np.max(err_s))
        metrics[f"rms_{var}_coupled"] = float(np.sqrt(np.mean(err_c**2)))
        metrics[f"rms_{var}_staggered"] = float(np.sqrt(np.mean(err_s**2)))

        mc = metrics[f"max_{var}_coupled"]
        ms = metrics[f"max_{var}_staggered"]
        metrics[f"ratio_{var}"] = ms / mc if mc > 1e-15 else float("inf") if ms > 1e-15 else 1.0

        # 정상상태 오차 (마지막 10% 구간 평균)
        n_tail = max(1, len(r) // 10)
        metrics[f"tail_{var}_coupled"] = float(np.mean(np.abs(c[-n_tail:] - r[-n_tail:])))
        metrics[f"tail_{var}_staggered"] = float(np.mean(np.abs(s[-n_tail:] - r[-n_tail:])))

    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

C_RK45 = "black"
C_COUPLED = "tab:blue"
C_STAGGERED = "tab:red"


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved: {path}")


def plot_per_fault(
    fault_name: str, scenario: Scenario, p: MotorParams,
    ref: dict, coupled: dict, staggered: dict,
    dt: float, out_dir: Path
) -> None:
    """단일 (고장, 시나리오)의 I/ω/τ 3행 비교 + 오차 패널."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex="col",
                             gridspec_kw={"width_ratios": [3, 1]})
    labels = [("I", "A", "Current"), ("omega", "rad/s", "Angular velocity"), ("tau", "N·m", "Torque")]

    for row, (key, unit, title) in enumerate(labels):
        # 좌: 시계열
        ax = axes[row, 0]
        ax.plot(ref["t"] * 1e3, ref[key], color=C_RK45, ls="--", lw=1.5, label="RK45", zorder=3)
        ax.plot(coupled["t"] * 1e3, coupled[key], color=C_COUPLED, lw=1.2, label="Coupled", zorder=2)
        ax.plot(staggered["t"] * 1e3, staggered[key], color=C_STAGGERED, lw=1.2, alpha=0.8, label="Staggered", zorder=1)
        ax.set_ylabel(f"{title} [{unit}]")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

        # 우: 오차 (vs RK45)
        ax_err = axes[row, 1]
        c_interp = interpolate_to_ref(ref, coupled, key)
        s_interp = interpolate_to_ref(ref, staggered, key)
        ax_err.plot(ref["t"] * 1e3, c_interp - ref[key], color=C_COUPLED, lw=1.0, label="Coupled err")
        ax_err.plot(ref["t"] * 1e3, s_interp - ref[key], color=C_STAGGERED, lw=1.0, alpha=0.8, label="Staggered err")
        ax_err.axhline(0, color="gray", lw=0.5)
        ax_err.set_ylabel(f"Δ{key}")
        ax_err.legend(fontsize=6, loc="best")
        ax_err.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Time [ms]")
    axes[-1, 1].set_xlabel("Time [ms]")
    tau_e_ms = p.tau_e * 1e3
    fig.suptitle(
        f"{fault_name} / {scenario.name}  |  dt={dt*1e3:.1f}ms  |  "
        f"τ_e={tau_e_ms:.2f}ms  |  dt/τ_e={dt/p.tau_e:.1f}\n"
        f"{scenario.description}",
        fontsize=9,
    )
    _save(fig, out_dir / f"{fault_name}_{scenario.name}_dt{dt*1e3:.1f}ms.png")


def plot_dt_sweep(
    fault_name: str, scenario: Scenario, p: MotorParams,
    ref: dict, results: dict[float, dict], out_dir: Path
) -> None:
    """한 (고장, 시나리오)의 dt별 수렴 (3행×N열)."""
    dts = sorted(results.keys())
    fig, axes = plt.subplots(3, len(dts), figsize=(4 * len(dts), 8), sharex="col")
    labels = [("I", "A"), ("omega", "rad/s"), ("tau", "N·m")]

    for col, dt in enumerate(dts):
        coupled = results[dt]["coupled"]
        staggered = results[dt]["staggered"]
        for row, (key, unit) in enumerate(labels):
            ax = axes[row, col]
            ax.plot(ref["t"] * 1e3, ref[key], color=C_RK45, ls="--", lw=1.2, label="RK45")
            ax.plot(coupled["t"] * 1e3, coupled[key], color=C_COUPLED, lw=1.0, label="Coupled")
            ax.plot(staggered["t"] * 1e3, staggered[key], color=C_STAGGERED, lw=1.0, alpha=0.8, label="Staggered")
            ax.grid(True, alpha=0.3)
            if col == 0:
                ax.set_ylabel(f"{key} [{unit}]")
            if row == 0:
                ax.set_title(f"dt={dt*1e3:.1f}ms", fontsize=9)
            if row == len(labels) - 1:
                ax.set_xlabel("Time [ms]")
            if row == 0 and col == len(dts) - 1:
                ax.legend(fontsize=6, loc="best")

    fig.suptitle(f"{fault_name} / {scenario.name}  |  τ_e={p.tau_e*1e3:.2f}ms\n{scenario.description}", fontsize=9)
    _save(fig, out_dir / f"{fault_name}_{scenario.name}_dt_sweep.png")


def plot_error_heatmap(
    all_metrics: dict[str, dict[str, dict[float, dict]]], out_dir: Path
) -> None:
    """(시나리오×고장)(행) × dt(열) error ratio heatmap."""
    # 행: 시나리오/고장 조합
    row_labels = []
    row_data_I = []
    row_data_omega = []

    dts = sorted(DT_VALUES)

    for sc_name, fault_dict in all_metrics.items():
        for fn, dt_dict in fault_dict.items():
            row_labels.append(f"{sc_name}/{fn}")
            row_I = []
            row_w = []
            for dt in dts:
                m = dt_dict[dt]
                row_I.append(min(m.get("ratio_I", 1.0), 100.0))
                row_w.append(min(m.get("ratio_omega", 1.0), 100.0))
            row_data_I.append(row_I)
            row_data_omega.append(row_w)

    for var, data in [("I", row_data_I), ("omega", row_data_omega)]:
        arr = np.array(data)
        fig, ax = plt.subplots(figsize=(8, max(6, len(row_labels) * 0.35)))
        im = ax.imshow(arr, aspect="auto", cmap="RdYlGn_r", vmin=0.8, vmax=5.0)
        ax.set_xticks(range(len(dts)))
        ax.set_xticklabels([f"{d*1e3:.1f}ms" for d in dts])
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=6)
        ax.set_xlabel("dt")
        ax.set_title(f"Error Ratio (staggered/coupled) — {var}")
        fig.colorbar(im, ax=ax, label="ratio (1.0=identical, >1=staggered worse)")

        for i in range(len(row_labels)):
            for j in range(len(dts)):
                val = arr[i, j]
                color = "white" if val > 3 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5, color=color)

        _save(fig, out_dir / f"heatmap_{var}.png")


def plot_scenario_summary(
    all_metrics: dict[str, dict[str, dict[float, dict]]], out_dir: Path
) -> None:
    """시나리오별 요약: dt vs max ratio across all faults."""
    dts = sorted(DT_VALUES)

    for var in ("I", "omega"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for sc_name, fault_dict in all_metrics.items():
            # 각 dt에서 모든 fault 중 최대 ratio
            max_ratios = []
            for dt in dts:
                ratios = [fault_dict[fn][dt].get(f"ratio_{var}", 1.0) for fn in fault_dict]
                max_ratios.append(max(ratios))
            ax.semilogy([d * 1e3 for d in dts], max_ratios, "o-", label=sc_name, lw=2, markersize=6)

        ax.axhline(1.0, color="gray", ls="--", lw=0.8, label="ratio=1 (identical)")
        ax.axhline(2.0, color="orange", ls=":", lw=0.8, label="ratio=2 threshold")
        ax.set_xlabel("dt [ms]")
        ax.set_ylabel(f"Max error ratio across faults ({var})")
        ax.set_title(f"Scenario Summary — {var}: worst-case staggered/coupled ratio")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which="both")
        _save(fig, out_dir / f"scenario_summary_{var}.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════


def main():
    out_dir = Path("solver_comparison")
    out_dir.mkdir(exist_ok=True)

    # all_metrics[scenario_name][fault_name][dt] = metrics_dict
    all_metrics: dict[str, dict[str, dict[float, dict]]] = {}
    lines: list[str] = []

    lines.append("=" * 110)
    lines.append("Staggered vs Coupled ODE Solver — Fair Comparison (v2)")
    lines.append("  Both solvers use Backward Euler for mechanical side.")
    lines.append("  Only difference: coupled solves I,ω simultaneously; staggered solves I first (ω frozen), then ω.")
    lines.append("=" * 110)

    for scenario in SCENARIOS:
        sc_name = scenario.name
        all_metrics[sc_name] = {}
        lines.append(f"\n{'─'*110}")
        lines.append(f"SCENARIO: {sc_name}  |  {scenario.description}  |  t_end={scenario.t_end*1e3:.0f}ms")
        lines.append(f"{'─'*110}")
        lines.append(f"{'Fault':<18} {'dt(ms)':>7} {'max_I_C':>10} {'max_I_S':>10} {'ratio_I':>8} "
                     f"{'max_ω_C':>10} {'max_ω_S':>10} {'ratio_ω':>8}")
        lines.append("-" * 90)

        print(f"\n{'='*60}")
        print(f"SCENARIO: {sc_name}  |  {scenario.description}")
        print(f"{'='*60}")

        for fault_name, p in FAULT_SCENARIOS.items():
            print(f"\n  [{fault_name}]  τ_e={p.tau_e*1e3:.3f}ms")
            ref = rk45_reference(p, scenario)
            all_metrics[sc_name][fault_name] = {}
            dt_results: dict[float, dict] = {}

            for dt in DT_VALUES:
                # dt가 시나리오 t_end보다 크면 스킵
                if dt > scenario.t_end / 2:
                    continue

                coupled = run_simulation(coupled_step, p, dt, scenario)
                staggered = run_simulation(staggered_step, p, dt, scenario)
                m = compute_metrics(ref, coupled, staggered)
                all_metrics[sc_name][fault_name][dt] = m
                dt_results[dt] = {"coupled": coupled, "staggered": staggered}

                lines.append(
                    f"{fault_name:<18} {dt*1e3:>7.1f} "
                    f"{m['max_I_coupled']:>10.4f} {m['max_I_staggered']:>10.4f} {m['ratio_I']:>8.2f} "
                    f"{m['max_omega_coupled']:>10.6f} {m['max_omega_staggered']:>10.6f} {m['ratio_omega']:>8.2f}"
                )
                print(f"    dt={dt*1e3:.1f}ms  ratio_I={m['ratio_I']:.2f}  ratio_ω={m['ratio_omega']:.2f}")

            # 플롯: dt=0.5ms 비교 (있으면)
            if 5e-4 in dt_results:
                plot_per_fault(fault_name, scenario, p, ref,
                               dt_results[5e-4]["coupled"], dt_results[5e-4]["staggered"],
                               5e-4, out_dir)

            # 플롯: dt sweep
            if dt_results:
                plot_dt_sweep(fault_name, scenario, p, ref, dt_results, out_dir)

    # heatmap
    plot_error_heatmap(all_metrics, out_dir)

    # 시나리오 요약
    plot_scenario_summary(all_metrics, out_dir)

    # 메트릭 저장
    lines.append("\n" + "=" * 110)
    metrics_text = "\n".join(lines)
    print(f"\n{metrics_text}")

    metrics_path = out_dir / "metrics.txt"
    metrics_path.write_text(metrics_text)
    print(f"\n[INFO] Metrics saved to {metrics_path}")
    print(f"[INFO] Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
