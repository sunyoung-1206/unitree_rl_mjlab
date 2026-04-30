"""
Phase 3: filterexact_coupled 검증
비교: (A) filterexact, (B) filterexact_coupled, (C) RK45 ground truth
입력: 전압(V) 통일, ctrl = (V - Ke*gr*omega) / R 로 변환
"""

import numpy as np
import mujoco
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import copy

# ─── 모터 파라미터 ───
R = 0.3          # Ω
L = 0.0001       # H (0.1 mH)
Kt = 0.128       # N·m/A
Ke = 0.128       # V·s/rad
gr = 6.33        # gear ratio
tau_e = L / R    # 0.000333 s
Kt_gr = Kt * gr  # 0.81024
Ke_gr = Ke * gr  # 0.81024
I_MAX = 29.0     # A
V_BUS = 24.0     # V

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "phase3_results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

XML_FILTEREXACT = (BASE_DIR / "test_filterexact.xml").read_text()
XML_COUPLED = (BASE_DIR / "test_coupled.xml").read_text()

DT_LIST = [0.0001, 0.0005, 0.001, 0.005]  # 0.1ms, 0.5ms, 1ms, 5ms
DT_LABELS = ["0.1ms", "0.5ms", "1.0ms", "5.0ms"]
T_END = 0.02  # 20ms


# ─── 시나리오 정의 ───
def make_scenarios():
    scenarios = {}

    # 3.1 Step voltage
    scenarios["step"] = {
        "V_func": lambda t: 8.0,
        "load_func": lambda t: 0.0,
        "R_func": lambda t: R,
        "title": "Step Voltage (8V)",
    }

    # 3.2 Sinusoidal voltage
    scenarios["sinusoidal"] = {
        "V_func": lambda t: 8.0 * np.sin(2 * np.pi * 100 * t),
        "load_func": lambda t: 0.0,
        "R_func": lambda t: R,
        "title": "Sinusoidal Voltage (8V, 100Hz)",
    }

    # 3.3 Step with load change
    scenarios["load_change"] = {
        "V_func": lambda t: 8.0,
        "load_func": lambda t: -3.0 if t >= 0.01 else 0.0,
        "R_func": lambda t: R,
        "title": "Step Voltage + Load Change at 10ms",
    }

    # 3.4 Voltage reversal (핵심 테스트)
    scenarios["reversal"] = {
        "V_func": lambda t: 8.0 if t < 0.01 else -8.0,
        "load_func": lambda t: 0.0,
        "R_func": lambda t: R,
        "title": "Voltage Reversal at 10ms (KEY TEST)",
    }

    # 3.5 Fault: R jump
    scenarios["fault_R"] = {
        "V_func": lambda t: 8.0,
        "load_func": lambda t: 0.0,
        "R_func": lambda t: 0.6 if t >= 0.01 else R,
        "title": "Fault: R 0.3→0.6Ω at 10ms",
    }

    return scenarios


# ─── RK45 Ground Truth ───
def solve_rk45(scenario, J_inertia):
    V_func = scenario["V_func"]
    load_func = scenario["load_func"]
    R_func = scenario["R_func"]

    def motor_ode(t, y):
        I, omega = y
        V = np.clip(V_func(t), -V_BUS, V_BUS)
        R_t = R_func(t)
        dI_dt = (V - R_t * I - Ke_gr * omega) / L
        tau_motor = Kt_gr * I
        tau_load = load_func(t)
        domega_dt = (tau_motor + tau_load) / J_inertia
        return [dI_dt, domega_dt]

    sol = solve_ivp(motor_ode, [0, T_END], [0.0, 0.0],
                    method='RK45', rtol=1e-10, atol=1e-12, max_step=1e-5,
                    dense_output=True)
    return sol


# ─── MuJoCo 시뮬레이션 ───
def run_mujoco(xml_str, scenario, dt):
    # dt를 XML에 반영
    xml = xml_str.replace('timestep="0.0001"', f'timestep="{dt}"')
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)

    # 관성 확인
    mujoco.mj_forward(m, d)

    V_func = scenario["V_func"]
    load_func = scenario["load_func"]
    R_func = scenario["R_func"]

    is_coupled = (m.actuator_dynprm[0, 1] > 0 and m.actuator_dynprm[0, 2] > 0)  # Strategy D

    n_steps = int(T_END / dt)
    times = np.zeros(n_steps + 1)
    currents = np.zeros(n_steps + 1)
    omegas = np.zeros(n_steps + 1)

    times[0] = 0.0
    currents[0] = d.act[0] if m.na > 0 else 0.0
    omegas[0] = d.qvel[0]

    for step in range(n_steps):
        t = step * dt
        omega = d.qvel[0]
        R_t = R_func(t)
        V = np.clip(V_func(t), -V_BUS, V_BUS)

        # ctrl = (V - Ke*gr*omega) / R — 전압 기준 입력
        ctrl = (V - Ke_gr * omega) / R_t
        d.ctrl[0] = ctrl

        # 외부 토크
        d.qfrc_applied[0] = load_func(t)

        # 시나리오 3.5: R 런타임 변경
        tau_e_t = L / R_t
        m.actuator_dynprm[0, 0] = tau_e_t
        # coupled의 경우 dynprm[1], [2]는 그대로 (Ke_gr, L 불변)

        mujoco.mj_step(m, d)

        times[step + 1] = d.time
        currents[step + 1] = d.act[0] if m.na > 0 else 0.0
        omegas[step + 1] = d.qvel[0]

    return times, currents, omegas


# ─── 관성 추출 ───
def get_inertia(xml_str):
    m = mujoco.MjModel.from_xml_string(xml_str)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    # 1-DOF: qM is scalar
    J_inertia = d.qM[0]
    return J_inertia


# ─── 플로팅 ───
def plot_scenario(scenario_name, scenario, dt, dt_label,
                  t_rk, I_rk, w_rk,
                  t_fe, I_fe, w_fe,
                  t_cp, I_cp, w_cp):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"{scenario['title']}  —  dt={dt_label}", fontsize=13)

    # Current
    ax = axes[0, 0]
    ax.plot(t_rk * 1000, I_rk, 'k-', linewidth=2, label='RK45')
    ax.plot(t_cp * 1000, I_cp, 'b-', linewidth=1.2, label='coupled')
    ax.plot(t_fe * 1000, I_fe, 'r--', linewidth=1.2, label='filterexact')
    ax.set_ylabel('Current I [A]')
    ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Omega
    ax = axes[1, 0]
    ax.plot(t_rk * 1000, w_rk, 'k-', linewidth=2, label='RK45')
    ax.plot(t_cp * 1000, w_cp, 'b-', linewidth=1.2, label='coupled')
    ax.plot(t_fe * 1000, w_fe, 'r--', linewidth=1.2, label='filterexact')
    ax.set_ylabel('Angular velocity ω [rad/s]')
    ax.set_xlabel('Time [ms]')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Current error
    ax = axes[0, 1]
    # Interpolate RK45 at MuJoCo timesteps
    I_rk_fe = np.interp(t_fe, t_rk, I_rk)
    I_rk_cp = np.interp(t_cp, t_rk, I_rk)
    ax.plot(t_fe * 1000, np.abs(I_fe - I_rk_fe), 'r--', linewidth=1, label='|filterexact - RK45|')
    ax.plot(t_cp * 1000, np.abs(I_cp - I_rk_cp), 'b-', linewidth=1, label='|coupled - RK45|')
    ax.set_ylabel('|ΔI| [A]')
    ax.set_xlabel('Time [ms]')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Omega error
    ax = axes[1, 1]
    w_rk_fe = np.interp(t_fe, t_rk, w_rk)
    w_rk_cp = np.interp(t_cp, t_rk, w_rk)
    ax.plot(t_fe * 1000, np.abs(w_fe - w_rk_fe), 'r--', linewidth=1, label='|filterexact - RK45|')
    ax.plot(t_cp * 1000, np.abs(w_cp - w_rk_cp), 'b-', linewidth=1, label='|coupled - RK45|')
    ax.set_ylabel('|Δω| [rad/s]')
    ax.set_xlabel('Time [ms]')
    ax.set_yscale('log')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = PLOTS_DIR / f"{scenario_name}_{dt_label}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def compute_errors(t_mj, I_mj, w_mj, t_rk, I_rk, w_rk):
    I_rk_interp = np.interp(t_mj, t_rk, I_rk)
    w_rk_interp = np.interp(t_mj, t_rk, w_rk)
    err_I_max = np.max(np.abs(I_mj - I_rk_interp))
    err_w_max = np.max(np.abs(w_mj - w_rk_interp))
    err_I_rms = np.sqrt(np.mean((I_mj - I_rk_interp) ** 2))
    err_w_rms = np.sqrt(np.mean((w_mj - w_rk_interp) ** 2))
    return err_I_max, err_w_max, err_I_rms, err_w_rms


# ─── 메인 ───
def main():
    scenarios = make_scenarios()
    J_inertia = get_inertia(XML_FILTEREXACT)
    print(f"Inertia (geom + armature): {J_inertia:.6f} kg·m²")

    # 결과 저장
    all_results = {}  # (scenario, dt_label) -> {fe_errors, cp_errors}

    metrics_lines = []
    metrics_lines.append("=" * 80)
    metrics_lines.append("Phase 3: filterexact_coupled Verification Results")
    metrics_lines.append("=" * 80)
    metrics_lines.append(f"Inertia: {J_inertia:.6f} kg·m²")
    metrics_lines.append(f"Motor: R={R}, L={L}, Kt={Kt}, Ke={Ke}, gr={gr}")
    metrics_lines.append(f"Kt*gr={Kt_gr}, Ke*gr={Ke_gr}, tau_e={tau_e}")
    metrics_lines.append("")

    for sc_name, scenario in scenarios.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {sc_name} — {scenario['title']}")
        print(f"{'='*60}")

        # RK45 ground truth
        sol = solve_rk45(scenario, J_inertia)
        t_rk = sol.t
        I_rk = sol.y[0]
        w_rk = sol.y[1]
        print(f"  RK45: {len(t_rk)} points, t=[{t_rk[0]*1000:.3f}, {t_rk[-1]*1000:.3f}] ms")

        metrics_lines.append(f"\n{'─'*60}")
        metrics_lines.append(f"Scenario: {sc_name} — {scenario['title']}")
        metrics_lines.append(f"{'─'*60}")

        for dt, dt_label in zip(DT_LIST, DT_LABELS):
            print(f"\n  dt = {dt_label}:")

            # filterexact
            t_fe, I_fe, w_fe = run_mujoco(XML_FILTEREXACT, scenario, dt)
            fe_errs = compute_errors(t_fe, I_fe, w_fe, t_rk, I_rk, w_rk)

            # coupled
            t_cp, I_cp, w_cp = run_mujoco(XML_COUPLED, scenario, dt)
            cp_errs = compute_errors(t_cp, I_cp, w_cp, t_rk, I_rk, w_rk)

            # ratio
            ratio_I = fe_errs[0] / cp_errs[0] if cp_errs[0] > 1e-15 else float('inf')
            ratio_w = fe_errs[1] / cp_errs[1] if cp_errs[1] > 1e-15 else float('inf')

            print(f"    filterexact: max|ΔI|={fe_errs[0]:.6e}, max|Δω|={fe_errs[1]:.6e}")
            print(f"    coupled:     max|ΔI|={cp_errs[0]:.6e}, max|Δω|={cp_errs[1]:.6e}")
            print(f"    ratio (fe/cp): I={ratio_I:.3f}, ω={ratio_w:.3f}")

            all_results[(sc_name, dt_label)] = {
                "fe": fe_errs, "cp": cp_errs,
                "ratio_I": ratio_I, "ratio_w": ratio_w
            }

            metrics_lines.append(f"\n  dt = {dt_label}:")
            metrics_lines.append(f"    filterexact: max|ΔI|={fe_errs[0]:.6e}  max|Δω|={fe_errs[1]:.6e}  rms_I={fe_errs[2]:.6e}  rms_ω={fe_errs[3]:.6e}")
            metrics_lines.append(f"    coupled:     max|ΔI|={cp_errs[0]:.6e}  max|Δω|={cp_errs[1]:.6e}  rms_I={cp_errs[2]:.6e}  rms_ω={cp_errs[3]:.6e}")
            metrics_lines.append(f"    ratio (fe/cp): max_I={ratio_I:.4f}  max_ω={ratio_w:.4f}")

            # 플롯
            fname = plot_scenario(sc_name, scenario, dt, dt_label,
                                  t_rk, I_rk, w_rk,
                                  t_fe, I_fe, w_fe,
                                  t_cp, I_cp, w_cp)
            print(f"    Plot: {fname}")

    # ─── Error Heatmap ───
    print("\n\nGenerating error heatmap...")
    sc_names = list(scenarios.keys())
    heatmap_I = np.zeros((len(sc_names), len(DT_LIST)))
    heatmap_w = np.zeros((len(sc_names), len(DT_LIST)))

    for i, sc_name in enumerate(sc_names):
        for j, dt_label in enumerate(DT_LABELS):
            r = all_results[(sc_name, dt_label)]
            heatmap_I[i, j] = r["ratio_I"]
            heatmap_w[i, j] = r["ratio_w"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Error Ratio: max|filterexact error| / max|coupled error|  (>1 = coupled better)", fontsize=11)

    for ax, data, title in [(ax1, heatmap_I, "Current I"), (ax2, heatmap_w, "Angular velocity ω")]:
        im = ax.imshow(data, cmap='RdYlGn', aspect='auto', vmin=0.5, vmax=max(2.0, np.nanmax(data)))
        ax.set_xticks(range(len(DT_LABELS)))
        ax.set_xticklabels(DT_LABELS)
        ax.set_yticks(range(len(sc_names)))
        ax.set_yticklabels(sc_names)
        ax.set_xlabel("dt")
        ax.set_title(title)
        for ii in range(len(sc_names)):
            for jj in range(len(DT_LABELS)):
                val = data[ii, jj]
                color = 'white' if val > 1.5 else 'black'
                ax.text(jj, ii, f"{val:.2f}", ha='center', va='center', fontsize=9, color=color)
        fig.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "error_heatmap.png", dpi=150)
    plt.close()

    # ─── Convergence Plot ───
    print("Generating convergence plot...")
    fig, axes = plt.subplots(2, len(sc_names), figsize=(4 * len(sc_names), 7))
    fig.suptitle("Convergence: max error vs dt", fontsize=12)

    for i, sc_name in enumerate(sc_names):
        fe_I_errs = [all_results[(sc_name, dl)]["fe"][0] for dl in DT_LABELS]
        cp_I_errs = [all_results[(sc_name, dl)]["cp"][0] for dl in DT_LABELS]
        fe_w_errs = [all_results[(sc_name, dl)]["fe"][1] for dl in DT_LABELS]
        cp_w_errs = [all_results[(sc_name, dl)]["cp"][1] for dl in DT_LABELS]

        for row, (fe_e, cp_e, ylabel) in enumerate([
            (fe_I_errs, cp_I_errs, "max|ΔI| [A]"),
            (fe_w_errs, cp_w_errs, "max|Δω| [rad/s]"),
        ]):
            ax = axes[row, i]
            dts = np.array(DT_LIST)
            ax.loglog(dts * 1000, fe_e, 'r--o', label='filterexact', markersize=5)
            ax.loglog(dts * 1000, cp_e, 'b-s', label='coupled', markersize=5)

            # slope (convergence order)
            if len(dts) >= 2 and fe_e[-1] > 0 and fe_e[0] > 0:
                slope_fe = np.polyfit(np.log(dts), np.log(np.maximum(fe_e, 1e-20)), 1)[0]
                slope_cp = np.polyfit(np.log(dts), np.log(np.maximum(cp_e, 1e-20)), 1)[0]
                ax.set_title(f"{sc_name}\nslope: fe={slope_fe:.1f}, cp={slope_cp:.1f}", fontsize=9)

            ax.set_xlabel("dt [ms]")
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "convergence.png", dpi=150)
    plt.close()

    # ─── 메트릭 저장 ───
    metrics_lines.append("\n\n" + "=" * 80)
    metrics_lines.append("SUCCESS CRITERIA EVALUATION")
    metrics_lines.append("=" * 80)

    # 기준 1: 정확성
    all_better = True
    for key, r in all_results.items():
        if r["cp"][0] > r["fe"][0] * 1.01:  # 1% 마진
            all_better = False
            metrics_lines.append(f"  FAIL criterion 1: {key} — coupled I error ({r['cp'][0]:.6e}) > filterexact ({r['fe'][0]:.6e})")
    if all_better:
        metrics_lines.append("  PASS criterion 1: coupled error <= filterexact error in all cases")

    # 기준 2: dt 강건성 (핵심)
    metrics_lines.append("")
    for sc_name in sc_names:
        r_small = all_results[(sc_name, "0.1ms")]
        r_large = all_results[(sc_name, "5.0ms")]
        if r_large["ratio_I"] > r_small["ratio_I"]:
            metrics_lines.append(f"  PASS criterion 2 ({sc_name}): ratio grows with dt (0.1ms: {r_small['ratio_I']:.3f} → 5ms: {r_large['ratio_I']:.3f})")
        else:
            metrics_lines.append(f"  FAIL criterion 2 ({sc_name}): ratio does NOT grow with dt (0.1ms: {r_small['ratio_I']:.3f} → 5ms: {r_large['ratio_I']:.3f})")

    # 기준 3: 무퇴행
    metrics_lines.append("")
    for sc_name in sc_names:
        r = all_results[(sc_name, "0.1ms")]
        if r["ratio_I"] >= 0.95:
            metrics_lines.append(f"  PASS criterion 3 ({sc_name}): no regression at dt=0.1ms (ratio={r['ratio_I']:.3f})")
        else:
            metrics_lines.append(f"  FAIL criterion 3 ({sc_name}): regression at dt=0.1ms (ratio={r['ratio_I']:.3f} < 0.95)")

    metrics_text = "\n".join(metrics_lines)
    (RESULTS_DIR / "metrics.txt").write_text(metrics_text)
    print(f"\n\nMetrics saved to {RESULTS_DIR / 'metrics.txt'}")
    print("\n" + metrics_text)


if __name__ == "__main__":
    main()
