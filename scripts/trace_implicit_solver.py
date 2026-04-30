"""Schur complement coupled implicit solver 3-timestep trace.

Go2 standing pose에서 ctrl=0.5를 모든 모터에 인가하고,
implicit solver의 내부 행렬/벡터를 매 timestep 추적합니다.

Usage:
    python scripts/trace_implicit_solver.py
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

np.set_printoptions(precision=6, linewidth=200, suppress=True)

OUT_DIR = Path("solver_comparison/phase4_results/solver_trace")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  1. 모델 로드 (Go2 coupled electric)
# ═══════════════════════════════════════════════════════════════

def load_go2():
    from mjlab.entity.entity import Entity
    from src.assets.robots.unitree_go2.go2_constants import get_go2_coupled_electric_robot_cfg
    cfg = get_go2_coupled_electric_robot_cfg()
    robot = Entity(cfg)
    m = robot.spec.compile()
    # implicit integrator 강제 설정
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    m.opt.timestep = 0.0001  # 0.1ms
    return m

# ═══════════════════════════════════════════════════════════════
#  2. Sparse → Dense 변환 유틸리티
# ═══════════════════════════════════════════════════════════════

def sparse_to_dense(flat, rownnz, rowadr, colind, n):
    """MuJoCo sparse (D-format: full rows) → dense matrix."""
    dense = np.zeros((n, n))
    for i in range(n):
        adr = rowadr[i]
        for k in range(rownnz[i]):
            j = colind[adr + k]
            dense[i, j] = flat[adr + k]
    return dense


def qM_to_dense(m, d):
    """d.qM (lower-triangular sparse) → dense symmetric."""
    nv = m.nv
    dense = np.zeros((nv, nv))
    mujoco.mj_fullM(m, dense, d.qM)
    return dense


def moment_to_dense(d, nu, nv):
    """Sparse actuator_moment → dense J (nu × nv)."""
    J = np.zeros((nu, nv))
    for i in range(nu):
        adr = d.moment_rowadr[i]
        for k in range(d.moment_rownnz[i]):
            j = d.moment_colind[adr + k]
            J[i, j] = d.actuator_moment[adr + k]
    return J


# ═══════════════════════════════════════════════════════════════
#  3. Schur complement 직접 계산
# ═══════════════════════════════════════════════════════════════

def compute_schur_terms(m, d):
    """각 모터의 Schur complement 항을 직접 계산."""
    dt = m.opt.timestep
    nu = m.nu
    nv = m.nv
    J = moment_to_dense(d, nu, nv)

    schur_JTJ = np.zeros((nv, nv))
    motor_info = []

    for i in range(nu):
        dynprm = m.actuator_dynprm[i]
        tau_e = max(1e-14, dynprm[0])
        Ke_gr = dynprm[1]
        L_val = dynprm[2]
        Kt_gr = m.actuator_gainprm[i, 0]
        R_val = L_val / tau_e

        info = {
            "joint_dof": int(d.moment_colind[d.moment_rowadr[i]]),
            "Kt_gr": Kt_gr, "Ke_gr": Ke_gr, "R": R_val, "L": L_val, "tau_e": tau_e,
        }

        if Ke_gr != 0 and L_val > 0:
            D_elec = 1.0 + dt / tau_e
            d_inv = 1.0 / D_elec
            schur_scale = dt * dt * Kt_gr * Ke_gr / L_val * d_inv
            info["D_elec"] = D_elec
            info["schur_scale"] = schur_scale

            # J_i^T · J_i (rank-1 outer product)
            Ji = J[i, :]  # (nv,)
            JTJ_i = np.outer(Ji, Ji) * schur_scale
            schur_JTJ += JTJ_i
        else:
            info["D_elec"] = 0
            info["schur_scale"] = 0

        motor_info.append(info)

    return schur_JTJ, motor_info, J


def compute_BDinvC(m, d):
    """B·D⁻¹·C를 명시적으로 계산하여 Schur complement와 일치 확인.

    B = -dt · Kt·gr · Jᵀ  (nv × na)
    C = -dt · Ke·gr / L · J  (na × nv)
    D = I + dt·R/L  (na × na, diagonal)
    """
    dt = m.opt.timestep
    nu = m.nu
    nv = m.nv
    J = moment_to_dense(d, nu, nv)

    BDinvC = np.zeros((nv, nv))
    for i in range(nu):
        dynprm = m.actuator_dynprm[i]
        tau_e = max(1e-14, dynprm[0])
        Ke_gr = dynprm[1]
        L_val = dynprm[2]
        Kt_gr = m.actuator_gainprm[i, 0]

        if Ke_gr == 0 or L_val <= 0:
            continue

        R_val = L_val / tau_e
        D_ii = 1.0 + dt * R_val / L_val  # = 1 + dt/tau_e

        # B_i = -dt * Kt_gr * J_i^T  (nv × 1)
        # C_i = -dt * Ke_gr / L * J_i  (1 × nv)
        # B_i · D_ii⁻¹ · C_i = dt² · Kt_gr · Ke_gr / (L · D_ii) · J_i^T · J_i
        Ji = J[i, :]
        scale = dt * dt * Kt_gr * Ke_gr / (L_val * D_ii)
        BDinvC += scale * np.outer(Ji, Ji)

    return BDinvC


# ═══════════════════════════════════════════════════════════════
#  4. Heatmap 시각화
# ═══════════════════════════════════════════════════════════════

def plot_matrix(mat, title, path, vabs=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    if vabs is None:
        vabs = max(abs(mat.min()), abs(mat.max()), 1e-10)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("DOF column")
    ax.set_ylabel("DOF row")
    # DOF labels
    labels = [f"bx","by","bz","brx","bry","brz",
              "FLh","FRh","RLh","FLt","FRt","RLt","FLc","FRc","RLc","RRh","RRt","RRc"]
    if mat.shape[0] == 18:
        ax.set_xticks(range(18)); ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_yticks(range(18)); ax.set_yticklabels(labels, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  5. 메인: 3 timestep 추적
# ═══════════════════════════════════════════════════════════════

def main():
    import src.tasks  # register tasks

    m = load_go2()
    d = mujoco.MjData(m)

    nq, nv, nu, na = m.nq, m.nv, m.nu, m.na
    dt = m.opt.timestep

    print("=" * 70)
    print("Go2 Coupled Implicit Solver Trace")
    print("=" * 70)
    print(f"nq={nq} (19: 3 pos + 4 quat + 12 joints)")
    print(f"nv={nv} (18: 3 lin + 3 ang + 12 joints, quat → 3 angular vel)")
    print(f"nu={nu} (12 actuators)")
    print(f"na={na} (12 activation states = currents)")
    print(f"dt={dt} ({dt*1000}ms)")
    print(f"integrator: implicit")
    print(f"nD={m.nD} (sparse qDeriv elements)")
    print()

    # 초기화
    mujoco.mj_forward(m, d)

    # ── 모터 파라미터 출력 ─────────────────────────────────────
    print("=" * 70)
    print("Motor Parameters")
    print("=" * 70)
    schur_JTJ_0, motor_info, J = compute_schur_terms(m, d)
    for i, info in enumerate(motor_info):
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  Motor {i:2d} ({jname:>20s}) → DOF {info['joint_dof']:2d} | "
              f"Kt·gr={info['Kt_gr']:.5f}  Ke·gr={info['Ke_gr']:.5f}  "
              f"R={info['R']:.3f}  L={info['L']:.1e}  τ_e={info['tau_e']:.6f}s")
    print()

    print("J (actuator moment arm, 12×18):")
    print(J)
    print()

    print(f"D_elec = 1 + dt/τ_e = {motor_info[0]['D_elec']:.6f}")
    print(f"schur_scale = dt²·Kt·Ke·gr²/(L·D_elec) = {motor_info[0]['schur_scale']:.10e}")
    print()

    # BDinvC 검증
    BDinvC = compute_BDinvC(m, d)
    print("Schur verification: ||schur_JTJ - BDinvC|| =", np.linalg.norm(schur_JTJ_0 - BDinvC))
    print()

    # ctrl 설정
    d.ctrl[:] = 0.5
    print(f"ctrl = {d.ctrl}")
    print()

    # ── 3 timestep 추적 ──────────────────────────────────────
    for step in range(3):
        print("=" * 70)
        print(f"TIMESTEP {step}")
        print("=" * 70)

        # 상태 저장 (before)
        qpos_before = d.qpos.copy()
        qvel_before = d.qvel.copy()
        act_before = d.act.copy()

        # ── step1: forward 계산 (position/velocity dependent) ──
        mujoco.mj_step1(m, d)

        # 이 시점에서 qfrc_smooth, qfrc_constraint가 계산됨
        # qDeriv, qLU도 implicit integrator에 의해 채워짐

        M_dense = qM_to_dense(m, d)
        D_dense = sparse_to_dense(d.qDeriv, m.D_rownnz, m.D_rowadr, m.D_colind, nv)
        qLU_dense = sparse_to_dense(d.qLU, m.D_rownnz, m.D_rowadr, m.D_colind, nv)

        # Schur complement 항 계산
        schur_JTJ, _, J = compute_schur_terms(m, d)

        # 우변 f
        f_rhs = d.qfrc_smooth + d.qfrc_constraint

        print(f"\n--- Forces (nv={nv}) ---")
        print(f"  qfrc_bias     = {d.qfrc_bias}")
        print(f"  qfrc_passive  = {d.qfrc_passive}")
        print(f"  qfrc_actuator = {d.qfrc_actuator}")
        print(f"  qfrc_applied  = {d.qfrc_applied}")
        print(f"  qfrc_smooth   = {d.qfrc_smooth}")
        print(f"  qfrc_constraint = {d.qfrc_constraint}")
        print(f"  f (rhs)       = {f_rhs}")

        print(f"\n--- Mass matrix M diagonal ---")
        print(f"  {np.diag(M_dense)}")
        print(f"  M nonzero count: {np.count_nonzero(M_dense)}/{nv*nv}")

        print(f"\n--- D = dqDeriv (includes Schur) diagonal ---")
        print(f"  {np.diag(D_dense)}")

        print(f"\n--- Schur complement JᵀJ diagonal ---")
        print(f"  {np.diag(schur_JTJ)}")
        print(f"  Schur nonzero: {np.count_nonzero(np.abs(schur_JTJ) > 1e-15)}/{nv*nv}")

        print(f"\n--- qLU = M - dt·D diagonal ---")
        print(f"  {np.diag(qLU_dense)}")

        # qLU 검증: M - dt·D와 비교
        expected_qLU = M_dense - dt * D_dense
        qLU_err = np.linalg.norm(qLU_dense - expected_qLU)
        print(f"  ||qLU - (M - dt·D)|| = {qLU_err:.2e}")

        # ── step2: constraint solve + state advance ──
        mujoco.mj_step2(m, d)

        # 상태 저장 (after)
        qacc_after = d.qacc.copy()
        qvel_after = d.qvel.copy()
        qpos_after = d.qpos.copy()
        act_after = d.act.copy()

        print(f"\n--- Solution ---")
        print(f"  qacc  = {qacc_after}")
        print(f"\n--- Activation (current) ---")
        print(f"  act_before = {act_before}")
        print(f"  act_after  = {act_after}")
        print(f"  act_delta  = {act_after - act_before}")

        print(f"\n--- Velocity ---")
        print(f"  qvel_before = {qvel_before}")
        print(f"  qvel_after  = {qvel_after}")
        print(f"  qvel_delta  = {qvel_after - qvel_before}")

        print(f"\n--- Position ---")
        print(f"  qpos_before = {qpos_before}")
        print(f"  qpos_after  = {qpos_after}")
        print()

        # ── 행렬 heatmap 저장 (step 0만) ──
        if step == 0:
            plot_matrix(M_dense, f"M (mass matrix, {nv}x{nv})", OUT_DIR / "M_heatmap.png")
            plot_matrix(D_dense, f"D = qDeriv (with Schur, {nv}x{nv})", OUT_DIR / "D_heatmap.png")
            plot_matrix(schur_JTJ, f"Schur complement term (dt²·Kt·Ke·gr²/(L·D_elec)·JᵀJ)", OUT_DIR / "Schur_heatmap.png")
            plot_matrix(qLU_dense, f"qLU = M - dt·D ({nv}x{nv})", OUT_DIR / "qLU_heatmap.png")
            plot_matrix(M_dense - dt * D_dense, f"M - dt·D (manual, {nv}x{nv})", OUT_DIR / "M_minus_dtD_heatmap.png")
            print(f"  Heatmaps saved to {OUT_DIR}")

    # ── 3 timestep 변화 요약 ──
    print("\n" + "=" * 70)
    print("3-TIMESTEP SUMMARY")
    print("=" * 70)
    print(f"  Final qvel[6:18] (joints) = {d.qvel[6:]}")
    print(f"  Final act (currents) = {d.act}")
    print(f"  Final qfrc_actuator = {d.qfrc_actuator}")
    print()


if __name__ == "__main__":
    main()
