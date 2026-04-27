"""MuJoCo coupled implicit solver 행렬 덤프.

mj_step() 후 내부 행렬(qDeriv, qLU, M 등)을 텍스트 파일로 저장.
qDeriv는 mj_step 내부의 mj_implicitSkip()에서 채워지며,
step 완료 후에도 d.qDeriv에 값이 유지됨.

Usage:
    python scripts/dump_solver_matrices.py
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import mujoco
from pathlib import Path
from io import StringIO

np.set_printoptions(precision=8, linewidth=300, suppress=True)

OUT_PATH = Path("solver_comparison/phase4_results/solver_trace/solver_trace_detailed.txt")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  Labels
# ═══════════════════════════════════════════════════════════════

DOF_LABELS = [
    "bx", "by", "bz", "brx", "bry", "brz",
    "FLh", "FLt", "FLc",
    "FRh", "FRt", "FRc",
    "RLh", "RLt", "RLc",
    "RRh", "RRt", "RRc",
]

# Actuator order: FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, FR_thigh, ...
MOTOR_LABELS = [
    "M_FLh", "M_FRh", "M_RLh", "M_RRh",
    "M_FLt", "M_FRt", "M_RLt", "M_RRt",
    "M_FLc", "M_FRc", "M_RLc", "M_RRc",
]

QPOS_LABELS = [
    "px", "py", "pz", "qw", "qx", "qy", "qz",
    "FLh", "FLt", "FLc",
    "FRh", "FRt", "FRc",
    "RLh", "RLt", "RLc",
    "RRh", "RRt", "RRc",
]


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def sparse_to_dense(flat, rownnz, rowadr, colind, n):
    dense = np.zeros((n, n))
    for i in range(n):
        adr = rowadr[i]
        for k in range(rownnz[i]):
            j = colind[adr + k]
            dense[i, j] = flat[adr + k]
    return dense


def qM_to_dense(m, d):
    nv = m.nv
    dense = np.zeros((nv, nv))
    mujoco.mj_fullM(m, dense, d.qM)
    return dense


def moment_to_dense(d, nu, nv):
    J = np.zeros((nu, nv))
    for i in range(nu):
        adr = d.moment_rowadr[i]
        for k in range(d.moment_rownnz[i]):
            j = d.moment_colind[adr + k]
            J[i, j] = d.actuator_moment[adr + k]
    return J


def fmt_matrix(mat, row_labels, col_labels, name, precision=8, force_dense=False):
    """Tab-separated matrix with labels. Sparse matrices use '.' for zeros."""
    nz_ratio = np.count_nonzero(np.abs(mat) > 1e-12) / max(mat.size, 1)
    use_sparse = (nz_ratio < 0.5) and not force_dense
    col_w = 14  # column width

    lines = [f"{name}  [nonzero: {np.count_nonzero(np.abs(mat) > 1e-12)}/{mat.size}]"]
    header = " " * 6 + "".join(f"{c:>{col_w}s}" for c in col_labels)
    lines.append(header)

    for i, row in enumerate(mat):
        parts = []
        for v in row:
            if use_sparse and abs(v) < 1e-12:
                parts.append(f"{'·':>{col_w}s}")
            elif abs(v) < 1e-4 and abs(v) > 1e-12:
                parts.append(f"{'*' + f'{v:.4e}':>{col_w}s}")
            else:
                parts.append(f"{'*' + f'{v:.{precision}f}' if use_sparse and abs(v) > 1e-12 else f'{v:.{precision}f}':>{col_w}s}")
        lines.append(f"{row_labels[i]:>5s} " + "".join(parts))
    return "\n".join(lines)


def fmt_vector(vec, labels, name, precision=8):
    lines = [f"{name}"]
    for i, v in enumerate(vec):
        lines.append(f"  {labels[i]:>8s} = {v:.{precision}f}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
#  Load model
# ═══════════════════════════════════════════════════════════════

def load_go2():
    from mjlab.entity.entity import Entity
    from src.assets.robots.unitree_go2.go2_constants import get_go2_coupled_electric_robot_cfg
    cfg = get_go2_coupled_electric_robot_cfg()
    robot = Entity(cfg)
    m = robot.spec.compile()
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
    m.opt.timestep = 0.0001
    return m


# ═══════════════════════════════════════════════════════════════
#  Compute Schur, B, C, D_elec
# ═══════════════════════════════════════════════════════════════

def compute_all_blocks(m, d):
    dt = m.opt.timestep
    nu, nv = m.nu, m.nv
    J = moment_to_dense(d, nu, nv)

    # Per-motor parameters
    params = []
    for i in range(nu):
        dynprm = m.actuator_dynprm[i]
        tau_e = max(1e-14, dynprm[0])
        Ke_gr = dynprm[1]
        L_val = dynprm[2]
        Kt_gr = m.actuator_gainprm[i, 0]
        R_val = L_val / tau_e if tau_e > 0 else 0
        params.append({"Kt_gr": Kt_gr, "Ke_gr": Ke_gr, "R": R_val, "L": L_val, "tau_e": tau_e})

    # Full coupled system:
    #   [A  B] [Δv ]   [f]
    #   [C  D] [ΔI ] = [g]
    #
    # A = M - dt·∂f/∂v          (mechanical, 18×18)
    # B = dt · Kt·gr · Jᵀ       (torque from current, 18×12)
    #     positive: more current → more torque → helps acceleration
    # C = -dt · Ke·gr/L · J     (back-EMF coupling, 12×18)
    #     negative: more speed → more back-EMF → reduces dI/dt
    # D = I + dt·R/L             (current self-decay, 12×12)
    #
    # Schur complement added to A: -B·D⁻¹·C (positive definite)
    #   = -dt·Kt·gr·Jᵀ · (1/D) · (-dt·Ke·gr/L·J)
    #   = dt²·Kt·gr·Ke·gr/(L·D) · JᵀJ  ← positive, matches C code

    # B = dt * diag(Kt_gr) * J^T  → (nv × nu)
    B = np.zeros((nv, nu))
    for i in range(nu):
        B[:, i] = dt * params[i]["Kt_gr"] * J[i, :]

    # C = -dt * diag(Ke_gr / L) * J  → (nu × nv)
    C = np.zeros((nu, nv))
    for i in range(nu):
        if params[i]["L"] > 0:
            C[i, :] = -dt * params[i]["Ke_gr"] / params[i]["L"] * J[i, :]

    # D_elec = I + dt * diag(R/L)  → (nu × nu)
    D_elec = np.eye(nu)
    for i in range(nu):
        if params[i]["L"] > 0:
            D_elec[i, i] += dt * params[i]["R"] / params[i]["L"]

    # -B * D_elec^{-1} * C = Schur complement (positive)
    D_elec_inv = np.diag(1.0 / np.diag(D_elec))
    BDinvC = -(B @ D_elec_inv @ C)  # note the minus sign

    # Schur via scale * J^T J
    schur_JTJ = np.zeros((nv, nv))
    for i in range(nu):
        if params[i]["Ke_gr"] != 0 and params[i]["L"] > 0:
            d_inv = 1.0 / D_elec[i, i]
            scale = dt * dt * params[i]["Kt_gr"] * params[i]["Ke_gr"] / params[i]["L"] * d_inv
            Ji = J[i, :]
            schur_JTJ += scale * np.outer(Ji, Ji)

    return J, B, C, D_elec, D_elec_inv, BDinvC, schur_JTJ, params


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    import src.tasks

    m = load_go2()
    d = mujoco.MjData(m)

    nq, nv, nu, na = m.nq, m.nv, m.nu, m.na
    dt = m.opt.timestep

    out = StringIO()
    def W(s=""):
        out.write(s + "\n")

    # ── Header ──
    W("=" * 80)
    W("MuJoCo Coupled Implicit Solver — Matrix Dump")
    W("=" * 80)
    W()
    W(f"nq = {nq}  (3 pos + 4 quat + 12 joints = 19)")
    W(f"nv = {nv}  (3 lin_vel + 3 ang_vel + 12 joint_vel = 18)")
    W(f"nu = {nu}  (12 actuators)")
    W(f"na = {na}  (12 activation states = motor currents)")
    W(f"dt = {dt}  ({dt*1000}ms)")
    W(f"integrator = implicit")
    W()

    # Motor parameters
    mujoco.mj_forward(m, d)
    _, _, _, _, _, _, _, params = compute_all_blocks(m, d)

    W("=" * 80)
    W("Motor Parameters (identical for all 12 motors)")
    W("=" * 80)
    p = params[0]
    W(f"  Kt·gr     = {p['Kt_gr']:.8f}  (gainprm[0])")
    W(f"  Ke·gr     = {p['Ke_gr']:.8f}  (dynprm[1])")
    W(f"  R         = {p['R']:.8f}  Ohm")
    W(f"  L         = {p['L']:.8e}  H")
    W(f"  tau_e     = {p['tau_e']:.8e}  s  (= L/R)")
    W(f"  D_elec    = 1 + dt·R/L = 1 + dt/tau_e = {1 + dt/p['tau_e']:.8f}")
    W(f"  schur_scale = dt²·Kt·gr·Ke·gr / (L·D_elec) = {dt**2 * p['Kt_gr'] * p['Ke_gr'] / (p['L'] * (1 + dt/p['tau_e'])):.10e}")
    W()

    W("=" * 80)
    W("DOF Labels (nv=18)")
    W("=" * 80)
    for i, label in enumerate(DOF_LABELS):
        jntid = m.dof_jntid[i]
        jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jntid) or "?"
        W(f"  DOF {i:2d} = {label:5s} ← {jname}")
    W()

    W("=" * 80)
    W("Actuator Labels (nu=12)")
    W("=" * 80)
    for i, label in enumerate(MOTOR_LABELS):
        aname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "?"
        dof = d.moment_colind[d.moment_rowadr[i]]
        W(f"  Act {i:2d} = {label:7s} ← {aname:25s} → DOF {dof:2d} ({DOF_LABELS[dof]})")
    W()

    # ── ctrl ──
    d.ctrl[:] = 0.5

    # ── 3 timestep loop ──
    for step in range(3):
        # Save state before step
        qpos_before = d.qpos.copy()
        qvel_before = d.qvel.copy()
        act_before = d.act.copy()

        # Execute full step — qDeriv, qLU are filled inside mj_implicitSkip
        mujoco.mj_step(m, d)

        # Now read matrices that were computed during the step
        M_dense = qM_to_dense(m, d)
        D_dense = sparse_to_dense(d.qDeriv, m.D_rownnz, m.D_rowadr, m.D_colind, nv)
        qLU_dense = sparse_to_dense(d.qLU, m.D_rownnz, m.D_rowadr, m.D_colind, nv)

        J, B, C, D_elec, D_elec_inv, BDinvC, schur_JTJ, _ = compute_all_blocks(m, d)
        D_without_schur = D_dense - schur_JTJ

        # Forces (read AFTER step — these are from the step just completed)
        # Note: qfrc values are from mj_step1 phase of this step
        qacc = d.qacc.copy()
        qvel_after = d.qvel.copy()
        qpos_after = d.qpos.copy()
        act_after = d.act.copy()

        # ctrl_eff reconstruction (coupled correction)
        ctrl_eff = np.zeros(nu)
        for i in range(nu):
            dynprm = m.actuator_dynprm[i]
            Ke_gr = dynprm[1]
            L_val = dynprm[2]
            tau_e = max(1e-14, dynprm[0])
            R_val = L_val / tau_e
            dof = d.moment_colind[d.moment_rowadr[i]]
            Jqacc = qacc[dof]  # J is identity for 1:1 joints
            ctrl_eff[i] = d.ctrl[i] - Ke_gr * dt * Jqacc / R_val

        force = np.array([m.actuator_gainprm[i, 0] * act_after[i] for i in range(nu)])

        W()
        W("=" * 80)
        W(f"===== TIMESTEP {step} =====")
        W("=" * 80)
        W()

        # ── 18×18 matrices ──
        W(fmt_matrix(M_dense, DOF_LABELS, DOF_LABELS,
                     f"M (mass matrix, {nv}x{nv}) — timestep {step}", force_dense=True))
        W()
        W(fmt_matrix(D_dense, DOF_LABELS, DOF_LABELS,
                     f"D_qDeriv (with Schur, {nv}x{nv}) — timestep {step}"))
        W()
        W(fmt_matrix(schur_JTJ, DOF_LABELS, DOF_LABELS,
                     f"Schur_JTJ = -B·D_elec⁻¹·C ({nv}x{nv}) — timestep {step}"))
        W()
        W(fmt_matrix(D_without_schur, DOF_LABELS, DOF_LABELS,
                     f"D_qDeriv_without_Schur (D - Schur, {nv}x{nv}) — timestep {step}"))
        W()
        W(fmt_matrix(qLU_dense, DOF_LABELS, DOF_LABELS,
                     f"qLU (M - dt·D, {nv}x{nv}) — timestep {step}", force_dense=True))
        W()

        # ── B, C, D_elec ──
        W(fmt_matrix(B, DOF_LABELS, MOTOR_LABELS,
                     f"B = dt · Jᵀ · diag(Kt·gr) ({nv}x{nu}) — timestep {step}"))
        W()
        W(fmt_matrix(C, MOTOR_LABELS, DOF_LABELS,
                     f"C = -dt · diag(Ke·gr/L) · J ({nu}x{nv}) — timestep {step}"))
        W()
        W(fmt_matrix(D_elec, MOTOR_LABELS, MOTOR_LABELS,
                     f"D_elec = I + dt·(R/L)·I ({nu}x{nu}) — timestep {step}"))
        W()

        # ── J ──
        W(fmt_matrix(J, MOTOR_LABELS, DOF_LABELS,
                     f"J (actuator moment arm, {nu}x{nv}) — timestep {step}"))
        W()

        # ── BDinvC verification ──
        err = np.max(np.abs(BDinvC - schur_JTJ))
        W(f"-B * D_elec^{{-1}} * C verification: max|(-BDinvC) - Schur_JTJ| = {err:.2e}")
        W()
        # Diagonal comparison for motor DOFs
        W(f"  Diagonal comparison (motor DOFs only):")
        W(f"  {'DOF':>5s}  {'Schur_JTJ':>14s}  {'-BDinvC':>14s}  {'diff':>14s}")
        for i in range(nv):
            s = schur_JTJ[i, i]
            b = BDinvC[i, i]
            if abs(s) > 1e-12 or abs(b) > 1e-12:
                W(f"  {DOF_LABELS[i]:>5s}  {s:>14.10e}  {b:>14.10e}  {s-b:>14.2e}")
        W()

        # ── Vectors ──
        W(fmt_vector(d.qfrc_smooth, DOF_LABELS, f"qfrc_smooth ({nv}x1) — timestep {step}"))
        W()
        W(fmt_vector(d.qfrc_passive, DOF_LABELS, f"qfrc_passive ({nv}x1) — timestep {step}"))
        W()
        W(fmt_vector(d.qfrc_bias, DOF_LABELS, f"qfrc_bias ({nv}x1) — timestep {step}"))
        W()
        W(fmt_vector(d.qfrc_actuator, DOF_LABELS, f"qfrc_actuator ({nv}x1) — timestep {step}"))
        W()
        W(fmt_vector(d.qfrc_applied, DOF_LABELS, f"qfrc_applied ({nv}x1) — timestep {step}"))
        W()
        W(fmt_vector(d.qfrc_constraint, DOF_LABELS, f"qfrc_constraint ({nv}x1) — timestep {step}"))
        W()
        W(fmt_vector(qacc, DOF_LABELS, f"qacc ({nv}x1) — timestep {step}"))
        W()
        W(fmt_vector(qvel_before, DOF_LABELS, f"qvel_before ({nv}x1) — timestep {step}"))
        W()
        W(fmt_vector(qvel_after, DOF_LABELS, f"qvel_after ({nv}x1) — timestep {step}"))
        W()
        W(fmt_vector(qpos_before, QPOS_LABELS, f"qpos_before ({nq}x1) — timestep {step}"))
        W()
        W(fmt_vector(qpos_after, QPOS_LABELS, f"qpos_after ({nq}x1) — timestep {step}"))
        W()
        W(fmt_vector(d.ctrl, MOTOR_LABELS, f"ctrl ({nu}x1) — timestep {step}"))
        W()
        W(fmt_vector(ctrl_eff, MOTOR_LABELS, f"ctrl_eff (corrected, {nu}x1) — timestep {step}"))
        W()
        W(fmt_vector(act_before, MOTOR_LABELS, f"act_before (current, {na}x1) — timestep {step}"))
        W()
        W(fmt_vector(act_after, MOTOR_LABELS, f"act_after (current, {na}x1) — timestep {step}"))
        W()
        W(fmt_vector(force, MOTOR_LABELS, f"force = Kt*gr * act ({nu}x1) — timestep {step}"))
        W()

    # ── Write file ──
    text = out.getvalue()
    OUT_PATH.write_text(text)
    print(f"Saved to {OUT_PATH} ({len(text)} bytes, {text.count(chr(10))} lines)")

    # Also save locally
    local = Path("solver_comparison/phase4_results/solver_trace/solver_trace_detailed.txt")
    local.write_text(text)
    print(f"Also saved to {local}")


if __name__ == "__main__":
    main()
