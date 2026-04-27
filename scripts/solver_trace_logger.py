"""MuJoCo Solver Trace Logger — Scenario A (PD) vs B (Coupled Electric).

Dumps implicit integrator internals around control target transitions.
Output format matches the reference solver_trace_detailed.txt exactly.

Usage:
  conda activate mjlab
  python scripts/solver_trace_logger.py
"""

import mujoco
import numpy as np
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────
XML_PATH = str(Path(__file__).resolve().parent.parent /
               "src/assets/robots/unitree_go2/xmls/scene_go2.xml")

OUTPUT_DIR = Path("results/solver_trace")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Control targets: two poses to switch between
# Order: FL_hip, FL_thigh, FL_calf, FR_hip, FR_thigh, FR_calf,
#         RL_hip, RL_thigh, RL_calf, RR_hip, RR_thigh, RR_calf
# (matches body/joint order in XML, NOT actuator order)
POSE_A = np.array([0.1, 0.9, -1.8,  -0.1, 0.9, -1.8,
                    0.1, 0.9, -1.8,  -0.1, 0.9, -1.8])  # standing
POSE_B = np.array([0.1, 1.1, -2.0,  -0.1, 1.1, -2.0,
                    0.1, 1.1, -2.0,  -0.1, 1.1, -2.0])  # slightly crouched

KP_HIP, KD_HIP = 20.0, 1.0
KP_THIGH, KD_THIGH = 20.0, 1.0
KP_CALF, KD_CALF = 40.0, 2.0
# PD gains per joint (FL_h,FL_t,FL_c, FR_h,FR_t,FR_c, RL_h,RL_t,RL_c, RR_h,RR_t,RR_c)
KP_JOINT = np.array([KP_HIP, KP_THIGH, KP_CALF] * 4)
KD_JOINT = np.array([KD_HIP, KD_THIGH, KD_CALF] * 4)

POLICY_DT = 0.005      # 5ms control period
DT_PHYSICS = 0.0005    # 0.5ms substep
SUBSTEPS = int(POLICY_DT / DT_PHYSICS)  # 10
INIT_HEIGHT = 0.32

# Dump window: policy steps around transition
SETTLE_POLICY_STEPS = 20  # let robot settle before transition
DUMP_BEFORE = 3   # substeps before transition
DUMP_AFTER = 12   # substeps after transition (total window ~15)

# Motor parameters for coupled scenario
TAU_E = 0.000333
KT_GR = 0.81024
KE_GR = 0.81024
L_VAL = 0.0001
I_MAX = 29.0
R_VAL = L_VAL / TAU_E  # ~0.3003 Ohm

# DOF labels (order matches MuJoCo body traversal)
DOF_LABELS = ["bx", "by", "bz", "brx", "bry", "brz",
              "FLh", "FLt", "FLc", "FRh", "FRt", "FRc",
              "RLh", "RLt", "RLc", "RRh", "RRt", "RRc"]

# qpos labels (7 floating-base + 12 joints)
QPOS_LABELS = ["px", "py", "pz", "qw", "qx", "qy", "qz",
               "FLh", "FLt", "FLc", "FRh", "FRt", "FRc",
               "RLh", "RLt", "RLc", "RRh", "RRt", "RRc"]


# ── Helpers ────────────────────────────────────────────────────────
def get_joint_addrs(model):
    """Get qpos/dof addresses for each actuator's joint."""
    qpos_addrs, dof_addrs = [], []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        qpos_addrs.append(model.jnt_qposadr[jid])
        dof_addrs.append(model.jnt_dofadr[jid])
    return np.array(qpos_addrs), np.array(dof_addrs)


def build_actuator_labels(model):
    """Build actuator label list from model joint names."""
    labels = []
    joint_to_short = {
        "FL_hip_joint": "M_FLh", "FL_thigh_joint": "M_FLt", "FL_calf_joint": "M_FLc",
        "FR_hip_joint": "M_FRh", "FR_thigh_joint": "M_FRt", "FR_calf_joint": "M_FRc",
        "RL_hip_joint": "M_RLh", "RL_thigh_joint": "M_RLt", "RL_calf_joint": "M_RLc",
        "RR_hip_joint": "M_RRh", "RR_thigh_joint": "M_RRt", "RR_calf_joint": "M_RRc",
    }
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        labels.append(joint_to_short.get(jname, f"A{i}"))
    return labels


def build_joint_to_pose_map(model):
    """Build mapping from actuator index to pose index.

    Pose order follows body/joint order: FL(h,t,c), FR(h,t,c), RL(h,t,c), RR(h,t,c).
    Actuator order may differ (e.g., FR first in XML).
    Returns array where pose_idx[act_i] gives the index into POSE_A/B arrays.
    """
    joint_order = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    pose_map = []
    for i in range(model.nu):
        jid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        pose_map.append(joint_order.index(jname))
    return pose_map


def sparse_to_dense(data_arr, nv, rownnz, rowadr, colind):
    """Convert MuJoCo sparse D-format to dense matrix."""
    M = np.zeros((nv, nv))
    for i in range(nv):
        adr = rowadr[i]
        for k in range(rownnz[i]):
            j = colind[adr + k]
            M[i, j] = data_arr[adr + k]
    return M


def get_dense_M(model, data):
    """Get full dense mass matrix."""
    nv = model.nv
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, M, data.qM)
    return M


def get_dense_qDeriv(model, data):
    """Get qDeriv as dense matrix from sparse D-format."""
    return sparse_to_dense(data.qDeriv, model.nv,
                           model.D_rownnz, model.D_rowadr, model.D_colind)


def get_dense_qLU(model, data):
    """Get qLU (factored) as dense matrix from sparse D-format."""
    return sparse_to_dense(data.qLU, model.nv,
                           model.D_rownnz, model.D_rowadr, model.D_colind)


def compute_A_raw(model, data):
    """Compute A_raw = M - dt*qDeriv (system matrix before factorization)."""
    M = get_dense_M(model, data)
    qDeriv = get_dense_qDeriv(model, data)
    dt = model.opt.timestep
    return M - dt * qDeriv, M, qDeriv


def set_init_pose(model, data, pose, pose_map):
    """Set Go2 to standing pose."""
    mujoco.mj_resetData(model, data)
    data.qpos[2] = INIT_HEIGHT
    data.qpos[3] = 1.0
    data.qpos[4:7] = 0.0
    qpos_addrs, _ = get_joint_addrs(model)
    for i, addr in enumerate(qpos_addrs):
        data.qpos[addr] = pose[pose_map[i]]
    mujoco.mj_forward(model, data)


def compute_pd_ctrl(model, data, q_des, qpos_addrs, dof_addrs, pose_map,
                    is_torque=True):
    """PD controller.
    is_torque=True: ctrl = torque (scenario A)
    is_torque=False: ctrl = I_ss = tau_des / (Kt*gr) (scenario B)
    """
    for i in range(model.nu):
        pi = pose_map[i]
        q = data.qpos[qpos_addrs[i]]
        qd = data.qvel[dof_addrs[i]]
        # Use joint-order PD gains
        tau = KP_JOINT[pi] * (q_des[pi] - q) - KD_JOINT[pi] * qd

        if is_torque:
            data.ctrl[i] = tau
        else:
            Kt_gr = model.actuator_gainprm[i, 0]
            I_des = tau / Kt_gr if Kt_gr > 0 else 0.0
            data.ctrl[i] = I_des


# ── Formatting ────────────────────────────────────────────────────
def fmt_cell(val, threshold=1e-5):
    """Format a single matrix cell matching reference style."""
    if val == 0.0:
        return "\u00b7".rjust(14)
    elif abs(val) < threshold:
        return f"*{val:.4e}".rjust(14)
    else:
        return f"{val:.8f}".rjust(14)


def count_nonzero(M):
    """Count non-zero entries."""
    return int(np.count_nonzero(M))


def fmt_matrix(M, title, row_labels, col_labels, timestep_idx):
    """Format a full matrix with labeled rows/columns, matching reference."""
    nrows, ncols = M.shape
    total = nrows * ncols
    nnz = count_nonzero(M)

    lines = []
    lines.append(f"{title} \u2014 timestep {timestep_idx}  [nonzero: {nnz}/{total}]")

    # Column header
    header = " " * 6  # row label width
    for c in range(ncols):
        header += col_labels[c].rjust(14)
    lines.append(header)

    # Rows
    for r in range(nrows):
        row_str = row_labels[r].rjust(6)
        for c in range(ncols):
            row_str += fmt_cell(M[r, c])
        lines.append(row_str)

    return "\n".join(lines)


def fmt_vector(v, title, labels, timestep_idx):
    """Format a vector with one element per line, matching reference."""
    lines = []
    lines.append(f"{title} \u2014 timestep {timestep_idx}")
    for i, val in enumerate(v):
        lines.append(f"{labels[i]:>10s} = {val:.8f}")
    return "\n".join(lines)


# ── Main trace function ───────────────────────────────────────────
def run_trace(scenario: str, output_path: Path, reorder_actuators: bool = True):
    """Run trace for one scenario.

    Matrix/vector dumps involving an actuator axis are printed with columns/rows
    permuted to match DOF/leg order (FL, FR, RL, RR). The underlying MuJoCo
    model keeps XML declaration order (FR, FL, RR, RL); simulation arrays
    (data.ctrl, data.act, actuator_trnid, …) are never reordered. Set
    reorder_actuators=False to dump in raw XML order.
    """
    is_coupled = (scenario == "B")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    model.opt.timestep = DT_PHYSICS
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

    if is_coupled:
        spec = mujoco.MjSpec.from_file(XML_PATH)
        for act in spec.actuators:
            act.dyntype = mujoco.mjtDyn.mjDYN_FILTEREXACT
            act.dynprm[0] = TAU_E
            act.dynprm[1] = KE_GR
            act.dynprm[2] = L_VAL
            act.dynprm[3] = 1.0  # A+ flag
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            act.gainprm[0] = KT_GR
            act.biastype = mujoco.mjtBias.mjBIAS_NONE
            act.actlimited = True
            act.actrange = [-I_MAX, I_MAX]
            act.ctrllimited = True
            act.ctrlrange = [-I_MAX * 2, I_MAX * 2]
        model = spec.compile()
        model.opt.timestep = DT_PHYSICS
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT

    data = mujoco.MjData(model)
    nv = model.nv
    nu = model.nu
    na = model.na

    qpos_addrs, dof_addrs = get_joint_addrs(model)
    pose_map = build_joint_to_pose_map(model)
    act_labels = build_actuator_labels(model)

    # Display-only permutation. Sort actuators by the DOF index they drive so
    # that B, C, D_elec etc. show a clean block-diagonal structure on the joint
    # DOFs. Simulation data stays in XML declaration order.
    act_dof_idx = np.array([
        model.jnt_dofadr[model.actuator_trnid[i, 0]]
        for i in range(model.nu)
    ])
    act_perm = np.argsort(act_dof_idx) if reorder_actuators else np.arange(model.nu)
    act_labels_disp = [act_labels[i] for i in act_perm]
    act_dof_sorted = act_dof_idx[act_perm]

    # Compute motor parameters
    dt = model.opt.timestep
    D_elec_val = 1.0 + dt * R_VAL / L_VAL  # 1 + dt/tau_e
    schur_scale = dt**2 * KT_GR * KE_GR / (L_VAL * D_elec_val)

    # Build actuator-to-DOF map
    act_dof_map = list(dof_addrs)  # act_dof_map[act_i] = dof index

    # Initialize
    set_init_pose(model, data, POSE_A, pose_map)

    lines = []

    # ── Header ──
    lines.append("=" * 80)
    if is_coupled:
        lines.append("MuJoCo Coupled Implicit Solver \u2014 Matrix Dump")
    else:
        lines.append("MuJoCo Pure PD Torque Implicit Solver \u2014 Matrix Dump")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"nq = {model.nq}  (3 pos + 4 quat + 12 joints = {model.nq})")
    lines.append(f"nv = {nv}  (3 lin_vel + 3 ang_vel + 12 joint_vel = {nv})")
    lines.append(f"nu = {nu}  (12 actuators)")
    lines.append(f"na = {na}  ({na} activation states{' = motor currents' if is_coupled else ''})")
    lines.append(f"dt = {dt}  ({dt*1e3:.1f}ms)")
    lines.append(f"integrator = implicit")
    lines.append("")

    # ── Motor parameters (coupled only) ──
    if is_coupled:
        lines.append("=" * 80)
        lines.append("Motor Parameters (identical for all 12 motors)")
        lines.append("=" * 80)
        lines.append(f"  Kt\u00b7gr     = {KT_GR:.8f}  (gainprm[0])")
        lines.append(f"  Ke\u00b7gr     = {KE_GR:.8f}  (dynprm[1])")
        lines.append(f"  R         = {R_VAL:.8f}  Ohm")
        lines.append(f"  L         = {L_VAL:.10e}  H")
        lines.append(f"  tau_e     = {TAU_E:.10e}  s  (= L/R)")
        lines.append(f"  D_elec    = 1 + dt\u00b7R/L = 1 + dt/tau_e = {D_elec_val:.8f}")
        lines.append(f"  schur_scale = dt\u00b2\u00b7Kt\u00b7gr\u00b7Ke\u00b7gr / (L\u00b7D_elec) = {schur_scale:.10e}")
        lines.append("")
    else:
        lines.append("=" * 80)
        lines.append("PD Control Parameters")
        lines.append("=" * 80)
        lines.append(f"  Kp_hip    = {KP_HIP:.1f}")
        lines.append(f"  Kd_hip    = {KD_HIP:.1f}")
        lines.append(f"  Kp_thigh  = {KP_THIGH:.1f}")
        lines.append(f"  Kd_thigh  = {KD_THIGH:.1f}")
        lines.append(f"  Kp_calf   = {KP_CALF:.1f}")
        lines.append(f"  Kd_calf   = {KD_CALF:.1f}")
        lines.append("")

    # ── DOF labels ──
    lines.append("=" * 80)
    lines.append(f"DOF Labels (nv={nv})")
    lines.append("=" * 80)
    for i in range(nv):
        # Find joint name for this DOF
        jname = ""
        for j in range(model.njnt):
            ndof = {0: 6, 1: 3}.get(model.jnt_type[j], 1)  # free:6, ball:3, else:1
            if model.jnt_dofadr[j] <= i < model.jnt_dofadr[j] + ndof:
                jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                if jname is None and model.jnt_type[j] == 0:
                    jname = "floating_base"
                break
        jname_str = f" \u2190 {jname}" if jname else ""
        lines.append(f"  DOF {i:2d} = {DOF_LABELS[i]:<6s}{jname_str}")
    lines.append("")

    # ── Actuator labels ──
    lines.append("=" * 80)
    lines.append(f"Actuator Labels (nu={nu})")
    lines.append("=" * 80)
    for i in range(nu):
        jid = model.actuator_trnid[i, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        dof_idx = dof_addrs[i]
        lines.append(f"  Act {i:2d} = {act_labels[i]:<8s} \u2190 {jname:<26s} \u2192 DOF {dof_idx:2d} ({DOF_LABELS[dof_idx]})")
    if reorder_actuators:
        lines.append("")
        lines.append("  Display order for actuator axes in matrices/vectors below:")
        lines.append("    " + ", ".join(act_labels_disp))
        lines.append("  (columns permuted to match DOF/leg order FL, FR, RL, RR;")
        lines.append("   underlying model uses XML declaration order FR, FL, RR, RL)")
    lines.append("")
    lines.append("")

    # ── Settling phase ──
    for ps in range(SETTLE_POLICY_STEPS):
        compute_pd_ctrl(model, data, POSE_A, qpos_addrs, dof_addrs, pose_map,
                        is_torque=not is_coupled)
        for _ in range(SUBSTEPS):
            mujoco.mj_step(model, data)

    # ── Dump window ──
    # 35 substeps = 3 pre + 10 (policy A→B) + 10 (policy B→A) + 10 (policy A→B) + 2 extra
    # Transitions at substep 3, 13, 23: three target changes visible
    TOTAL_DUMP = 35
    targets_sequence = [POSE_A, POSE_B, POSE_A, POSE_B]  # alternating
    current_target = POSE_A.copy()
    current_policy_idx = 0  # index into targets_sequence (0 = still on A)

    # Initialize ctrl with current target (POSE_A)
    compute_pd_ctrl(model, data, POSE_A, qpos_addrs, dof_addrs, pose_map,
                    is_torque=not is_coupled)

    # Substep counter within current policy period
    # We start DUMP_BEFORE substeps before the first transition.
    # First transition at substep DUMP_BEFORE (=3).
    # Then every SUBSTEPS (=10) substeps: transitions at 3, 13, 23.
    substep_in_policy = SUBSTEPS - DUMP_BEFORE  # start near end of policy period

    for ss in range(TOTAL_DUMP):
        # Check if this substep is a policy boundary → update target
        target_changed = False
        if substep_in_policy >= SUBSTEPS:
            substep_in_policy = 0
            current_policy_idx += 1
            if current_policy_idx < len(targets_sequence):
                current_target = targets_sequence[current_policy_idx].copy()
                target_changed = True
            compute_pd_ctrl(model, data, current_target, qpos_addrs, dof_addrs,
                            pose_map, is_torque=not is_coupled)
        substep_in_policy += 1

        # Save pre-step state
        qvel_before = data.qvel.copy()
        qpos_before = data.qpos.copy()
        act_before = data.act.copy() if na > 0 else np.array([])
        ctrl_snap = data.ctrl[:nu].copy()

        # Step (this updates qDeriv, qLU, qacc, etc.)
        mujoco.mj_step(model, data)

        # Post-step snapshots
        qvel_after = data.qvel.copy()
        qpos_after = data.qpos.copy()
        act_after = data.act.copy() if na > 0 else np.array([])
        delta_v = qvel_after - qvel_before
        delta_act = (act_after - act_before) if na > 0 else np.array([])

        # Extract matrices (after step, qDeriv/qLU from last Euler sub-step)
        A_raw, M_dense, qDeriv_dense = compute_A_raw(model, data)
        qLU_dense = get_dense_qLU(model, data)

        # ── Write timestep block ──
        lines.append("=" * 80)
        ts_header = f"===== TIMESTEP {ss} ====="
        if target_changed:
            tgt_name = "POSE_B" if np.allclose(current_target, POSE_B) else "POSE_A"
            ts_header += f"  *** TARGET CHANGED TO {tgt_name} ***"
        lines.append(ts_header)
        lines.append("=" * 80)
        lines.append("")

        # ── M (mass matrix) ──
        lines.append(fmt_matrix(M_dense, f"M (mass matrix, {nv}x{nv})",
                                DOF_LABELS, DOF_LABELS, ss))
        lines.append("")

        # ── D_qDeriv ──
        if is_coupled:
            lines.append(fmt_matrix(qDeriv_dense, f"D_qDeriv (with Schur, {nv}x{nv})",
                                    DOF_LABELS, DOF_LABELS, ss))
        else:
            lines.append(fmt_matrix(qDeriv_dense, f"D_qDeriv (damping only, {nv}x{nv})",
                                    DOF_LABELS, DOF_LABELS, ss))
        lines.append("")

        # ── Coupled-only matrices ──
        if is_coupled:
            # Schur_JTJ: diagonal contribution on motor DOFs
            Schur_JTJ = np.zeros((nv, nv))
            for ai in range(nu):
                d = act_dof_map[ai]
                Schur_JTJ[d, d] = schur_scale
            lines.append(fmt_matrix(Schur_JTJ,
                                    f"Schur_JTJ = -B\u00b7D_elec\u207b\u00b9\u00b7C ({nv}x{nv})",
                                    DOF_LABELS, DOF_LABELS, ss))
            lines.append("")

            # D_qDeriv_without_Schur
            D_no_schur = qDeriv_dense - Schur_JTJ
            lines.append(fmt_matrix(D_no_schur,
                                    f"D_qDeriv_without_Schur (D - Schur, {nv}x{nv})",
                                    DOF_LABELS, DOF_LABELS, ss))
            lines.append("")

            # qLU
            lines.append(fmt_matrix(qLU_dense, f"qLU (M - dt\u00b7D, {nv}x{nv})",
                                    DOF_LABELS, DOF_LABELS, ss))
            lines.append("")

            # B = dt * J^T * diag(Kt_gr)  (nv x nu)
            B_mat = np.zeros((nv, nu))
            for ai in range(nu):
                d = act_dof_map[ai]
                B_mat[d, ai] = dt * KT_GR  # J^T is identity on motor DOFs
            B_mat_disp = B_mat[:, act_perm]
            if reorder_actuators:
                # After perm, nonzero should sit at (act_dof_sorted[i], i) only.
                B_check = B_mat_disp.copy()
                for i in range(nu):
                    B_check[act_dof_sorted[i], i] = 0.0
                assert np.allclose(B_check, 0.0), \
                    "Permuted B is not block-diagonal \u2014 act_perm is wrong"
            lines.append(fmt_matrix(B_mat_disp,
                                    f"B = dt \u00b7 J\u1d40 \u00b7 diag(Kt\u00b7gr) ({nv}x{nu})",
                                    DOF_LABELS, act_labels_disp, ss))
            lines.append("")

            # C = -dt * diag(Ke_gr/L) * J  (nu x nv)
            C_mat = np.zeros((nu, nv))
            for ai in range(nu):
                d = act_dof_map[ai]
                C_mat[ai, d] = -dt * KE_GR / L_VAL
            lines.append(fmt_matrix(C_mat[act_perm, :],
                                    f"C = -dt \u00b7 diag(Ke\u00b7gr/L) \u00b7 J ({nu}x{nv})",
                                    act_labels_disp, DOF_LABELS, ss))
            lines.append("")

            # D_elec = I + dt*(R/L)*I  (nu x nu)
            D_elec_mat = np.eye(nu) * D_elec_val
            lines.append(fmt_matrix(D_elec_mat[np.ix_(act_perm, act_perm)],
                                    f"D_elec = I + dt\u00b7(R/L)\u00b7I ({nu}x{nu})",
                                    act_labels_disp, act_labels_disp, ss))
            lines.append("")

            # J (actuator moment arm, nu x nv)
            J_mat = np.zeros((nu, nv))
            for ai in range(nu):
                d = act_dof_map[ai]
                J_mat[ai, d] = 1.0
            lines.append(fmt_matrix(J_mat[act_perm, :],
                                    f"J (actuator moment arm, {nu}x{nv})",
                                    act_labels_disp, DOF_LABELS, ss))
            lines.append("")

            # -B * D_elec^{-1} * C verification
            D_elec_inv = np.eye(nu) / D_elec_val
            BDinvC = -B_mat @ D_elec_inv @ C_mat
            max_diff = np.max(np.abs(BDinvC - Schur_JTJ))
            lines.append(f"-B * D_elec^{{-1}} * C verification: max|(-BDinvC) - Schur_JTJ| = {max_diff:.2e}")
            lines.append("")
            lines.append("  Diagonal comparison (motor DOFs only):")
            lines.append(f"    {'DOF':>5s}  {'Schur_JTJ':>18s}  {'   -BDinvC':>18s}  {'          diff':>14s}")
            for ai in act_perm:
                d = act_dof_map[ai]
                s_val = Schur_JTJ[d, d]
                b_val = BDinvC[d, d]
                diff = b_val - s_val
                lines.append(f"    {DOF_LABELS[d]:>5s}  {s_val:.10e}  {b_val:.10e}  {diff:>14.2e}")
            lines.append("")
        else:
            # Scenario A: just qLU
            lines.append(fmt_matrix(qLU_dense, f"qLU (M - dt\u00b7D, {nv}x{nv})",
                                    DOF_LABELS, DOF_LABELS, ss))
            lines.append("")

        # ── A_raw = M - dt*qDeriv ──
        lines.append(fmt_matrix(A_raw, f"A_raw = M - dt\u00b7qDeriv ({nv}x{nv})",
                                DOF_LABELS, DOF_LABELS, ss))
        lines.append("")

        # ── A_raw vs qLU verification ──
        diag_diff = np.array([abs(A_raw[i, i] - qLU_dense[i, i]) for i in range(nv)])
        max_diag_diff = np.max(diag_diff)
        max_all_diff = np.max(np.abs(A_raw - qLU_dense))
        lines.append(f"A_raw vs qLU verification:")
        lines.append(f"  max|A_raw_diag - qLU_diag| = {max_diag_diff:.2e}")
        lines.append(f"  max|A_raw - qLU| (all elements) = {max_all_diff:.2e}")
        lines.append("")

        # ── Force vectors ──
        lines.append(fmt_vector(data.qfrc_smooth, f"qfrc_smooth ({nv}x1)",
                                DOF_LABELS, ss))
        lines.append("")
        lines.append(fmt_vector(data.qfrc_passive, f"qfrc_passive ({nv}x1)",
                                DOF_LABELS, ss))
        lines.append("")
        lines.append(fmt_vector(data.qfrc_bias, f"qfrc_bias ({nv}x1)",
                                DOF_LABELS, ss))
        lines.append("")
        lines.append(fmt_vector(data.qfrc_actuator, f"qfrc_actuator ({nv}x1)",
                                DOF_LABELS, ss))
        lines.append("")
        lines.append(fmt_vector(data.qfrc_applied, f"qfrc_applied ({nv}x1)",
                                DOF_LABELS, ss))
        lines.append("")
        lines.append(fmt_vector(data.qfrc_constraint, f"qfrc_constraint ({nv}x1)",
                                DOF_LABELS, ss))
        lines.append("")

        # ── State vectors ──
        lines.append(fmt_vector(data.qacc, f"qacc ({nv}x1)", DOF_LABELS, ss))
        lines.append("")

        lines.append(fmt_vector(qvel_before, f"qvel_before ({nv}x1)",
                                DOF_LABELS, ss))
        lines.append("")
        lines.append(fmt_vector(qvel_after, f"qvel_after ({nv}x1)",
                                DOF_LABELS, ss))
        lines.append("")

        lines.append(fmt_vector(qpos_before, f"qpos_before ({model.nq}x1)",
                                QPOS_LABELS, ss))
        lines.append("")
        lines.append(fmt_vector(qpos_after, f"qpos_after ({model.nq}x1)",
                                QPOS_LABELS, ss))
        lines.append("")

        # ── ctrl ──
        lines.append(fmt_vector(ctrl_snap[act_perm], f"ctrl ({nu}x1)",
                                act_labels_disp, ss))
        lines.append("")

        # ── Coupled-only: ctrl_eff, act, force ──
        if is_coupled:
            # ctrl_eff: corrected control = ctrl - (Ke_gr/R) * omega
            # Actually ctrl_eff = (V - Ke*omega)/R in current form
            # For filterexact: ctrl_eff[i] = ctrl[i] - (Ke_gr * omega_i * dt / L) / D_elec
            # Simpler: just show what MuJoCo computed
            # ctrl_eff = ctrl - Ke_gr * omega / R_val ... this depends on internal impl
            # Let's compute: tau_applied = Kt_gr * act_after; ctrl_eff = act_after * R / (Kt_gr * dt)
            # Actually, let's show ctrl_eff as the effective ctrl input corrected for back-EMF
            ctrl_eff = np.zeros(nu)
            for ai in range(nu):
                d = act_dof_map[ai]
                omega = qvel_before[d]
                # Effective ctrl = ctrl - Ke_gr * omega / (R_val) ... no, let's just compute
                # from the ODE: i_dot = (ctrl*R - Ke_gr*omega - R*i) / L
                # ctrl_eff = ctrl - Ke_gr * omega / R_val
                ctrl_eff[ai] = ctrl_snap[ai] - KE_GR * omega / R_VAL
            lines.append(fmt_vector(ctrl_eff[act_perm],
                                    f"ctrl_eff (corrected, {nu}x1)",
                                    act_labels_disp, ss))
            lines.append("")

            lines.append(fmt_vector(act_before[act_perm],
                                    f"act_before (current, {nu}x1)",
                                    act_labels_disp, ss))
            lines.append("")
            lines.append(fmt_vector(act_after[act_perm],
                                    f"act_after (current, {nu}x1)",
                                    act_labels_disp, ss))
            lines.append("")

            # force = Kt*gr * act
            force = KT_GR * act_after
            lines.append(fmt_vector(force[act_perm],
                                    f"force = Kt*gr * act ({nu}x1)",
                                    act_labels_disp, ss))
            lines.append("")

        # ── Delta vectors ──
        lines.append(fmt_vector(delta_v, f"\u0394v = qvel_after - qvel_before ({nv}x1)",
                                DOF_LABELS, ss))
        lines.append(f"  target_changed: {target_changed}")
        lines.append("")

        if is_coupled and na > 0:
            lines.append(fmt_vector(delta_act[act_perm],
                                    f"\u0394act ({nu}x1)",
                                    act_labels_disp, ss))
            lines.append("")

            # act_dot approximation
            act_dot = delta_act / dt
            lines.append(fmt_vector(act_dot[act_perm],
                                    f"act_dot \u2248 \u0394act/dt ({nu}x1)",
                                    act_labels_disp, ss))
            lines.append("")

        lines.append("")

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {output_path} ({len(lines)} lines)")


# ── Main ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running Scenario A (Pure PD Torque)...")
    run_trace("A", OUTPUT_DIR / "trace_scenarioA_puretorque.txt")

    print("\nRunning Scenario B (Coupled Electric + Schur)...")
    run_trace("B", OUTPUT_DIR / "trace_scenarioB_withmotor.txt")

    print(f"\nDone. Files in {OUTPUT_DIR}/")
