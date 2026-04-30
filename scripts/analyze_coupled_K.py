"""Analyze MuJoCo coupled implicit solver matrix dump.

Parses solver trace file containing A_raw, B, C, D_elec blocks per timestep,
assembles the 30x30 coupled KKT-like matrix K = [[A, B], [C, D]], then:
  1. Visualizes sparsity of K at t=0 (base|joint|motor block structure).
  2. Visualizes signed log-magnitude of K at t=0.
  3. Checks sparsity pattern invariance across all timesteps.
  4. Reports rank / sigma_min / cond / |det| for K, A_raw, D_elec, Schur complement.

Usage:
    python scripts/analyze_coupled_K.py \
        --trace results/solver_trace/trace_scenarioB_withmotor.txt \
        --outdir results/solver_analysis
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


DOF_LABELS = [
    "bx", "by", "bz", "brx", "bry", "brz",
    "FLh", "FLt", "FLc",
    "FRh", "FRt", "FRc",
    "RLh", "RLt", "RLc",
    "RRh", "RRt", "RRc",
]
MOTOR_LABELS = [
    "M_FLh", "M_FLt", "M_FLc",
    "M_FRh", "M_FRt", "M_FRc",
    "M_RLh", "M_RLt", "M_RLc",
    "M_RRh", "M_RRt", "M_RRc",
]
ALL_LABELS = DOF_LABELS + MOTOR_LABELS  # 30

HDR_RE = {
    "A_raw": re.compile(r"^A_raw = M - dt.*?\(18x18\) . timestep (\d+)\s*\[nonzero: (\d+)/(\d+)\]"),
    "B":     re.compile(r"^B = dt .*?\(18x12\) . timestep (\d+)\s*\[nonzero: (\d+)/(\d+)\]"),
    "C":     re.compile(r"^C = -dt .*?\(12x18\) . timestep (\d+)\s*\[nonzero: (\d+)/(\d+)\]"),
    "D_elec": re.compile(r"^D_elec = I.*?\(12x12\) . timestep (\d+)\s*\[nonzero: (\d+)/(\d+)\]"),
}

MATRIX_SHAPES = {
    "A_raw": (18, 18),
    "B":     (18, 12),
    "C":     (12, 18),
    "D_elec": (12, 12),
}


def parse_cell(tok: str) -> tuple[float, bool]:
    """Return (value, is_nonzero) for a single cell token."""
    if tok == "·":
        return 0.0, False
    if tok.startswith("*"):
        # tiny-but-nonzero value
        return float(tok[1:]), True
    return float(tok), True


def parse_matrix_block(lines: list[str], start_idx: int, nrows: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse a matrix of `nrows` rows starting at lines[start_idx] (which is the column-header line)."""
    # skip the column-header line
    data_start = start_idx + 1
    values = []
    mask = []
    for r in range(nrows):
        row = lines[data_start + r].rstrip("\n").split()
        # first token is the row label (e.g. "bx", "M_FLh")
        # the rest are data tokens
        cells = row[1:]
        row_vals = []
        row_mask = []
        for tok in cells:
            v, nz = parse_cell(tok)
            row_vals.append(v)
            row_mask.append(nz)
        values.append(row_vals)
        mask.append(row_mask)
    arr = np.array(values, dtype=float)
    m = np.array(mask, dtype=bool)
    return arr, m


def parse_trace(path: Path) -> dict[int, dict[str, tuple[np.ndarray, np.ndarray, int]]]:
    """Return {timestep: {name: (matrix, mask, reported_nonzero)}}."""
    text = path.read_text()
    lines = text.splitlines()

    result: dict[int, dict[str, tuple[np.ndarray, np.ndarray, int]]] = {}

    for i, line in enumerate(lines):
        for name, rx in HDR_RE.items():
            m = rx.match(line)
            if m is None:
                continue
            t = int(m.group(1))
            nz_reported = int(m.group(2))
            nrows, _ = MATRIX_SHAPES[name]
            arr, mask = parse_matrix_block(lines, i + 1, nrows)
            result.setdefault(t, {})[name] = (arr, mask, nz_reported)
            break
    return result


def assemble_K(block: dict[str, tuple[np.ndarray, np.ndarray, int]]) -> tuple[np.ndarray, np.ndarray]:
    A = block["A_raw"][0]
    B = block["B"][0]
    C = block["C"][0]
    D = block["D_elec"][0]
    K = np.block([[A, B], [C, D]])
    mA = block["A_raw"][1]
    mB = block["B"][1]
    mC = block["C"][1]
    mD = block["D_elec"][1]
    Km = np.block([[mA, mB], [mC, mD]])
    return K, Km


CAPTION = (
    "Block zeros: motor torque does not act on base (top-right); "
    "motor EMF is driven only by joint velocity, not base velocity (bottom-left)."
)

ROW_BLOCK_LABELS = [
    (r"Base  $v$  (6)", 2.5),
    (r"Joint  $\dot q$  (12)", 11.5),
    (r"Motor  $i$  (12)", 23.5),
]


def _apply_block_decor(ax, n: int) -> None:
    """Thin grid, thick separators, and LEFT-side block labels (no top labels)."""
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="#dddddd", linewidth=0.4)
        ax.axvline(k - 0.5, color="#dddddd", linewidth=0.4)
    for k in (6, 18):
        ax.axhline(k - 0.5, color="black", linewidth=2.2)
        ax.axvline(k - 0.5, color="black", linewidth=2.2)

    # Left-side block labels only (rows/cols share order → no info loss)
    for label, y in ROW_BLOCK_LABELS:
        ax.text(
            -4.5, y, label,
            ha="right", va="center",
            fontsize=12, fontweight="bold",
            rotation=0,
            clip_on=False,
        )


def plot_sparsity(mask: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    cmap = mcolors.ListedColormap(["white", "#2a7f7a"])
    ax.imshow(mask.astype(int), cmap=cmap, vmin=0, vmax=1, aspect="equal")

    n = mask.shape[0]
    ax.set_xticks(np.arange(n), ALL_LABELS, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(n), ALL_LABELS, fontsize=12)
    ax.tick_params(axis="both", length=0)

    _apply_block_decor(ax, n)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    fig.suptitle(title, y=0.98, fontsize=13)
    fig.text(0.5, 0.02, CAPTION, ha="center", fontsize=9, style="italic")
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.16)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_signed(K: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))
    absmax = np.max(np.abs(K)) if np.any(K) else 1.0
    linthresh = max(absmax * 1e-6, 1e-12)
    norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=-absmax, vmax=absmax, base=10)
    im = ax.imshow(K, cmap="RdBu_r", norm=norm, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("value (symlog)", fontsize=10)

    n = K.shape[0]
    ax.set_xticks(np.arange(n), ALL_LABELS, rotation=45, ha="right", fontsize=12)
    ax.set_yticks(np.arange(n), ALL_LABELS, fontsize=12)
    ax.tick_params(axis="both", length=0)

    _apply_block_decor(ax, n)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    fig.suptitle(title, y=0.98, fontsize=13)
    fig.text(0.5, 0.02, CAPTION, ha="center", fontsize=9, style="italic")
    fig.subplots_adjust(top=0.92, bottom=0.10, left=0.16)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def rank_report(K: np.ndarray, A: np.ndarray, D: np.ndarray, B: np.ndarray, C: np.ndarray) -> dict:
    # full K
    svK = np.linalg.svd(K, compute_uv=False)
    rankK_def = np.linalg.matrix_rank(K)
    rankK_1em10 = np.linalg.matrix_rank(K, tol=1e-10)
    sigK_max = svK[0]
    sigK_min = svK[-1]
    condK = sigK_max / sigK_min if sigK_min > 0 else np.inf
    sign, logabsdet = np.linalg.slogdet(K)
    abs_det_log = logabsdet  # natural-log |det|

    # A
    svA = np.linalg.svd(A, compute_uv=False)
    rankA = np.linalg.matrix_rank(A)
    sigA_min = svA[-1]

    # D
    rankD = np.linalg.matrix_rank(D)

    # Schur complement S = A - B D^-1 C
    Dinv = np.linalg.inv(D)
    S = A - B @ Dinv @ C
    svS = np.linalg.svd(S, compute_uv=False)
    rankS = np.linalg.matrix_rank(S)
    sigS_min = svS[-1]

    # for the smallest singular vector of K, decompose its support into base/joint/motor
    U, s, Vt = np.linalg.svd(K)
    v_min = Vt[-1]  # right singular vector for smallest sigma
    base_n = float(np.linalg.norm(v_min[0:6]))
    joint_n = float(np.linalg.norm(v_min[6:18]))
    motor_n = float(np.linalg.norm(v_min[18:30]))

    return dict(
        rankK=rankK_def,
        rankK_1em10=rankK_1em10,
        sigK_min=sigK_min,
        sigK_max=sigK_max,
        condK=condK,
        logabsdetK=abs_det_log,
        rankA=rankA,
        sigA_min=sigA_min,
        rankD=rankD,
        rankS=rankS,
        sigS_min=sigS_min,
        vmin_base=base_n,
        vmin_joint=joint_n,
        vmin_motor=motor_n,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"Parsing {args.trace} ...")
    blocks = parse_trace(args.trace)
    timesteps = sorted(blocks.keys())
    print(f"  Found {len(timesteps)} timesteps: {timesteps[0]}..{timesteps[-1]}")

    # Sanity check nonzero counts
    print("\n--- sanity check: reported nonzero vs counted ---")
    mismatches = 0
    for t in timesteps:
        for name, (arr, mask, nz_reported) in blocks[t].items():
            nz_counted = int(mask.sum())
            if nz_counted != nz_reported:
                print(f"  [MISMATCH] t={t} {name}: reported={nz_reported} counted={nz_counted}")
                mismatches += 1
    if mismatches == 0:
        print(f"  OK: all {len(timesteps) * 4} matrices match their reported nonzero counts.")

    # Assemble K for all timesteps
    K_numeric = {}
    K_mask = {}
    for t in timesteps:
        K, Km = assemble_K(blocks[t])
        K_numeric[t] = K
        K_mask[t] = Km

    # --- (2) sparsity plot at t=0
    print("\n--- plotting sparsity ---")
    plot_sparsity(
        K_mask[0],
        args.outdir / "sparsity_t0.png",
        "Coupled K sparsity (30x30) — timestep 0",
    )
    print(f"  wrote {args.outdir / 'sparsity_t0.png'}")

    # --- (3) signed magnitude plot
    plot_signed(
        K_numeric[0],
        args.outdir / "sparsity_signed_t0.png",
        "Coupled K signed log-magnitude — timestep 0",
    )
    print(f"  wrote {args.outdir / 'sparsity_signed_t0.png'}")

    # --- (4) check pattern invariance
    print("\n--- sparsity pattern invariance across timesteps ---")
    ref = K_mask[timesteps[0]]
    differ_list = []
    for t in timesteps[1:]:
        diff = np.where(K_mask[t] != ref)
        if diff[0].size > 0:
            differ_list.append((t, diff))
    if not differ_list:
        print(f"  OK: all {len(timesteps)} K_mask_t identical to K_mask_0.")
    else:
        print(f"  [DIFFER] {len(differ_list)} timesteps differ from t=0 pattern.")
        for t, (rows, cols) in differ_list[:5]:
            for r, c in zip(rows[:10], cols[:10]):
                label_r = ALL_LABELS[r]
                label_c = ALL_LABELS[c]
                print(f"    t={t}: cell ({r},{c}) = ({label_r}, {label_c}) flipped")

    # --- (5) rank / dof report
    print("\n--- rank / dof report ---")
    header = (
        "timestep",
        "rank(K)",
        "rank(K,tol=1e-10)",
        "sigma_min(K)",
        "sigma_max(K)",
        "cond(K)",
        "log|det K|",
        "rank(A)",
        "sigma_min(A)",
        "rank(D)",
        "rank(S)",
        "sigma_min(S)",
        "vmin_base",
        "vmin_joint",
        "vmin_motor",
    )
    rows = []
    print(
        f"{'t':>3} | {'r(K)':>4} | {'r1e-10':>6} | {'sigK_min':>10} | {'condK':>10} | "
        f"{'r(A)':>4} | {'sigA_min':>10} | {'r(D)':>4} | {'r(S)':>4} | {'sigS_min':>10}"
    )
    print("-" * 110)
    min_rank = 30
    for t in timesteps:
        rep = rank_report(
            K_numeric[t],
            blocks[t]["A_raw"][0],
            blocks[t]["D_elec"][0],
            blocks[t]["B"][0],
            blocks[t]["C"][0],
        )
        rows.append(
            [
                t,
                rep["rankK"],
                rep["rankK_1em10"],
                rep["sigK_min"],
                rep["sigK_max"],
                rep["condK"],
                rep["logabsdetK"],
                rep["rankA"],
                rep["sigA_min"],
                rep["rankD"],
                rep["rankS"],
                rep["sigS_min"],
                rep["vmin_base"],
                rep["vmin_joint"],
                rep["vmin_motor"],
            ]
        )
        print(
            f"{t:>3} | {rep['rankK']:>4} | {rep['rankK_1em10']:>6} | "
            f"{rep['sigK_min']:>10.3e} | {rep['condK']:>10.3e} | "
            f"{rep['rankA']:>4} | {rep['sigA_min']:>10.3e} | "
            f"{rep['rankD']:>4} | {rep['rankS']:>4} | {rep['sigS_min']:>10.3e}"
        )
        min_rank = min(min_rank, rep["rankK"])

    csv_path = args.outdir / "rank_report.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"\n  wrote {csv_path}")

    # one-line summary
    print("\n--- summary ---")
    if min_rank == 30:
        print("모든 timestep에서 rank=30 이었음 -> 30 DOF가 모두 독립 (full rank 유지)")
    else:
        # find worst-case timestep
        worst = min(rows, key=lambda r: r[1])
        t = worst[0]
        vb, vj, vm = worst[12], worst[13], worst[14]
        print(
            f"timestep {t}에서 rank={worst[1]}로 떨어짐 -> {30 - worst[1]} DOF 손실, "
            f"smallest right singular vector support: "
            f"base={vb:.3f}, joint={vj:.3f}, motor={vm:.3f}"
        )


if __name__ == "__main__":
    main()
