"""Phase 1 sanity 결과 비교 — 3 modes (no_heading / auto / explicit_0).

각 모드별 healthy.npz 를 읽어:
  - 시작 yaw (deg) — seed 통제 확인
  - 최종 y 변위 |y_final - y_start| [m]
  - target_yaw 대비 max |yaw - target| [deg]
  - cmd_wz 분포 (heading-lock 가 매 step wz 를 다르게 쓰는지)
요약 출력.
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path("results/heading_lock_compare")
MODES = [
    ("no_heading", ROOT / "sanity_no_heading" / "data" / "aplus_tloop_ki000"
                       / "healthy.npz"),
    ("auto",       ROOT / "sanity_auto"       / "data" / "aplus_tloop_ki000"
                       / "healthy.npz"),
    ("explicit_0", ROOT / "sanity_explicit_0" / "data" / "aplus_tloop_ki000"
                       / "healthy.npz"),
]


def unwrap(y):
    """rad 시계열 unwrap (≈ atan2 결과의 ±π 점프 제거)."""
    return np.unwrap(y)


def summarize(tag, path):
    d = np.load(path, allow_pickle=True)
    m = json.loads(str(d["meta"]))
    rpy = d["base_rpy"]                            # (N, 3) [r, p, y]
    pos = d["base_pos"]                            # (N, 3)
    cmd = d["cmd_vel"]                             # (N, 3) [vx, vy, wz]
    yaw = unwrap(rpy[:, 2])
    yaw_start = float(yaw[0])
    target = (yaw_start if m.get("heading_mode") == "auto"
              else (m.get("target_heading") if m.get("heading_mode") == "explicit"
                    else yaw_start))   # no_heading → 비교용으로 start yaw 기준
    err = yaw - target
    y_drift = float(pos[-1, 1] - pos[0, 1])
    x_travel = float(pos[-1, 0] - pos[0, 0])
    return dict(
        tag=tag,
        path=str(path),
        n=int(m["num_steps"]),
        dt=float(m["dt"]),
        heading_mode=m.get("heading_mode", "?"),
        target_meta=m.get("target_heading"),
        yaw_start_deg=np.degrees(yaw_start),
        target_deg=np.degrees(target),
        x_travel=x_travel,
        y_drift=y_drift,
        max_yaw_err_deg=float(np.degrees(np.max(np.abs(err)))),
        rms_yaw_err_deg=float(np.degrees(np.sqrt(np.mean(err**2)))),
        cmd_wz_min=float(cmd[:, 2].min()),
        cmd_wz_max=float(cmd[:, 2].max()),
        cmd_wz_std=float(cmd[:, 2].std()),
    )


def main():
    rows = []
    for tag, p in MODES:
        if not p.exists():
            print(f"[MISS] {tag}: {p}")
            continue
        rows.append(summarize(tag, p))

    if not rows:
        print("No data."); return

    cols = ("tag", "heading_mode", "yaw_start_deg", "target_deg",
            "x_travel", "y_drift",
            "max_yaw_err_deg", "rms_yaw_err_deg",
            "cmd_wz_min", "cmd_wz_max", "cmd_wz_std")
    header = ("tag", "mode", "yaw0[deg]", "tgt[deg]",
              "x[m]", "y[m]",
              "max|yawErr|", "rms|yawErr|",
              "wz_min", "wz_max", "wz_std")
    widths = (12, 10, 10, 10, 8, 8, 12, 12, 8, 8, 8)

    print("=" * sum(widths) + "=" * (len(widths) - 1))
    print(" ".join(f"{h:>{w}}" for h, w in zip(header, widths)))
    print("-" * sum(widths) + "-" * (len(widths) - 1))
    for r in rows:
        vals = (
            r["tag"], r["heading_mode"],
            f"{r['yaw_start_deg']:+.3f}", f"{r['target_deg']:+.3f}",
            f"{r['x_travel']:+.3f}", f"{r['y_drift']:+.3f}",
            f"{r['max_yaw_err_deg']:.3f}", f"{r['rms_yaw_err_deg']:.3f}",
            f"{r['cmd_wz_min']:+.3f}", f"{r['cmd_wz_max']:+.3f}",
            f"{r['cmd_wz_std']:.3f}",
        )
        print(" ".join(f"{v:>{w}}" for v, w in zip(vals, widths)))

    # seed control: 세 모드의 yaw_start 가 동일해야 한다.
    yaw0s = {r["yaw_start_deg"] for r in rows}
    if len(yaw0s) == 1:
        print(f"\n[OK] seed 통제: 모든 모드의 시작 yaw 동일 ({list(yaw0s)[0]:+.4f} deg)")
    else:
        print(f"\n[WARN] 시작 yaw 가 모드별로 다름: {sorted(yaw0s)}")


if __name__ == "__main__":
    main()
