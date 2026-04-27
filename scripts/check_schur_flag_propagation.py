"""Verify that use_filterexact_schur=True actually reaches mj_model.actuator_dynprm."""
from __future__ import annotations

import numpy as np
import torch

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

TASK = "Unitree-Go2-Flat-MethodA-Electric"


def show(label: str, schur: bool) -> None:
    env_cfg = load_env_cfg(TASK, play=True)
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = int(1e9)
    env_cfg.seed = 42

    changed = 0
    for a in env_cfg.scene.entities["robot"].articulation.actuators:
        if hasattr(a, "use_filterexact_schur"):
            a.use_filterexact_schur = schur
            changed += 1
    print(f"[{label}] toggled {changed} actuator cfgs; schur={schur}")

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    env = ManagerBasedRlEnv(cfg=env_cfg, device=dev, render_mode=None)

    mj = env.sim.mj_model
    wp = env.sim.model.struct
    print(f"  mj_model.nu = {mj.nu}")

    # print dynprm / gainprm for the 12 actuators
    print("  mj_model.actuator_dynprm (all 12):")
    for i in range(mj.nu):
        jid = int(mj.actuator_trnid[i, 0])
        name = mj.joint(jid).name.replace("robot/", "")
        print(f"    [{i:2d}] {name:<20} dyntype={mj.actuator_dyntype[i]:2d} "
              f"dynprm={np.array2string(mj.actuator_dynprm[i, :4], precision=4, suppress_small=True):<40} "
              f"gainprm[0]={mj.actuator_gainprm[i, 0]:.4f}")

    # print warp data
    dyn_wp = wp.actuator_dynprm.numpy()
    print(f"\n  wp.actuator_dynprm[0, :, :4] (warp):")
    print(f"    {dyn_wp[0, :, :4]}")

    env.close()
    print()


if __name__ == "__main__":
    show("Schur=False (MethodA default)", schur=False)
    show("Schur=True  (flipped)",         schur=True)
