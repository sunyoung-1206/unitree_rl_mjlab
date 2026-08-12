# Rerun of 260716 (go2_methoda_deploydr_gait05/seed42) with Ke=Kt=0.14

**Started:** 2026-07-24

**IMPORTANT — deviates from baseline only in one place:**
`_GO2_MOTOR_PHYS` in `src/assets/robots/unitree_go2/go2_constants.py`
was changed for this run:

- `Kt`: 0.26 → **0.14** [N·m/A]
- `Ke`: 0.26 → **0.14** [V·s/rad_motor]

This is an uncommitted, in-place edit to `go2_constants.py` (not a
separate override path). To restore the nominal datasheet value, set
`Kt`/`Ke` back to `0.26` in that file.

## Everything else is identical to the 260716 run (wandb run
`p113vzbv`, `logs/rsl_rl/go2_methoda_deploydr_gait05/`, started
2026-07-16T04:05:35Z):

```
python scripts/train.py Unitree-Go2-Flat-MethodA-Electric-DeployDR-Gait05-v0 \
  --env.scene.num-envs 4096 \
  --agent.seed 42 \
  --agent.max-iterations 5000 \
  --agent.experiment-name go2_methoda_deploydr_gait05 \
  --agent.run-name seed42
```

Task, num_envs, seed, max_iterations, experiment-name and run-name are
all unchanged from the 260716 run — same code path
(`Unitree-Go2-Flat-MethodA-Electric-DeployDR-Gait05-v0` →
`_go2_flat_methoda_deploydr_gait05_cfg` → `base_cfg_fn=
unitree_go2_flat_methoda_electric_env_cfg`, foot_gait weight 0.50).
Since `--agent.experiment-name`/`--agent.run-name` match the 260716
run, the two are distinguished only by the timestamp prefix in
`logs/rsl_rl/go2_methoda_deploydr_gait05/<timestamp>_seed42/` and by
the wandb run id — check `params/*.yaml` or the wandb-saved
`unitree_rl_mjlab.diff` in each run dir to see the Kt/Ke value actually
used.
