# Kt/Ke constant bug (fixed 2026-07-23)

`run_demag_experiment.py` (copied from `results/heading_lock_compare/`) hardcoded
`KT_NOMINAL_JOINT = KE_NOMINAL_JOINT = 0.128 * 6.33 = 0.8102`.

The actual robot config (`src/assets/robots/unitree_go2/go2_constants.py`
`_GO2_MOTOR_PHYS`, used by `GO2_METHODA_CALF` which `gait05`'s base actuator
config is built on) is `Kt = Ke = 0.26 N*m/A`, `gear_ratio = 6.33` ->
true nominal `Kt*gr = Ke*gr = 1.6458`. Verified live via
`env.scene["robot"].actuators[calf]._Ktgr == 1.6458`.

Impact on the runs archived in this folder (`_archive_wrong_kt0.128/`):
- `inject_demagnetization()` sets `gainprm[idx,0] = KT_NOMINAL_JOINT * factor`
  as an **absolute** value, not a relative scale of the true nominal. So
  factor=0.8/0.6/0.4 (intended 20/40/60% severity) actually produced
  `0.8102*factor / 1.6458` = **60.6% / 70.5% / 80.3%** severity relative to
  the true nominal -- much more severe than labeled. This is why RR_0.60 and
  RR_0.40 (actually ~70-80% severity) couldn't make forward progress.
- All derived `tau_cmd` / torque-estimate values in the old plots understate
  the true torque by 0.8102/1.6458 = 0.492x (~half).
- `I_MAX_CALF` (rated-current bound) was computed as `45/0.8102 = 55.5A`,
  should have been `45/1.6458 = 27.34A` -- the earlier claim of
  "current/effort-limit saturation never reached" was wrong; `max|I_cmd|`
  was actually sitting right at the true 27.34A rated-current bound.

Fixed in the live `run_demag_experiment.py`; the corrected 20/40/60% re-run
lives in `../data`, `../videos`, `../view_data.ipynb`.
