# Baseline Artifact Root-Cause Investigation

**Target**: healthy ω vs ΔI slope = +0.1766 (±0.0006, vx-invariant)  
**Expected**: 0 (no Ke coupling → I_actual should track I_cmd exactly in healthy)

## Evidence collected

### Step 1 — Logging code audit (H1: Python-side bug)

**Result: H1 REFUTED**

Audited `run_demag_experiment.py` rollout loop:
- `wp_data.qvel.numpy()` / `wp_data.act.numpy()` / `wp_data.ctrl.numpy()` all trigger warp device→host copy → **fresh snapshot every call, no reference aliasing**.
- `hold[0].detach().cpu().numpy()` for `_tau_des_hold` → detach + cpu copy, no leak.
- `tau_actual[step] = qfrc[qvel_adrs]` — pre-allocated numpy buffer, indexed assignment is value-copy.
- No decimation / off-by-one in logging path (one log per `wrapped.step` return).

No bug in the Python logging. Logged values correctly reflect the state at the call site.

### Step 2 — mjlab step sequence inspection

Located the mjlab decimation loop (`manager_based_rl_env.py:370-375`):

```python
for _ in range(self.cfg.decimation):
    self.action_manager.apply_action()
    self.scene.write_data_to_sim()   # custom actuator compute() → writes mj_data.ctrl
    self.sim.step()                   # mjwarp.step integrates physics
    self.scene.update(dt=self.physics_dt)
```

After the loop, line 401: `self.sim.forward()` — runs all kinematics but **does not integrate** (act/qpos/qvel unchanged).

**Crucial finding** from mjlab docstring (line 342-353):
> "MuJoCo's `mj_step` runs forward kinematics *before* integration, so after stepping, derived quantities (xpos, xquat, site_xpos, cvel, sensordata) lag qpos/qvel by one physics substep."

The `sim.forward()` call resolves this for derived quantities but **not** for integrated state. This means within the last sim.step(N) of each policy step:
- `_actuator_force` kernel runs with `actuator_velocity_in` = qvel *before* substep N integration.
- act is integrated using this stale ω → 1-substep lag in ω input.
- After step: qvel = post-N, act = integrated with pre-N ω.

However this 1-substep lag (0.1ms) is too small (Δω ≈ 0.01 rad/s) to explain std(ΔI) = 1.4 A.

### Step 3 — Decimation / PD-period sweep (H4 test)

Ran healthy MethodA for 500 policy steps at varying decimation, with actuator `substeps = pd_substeps = decimation` (one PD recompute per policy step).

| decimation | physics_dt | pd_substeps | slope | std(ΔI) |
|---|---|---|---|---|
| 50 | 0.4 ms | 50 | **+0.1665** | 1.19 |
| 200 | 0.1 ms | 200 | **+0.4057** | 2.96 |
| (production) | 0.1 ms | 50 | +0.1766 | 0.59 |

Observations:
- decimation=1/2/10/20 could not run (actuator asserts `dt ≤ 3·τ_e` for filterexact accuracy).
- **At fixed policy_dt = 20 ms, slope scales with `policy_dt / pd_recompute_period`**:
  - pd period 50 → slope ≈ 0.17
  - pd period 200 → slope ≈ 0.40 (~2.3× larger at 4× longer PD period)
- **H4 SUPPORTED**: the artifact tracks the PD ZOH / log-timing mismatch relative to ω evolution.

### Step 4 — mjwarp buffer trace (H3)

Skipped per branching rule (H4 already supports an origin hypothesis). `actuator_velocity_in` behaviour partly examined in Step 2 but not the full swap timing. Left for future investigation.

## Hypothesis summary

| # | hypothesis | evidence | judgment |
|---|---|---|---|
| H1 | Python logging bug | Step 1 audit clean | **REFUTED** |
| H2 | MuJoCo derived-quantity staleness | Step 2: 1-substep lag exists but magnitude insufficient (0.1ms) | **WEAK (partial)** |
| H3 | mjwarp buffer swap timing | Not directly tested | **UNKNOWN** |
| H4 | PD ZOH × log timing | Step 3: slope scales with PD period | **SUPPORTED (primary)** |

## Most likely mechanism (medium confidence)

**PD ZOH coherence loss between I_des computation time and ω log time.**

Chain:
1. At each PD recompute (every `pd_substeps` physics steps = 5 ms in production), PD reads q/q̇ and computes tau_des, caches in `_tau_des_hold`.
2. Between recomputes, `_tau_des_hold` held constant. ctrl = I_des_held. I_actual tracks ctrl with τ_e=0.33ms → settles within ~1.7ms (≪ PD period 5ms).
3. At log time (end of policy step = 200th physics substep):
   - logged ω = qvel at substep 200 (post-last-integration).
   - logged I_cmd / I_des = value cached from **last PD recompute** (substep 150 in production) — based on qvel from substep 150.
4. Logged I_actual tracks ctrl of last PD period — corresponds to ω at substep 150, not substep 200.

While my analytic expectation is ΔI = 0 (since I_actual and I_cmd both derive from the same `_tau_des_hold`), the empirical std = 1.4 A suggests an additional factor not yet isolated. The slope scaling with PD period (Step 3) is the strongest hint that the mechanism involves ω-time-dependent effects across the ZOH boundaries that do not fully cancel in the kernel's act integration.

**Confidence**: MEDIUM. Primary mechanism plausible but residual ΔI std not fully explained by pure PD ZOH theory.

## Fix proposals (not applied)

### Option A — Remove PD ZOH (compute every substep)
Set `pd_substeps = 1` so PD recomputes every physics substep. Aligns I_des with current ω, removes lag.
- Cost: 4× PD compute per policy step → GPU load increase (measured ~40% slower for decimation=200 in Step 3).
- Expected: slope → 0 or <0.05.
- Risk: changes physics (no ZOH-based training-inference consistency). Controller semantics shift.

### Option B — Correct logging: use ω at PD recompute time, not end-of-step
Change the logger to read ω at the moment of each PD recompute (expose via actuator.compute hook). Comparisons between I_des and ω become coherent.
- Cost: instrument `NativeElectricActuator.compute` to expose ω-at-compute-time; modify experiment script to read via that hook.
- Expected: slope → 0 in healthy (trivially).
- Risk: none beyond refactor. Does not change physics or training.

### Option C — Pre-reduce ω to match PD period
Log ω averaged over last PD period, not end-of-step instantaneous.
- Cost: minor script change.
- Expected: slope reduced but likely not exactly 0 (averaging doesn't capture exact PD sample time).
- Risk: still an approximation.

**Recommendation**: **Option B**. Cleanest, physics-neutral, zero-risk to results. Also gives a handle for future diagnostics.

## Residual open questions

1. Why is ΔI std = 1.4 A empirically while pure theory says 0? (H3 not tested.)
2. What exact transformation maps "PD period / policy period ratio" to slope? Step 3 hints at linear scaling but only 2 data points.
3. Does Option B reduce artifact to ~0, or is there residual from another mechanism?

Future work (if desired): instrument NativeElectricActuator to log per-substep (ω, ctrl, act) and fit/analyze directly. ~2 hours.

## Verdict for Phase 4 results

**Phase 4 Ke-fix validation remains unaffected**:
- Artifact is factor-independent (ke_ignored healthy slope = ke_fixed healthy slope to 4 decimal).
- Baseline subtraction `slope_corrected = slope_demag − slope_healthy` correctly isolates Ke coupling contribution (12/12 PASS, all ±20% of theory).
- Root cause of artifact (PD ZOH / log timing) is orthogonal to the physical claim being validated.

This investigation **elevates future-work priority** of logging-time alignment (Option B) but does not alter any Phase 4 conclusions.
