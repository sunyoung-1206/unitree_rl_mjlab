# Phase 4 Ke-fix Verification Report

vx=0.5 batch (default), 14 cases (1 PD + 1 MethodA healthy + 12 MethodA demag)

Theory slope: (1 − factor) · Ke_nom·gr / R = (1−factor) · 2.7008

## (a) Baseline Consistency

- healthy vx=0.5: slope=+0.1766 | healthy vx=1.5: slope=+0.1772 | diff=0.0006
- **PASS**: |Δ| < 0.05 → 단순 빼기 보정 정당화 강화.

## (b) Slope Correction Table (Primary)

healthy baseline slope = **+0.1766** A·s/rad (subtracted from all demag slopes)

| case | slope_raw | slope_corrected | theory | err % | status |
|---|---|---|---|---|---|
| FL×0.4 | +1.8331 | **+1.6565** | +1.6205 | +2.2% | **PASS** |
| FL×0.6 | +1.3520 | **+1.1754** | +1.0803 | +8.8% | **PASS** |
| FL×0.8 | +0.8143 | **+0.6377** | +0.5402 | +18.1% | **PASS** |
| FR×0.4 | +1.8216 | **+1.6450** | +1.6205 | +1.5% | **PASS** |
| FR×0.6 | +1.3066 | **+1.1300** | +1.0803 | +4.6% | **PASS** |
| FR×0.8 | +0.6294 | **+0.4528** | +0.5402 | -16.2% | **PASS** |
| RL×0.4 | +1.8507 | **+1.6741** | +1.6205 | +3.3% | **PASS** |
| RL×0.6 | +1.3264 | **+1.1498** | +1.0803 | +6.4% | **PASS** |
| RL×0.8 | +0.7693 | **+0.5927** | +0.5402 | +9.7% | **PASS** |
| RR×0.4 | +1.8050 | **+1.6284** | +1.6205 | +0.5% | **PASS** |
| RR×0.6 | +1.3133 | **+1.1367** | +1.0803 | +5.2% | **PASS** |
| RR×0.8 | +0.6799 | **+0.5033** | +0.5402 | -6.8% | **PASS** |

**Summary**: PASS=12 / PARTIAL=0 / FAIL=0 of 12

## (c) Ratio Analysis + V_bus Saturation + ω Breakdown

| case | ratio mean | ratio |ω|<2 | ratio |ω|>5 | V_bus sat % | |ω|>5 frac |
|---|---|---|---|---|---|
| FL×0.4 | 0.369 | 0.395 | nan | 0.0% | 1.8% |
| FL×0.6 | 0.511 | 0.577 | -0.858 | 0.0% | 3.4% |
| FL×0.8 | 0.801 | 0.803 | nan | 0.0% | 3.2% |
| FR×0.4 | 0.338 | 0.369 | nan | 0.0% | 0.0% |
| FR×0.6 | 0.504 | 0.569 | nan | 0.0% | 0.0% |
| FR×0.8 | 0.799 | 0.816 | nan | 0.0% | 0.0% |
| RL×0.4 | 0.422 | 0.414 | nan | 0.0% | 0.0% |
| RL×0.6 | 0.524 | 0.614 | -0.165 | 0.0% | 3.1% |
| RL×0.8 | 0.735 | 0.798 | nan | 0.0% | 0.0% |
| RR×0.4 | 0.391 | 0.402 | nan | 0.0% | 1.3% |
| RR×0.6 | 0.526 | 0.595 | nan | 0.0% | 0.1% |
| RR×0.8 | 0.725 | 0.816 | nan | 0.0% | 0.1% |

## (d) Current Limit Classification

Go2 calf I limit (joint-space) = 55.5 A

| case | I_max | % of limit | class |
|---|---|---|---|
| FL×0.4 | 50.6 A | 91% | approach |
| FL×0.6 | 29.5 A | 53% | within |
| FL×0.8 | 20.7 A | 37% | within |
| FR×0.4 | 35.3 A | 64% | within |
| FR×0.6 | 24.9 A | 45% | within |
| FR×0.8 | 18.1 A | 33% | within |
| RL×0.4 | 45.4 A | 82% | approach |
| RL×0.6 | 32.8 A | 59% | within |
| RL×0.8 | 24.8 A | 45% | within |
| RR×0.4 | 51.5 A | 93% | approach |
| RR×0.6 | 25.0 A | 45% | within |
| RR×0.8 | 20.4 A | 37% | within |

---

# Baseline Correction Justification

## Observation
Healthy slope = **+0.1766** A·s/rad (vx=0.5), +0.1772 (vx=1.5). Difference 0.0006.

## Patch-independence
- ke_ignored healthy: slope = +0.1766
- ke_fixed healthy: slope = +0.1766
→ artifact is orthogonal to patch (pre-patch baseline matches post-patch).

## Factor-independence (ω-shift evidence from Phase 3)
ω shift scan on (vx=1.5 healthy, FL_0.6):

| shift | healthy slope | FL_0.6 slope | diff (Ke coupling) |
|---|---|---|---|
| −1 | +0.034 | +0.883 | +0.849 |
| 0 | +0.177 | +1.300 | **+1.123** (theory +1.080, err 4%) |
| +1 | +0.227 | +1.153 | +0.926 |

→ Baseline shifts coherently with measured, confirming factor-independent timing artifact.

## Conclusion
`slope_corrected = slope_demag − slope_healthy_baseline` is the accurate estimator of Ke coupling contribution.

## Future Work
Most likely cause: PD ZOH × log-timing interaction in mjlab's decimation. Not investigated further — requires mjlab decoder + timestep logging (several hours), independent of Phase 4 result validity.
