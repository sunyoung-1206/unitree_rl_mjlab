# WORKLOG

## [2026-05-02 18:05] dataflow 노트북 7/8절을 mujoco fork 위치 정보로 업데이트

### 실행 환경
| 항목 | 값 |
|------|-----|
| 스크립트 | `scripts/update_dataflow_notebook_section8.py` |
| 외부 참조 | `https://github.com/sunyoung-1206/mujoco` (사용자 fork, `/tmp/mujoco_fork/` 에 shallow clone) |
| 대상 파일 | `unitree_go2_flat_dataflow.ipynb` 의 셀 33 (7절), 셀 34 (8절) |

### 변경 사항
| 파일 | 유형 | 설명 |
|------|------|------|
| `scripts/update_dataflow_notebook_section8.py` | 생성 | ipynb 의 셀 33/34 만 in-place 교체 (다른 셀 보존) |
| `unitree_go2_flat_dataflow.ipynb` | 수정 | 7절에 mujoco engine 위치 행 8 개 추가, 8절을 §8.1 (확인됨) / §8.2 (여전히 미확인) / §8.3 (단순화 표시) 로 재구성 |

### 작업 상세
- 1차 작성 시 8절 "미확인" 으로 둔 6 항목 중 mujoco engine 본체에 있는 5 항목의 위치를 mujoco fork 에서 확인:
  - 비구속 가속도 어셈블리: `engine_forward.c:788` (`mj_fwdAcceleration`)
  - 운동방정식 $M$ 어셈블리: `engine_core_smooth.c:1745` (`mj_crb`)
  - $M$ factor / solve: `engine_core_smooth.c:1894 / 2152 / 2064` (`mj_factorM`, `mj_solveM`, `mj_solveLD`)
  - $C(q,\dot q) + g(q)$ (`qfrc_bias`): `engine_core_smooth.c:2359` (`mj_rne`)
  - $J_c$ 어셈블리: `engine_core_constraint.c:2495` (`mj_makeConstraint`)
  - 접촉력 $\lambda$ Schur 풀이: `engine_forward.c:953` (`mj_fwdConstraint`) → `engine_solver.c` (CG/Newton iteration)
  - semi-implicit Euler: `engine_forward.c:1047, 1126, 1132` (`mj_advance` 안의 속도→위치 갱신 순서)
- 노트북 4.3 의 `q_dot_new = q_dot + dt*q_ddot` → `q_new = q + dt*q_dot_new` 가 mujoco 의 `mj_advance` (속도 먼저, 그 다음 위치) 와 정확히 같은 semi-implicit / symplectic Euler 임을 8절에 명시.
- 여전히 미확인: `DcMotorActuator.compute()` PD 본문과 saturation 함수 형태. 이는 `mjlab.actuator.dc_actuator` 에 있고 mujoco fork 에는 없음 (mjlab 은 별도 repo).
- 검증: 셀 35 개 유지, 모든 code 셀 `compile()` 통과.

### 로그
- `logs/2026-05-02_18-05_update_dataflow_section8.log`

### 관련 작업
- 본 작업의 입력은 1차 노트북 ([2026-05-02 17:35] 항목 참조).

---

## [2026-05-02 17:35] Unitree-Go2-Flat-MethodA-Electric 계산 흐름 노트북 생성

### 실행 환경 (스크립트 실행 시)
| 항목 | 값 |
|------|-----|
| 스크립트 | `scripts/build_dataflow_notebook.py` |
| Python | 3 (system) |
| 외부 의존성 | numpy, pandas (셀 실행 시) |
| 산출물 | `unitree_go2_flat_dataflow.ipynb` (셀 35개) |

### 변경 사항
| 파일 | 유형 | 설명 |
|------|------|------|
| `scripts/build_dataflow_notebook.py` | 생성 | ipynb 생성기 (markdown/code 셀을 Python 함수로 조립) |
| `unitree_go2_flat_dataflow.ipynb` | 생성 | 정책→PD→물리 3박자 계산 흐름 노트북 |
| `WORKLOG.md` | 생성 | 작업 기록 시작 |

### 작업 상세
- **목적:** 기계공학 연구실 구성원이 Method A actuator 의 한 정책 주기 흐름을 손으로 따라가도록, (A) 소스 발췌 → (B) 1∼2 자유도 축소 예제 값 대입 → (C) 중간 결과 의 3박자 셀 묶음을 구성.
- **사전 확인 결과 (코드 직접 확인):**
  - 정책 주기 = `cfg.sim.mujoco.timestep × cfg.decimation` = 0.0001 × 200 = **20 ms** (`src/tasks/velocity/config/go2/env_cfgs.py:206-207`)
  - PD 재계산 주기 = `_PD_RECOMPUTE × physics_dt` = 50 × 0.1 ms = **5 ms** (`src/assets/robots/unitree_go2/go2_constants.py:305-306`)
  - 물리 적분 주기 = **0.1 ms**
  - "MethodA" = BE 일관 (적분기/Schur/Force RHS 모두 β = 1/(1+h/τ)). `dynprm[4]=0`. (`mj_native_electric_actuator.py:84, 333`)
  - "Electric" = 모터 전류 I 를 MuJoCo `d->act` 에 통합. `dyntype=filterexact` + Schur cross-Jacobian.
- **β_imp / β_int 위치 매핑:**
  - β_int (적분기): `vendor/mujoco_warp_3.6.0_patch/_src/forward.py:147-170` (kernel `_next_act`)
  - β_imp Schur 좌변: `vendor/mujoco_warp_3.6.0_patch/_src/derivative.py:68-89` (kernel `_qderiv_actuator_passive_vel`)
  - β_imp Force RHS 우변: `vendor/mujoco_warp_3.6.0_patch/_src/forward.py:752-770` (kernel `_actuator_force`)
- **검증:** 35 셀 전부 `compile()` 통과, 13 개 코드 셀을 namespace 공유 상태로 순차 실행하여 끝까지 정상 동작 확인. 수치 정합도 OK (β_imp = 0.769 = 1/(1+0.1/0.333), I → I_des = 5.0 A 수렴).

### 미확인 항목 (노트북 8절에도 동일 기재)
- `DcMotorActuator.compute()` 본문 PD 식 — `mjlab.actuator.dc_actuator` 외부 패키지. 학습은 원격 conda env 에 설치된 mjlab 사용, 로컬엔 mjlab 미설치. 노트북에는 cfg 값과 wrapper docstring 으로부터 재구성한 표준형 `τ_pd = Kp·(q_des−q) − Kd·q̇`, `τ_max(ω) = sat·max(0, 1 − |ω|/v_max)` 사용.
- 기계측 적분기 (semi-implicit Euler) 본문 — `mujoco_warp/_src/euler.py` (이 레포 vendor 디렉토리에는 patched `forward.py`, `derivative.py` 만 존재).
- 비구속 가속도 q̈_free 어셈블리 본문 — mujoco_warp 본체 (`smooth.py`, `solver.py` 등).
- 운동방정식 M, C, g, J_c 어셈블리 본문 — mujoco_warp 본체.
- 접촉력 λ Schur 풀이 본문 (`solver_init_efc`, `solver_step`) — mujoco_warp 본체.

### 로그
- `logs/2026-05-02_17-34_build_dataflow_notebook.log` (1차 — `"""` 충돌 SyntaxError)
- `logs/2026-05-02_17-35_build_dataflow_notebook.log` (2차 — 성공)
