/**
 * electric_motor_callback.c
 *
 * MuJoCo mjcb_act_dyn callback: 전기모터 전류 ODE.
 *
 * 수학:
 *   L · dI/dt = V_cmd − R·I − Ke·gr·ω
 *
 * MuJoCo 매핑:
 *   act[i]  = I  (전류)
 *   ctrl[i] = V_cmd (인가 전압)
 *   force   = Kt·gr·I  (gaintype=fixed, gainprm[0]=Kt·gr)
 *
 * 모터 파라미터 (dynprm에 저장):
 *   dynprm[0] = R   (저항, Ω)
 *   dynprm[1] = L   (인덕턴스, H)
 *   dynprm[2] = Ke  (역기전력 상수, V·s/rad)
 *   dynprm[3] = gr  (기어비)
 *
 * 빌드:
 *   gcc -shared -fPIC -O2 \
 *       -I$(python3 -c "import mujoco; print(mujoco.mj_path())")/include \
 *       -o electric_motor_callback.so electric_motor_callback.c
 *
 * Python 사용:
 *   import ctypes, mujoco
 *   lib = ctypes.CDLL("./electric_motor_callback.so")
 *   lib.install_electric_motor_callback()
 *
 * 참고: 이 C callback은 standard mujoco 전용.
 *       mujoco_warp (GPU)에서는 dyntype=filterexact 권장.
 */

#include <mujoco/mujoco.h>

/* ── callback 구현 ──────────────────────────────────────────────── */

static mjtNum electric_motor_act_dyn(
    const mjModel* m, const mjData* d, int id)
{
    /* dyntype=USER 아닌 액추에이터는 무시 */
    if (m->actuator_dyntype[id] != mjDYN_USER) {
        return 0.0;
    }

    /* dynprm에서 모터 파라미터 읽기 */
    const mjtNum* prm = m->actuator_dynprm + id * mjNDYN;
    mjtNum R  = prm[0];
    mjtNum L  = prm[1];
    mjtNum Ke = prm[2];
    mjtNum gr = prm[3];

    /* 전류 I = act[act_adr] */
    int act_adr = m->actuator_actadr[id];
    mjtNum I = d->act[act_adr];

    /* 인가 전압 V_cmd = ctrl[id] */
    mjtNum V_cmd = d->ctrl[id];

    /* 관절 각속도 ω = qvel[dof_adr] */
    int joint_id = m->actuator_trnid[2 * id];       /* transmission target */
    int dof_adr  = m->jnt_dofadr[joint_id];
    mjtNum omega = d->qvel[dof_adr];

    /* 전기 ODE: L·dI/dt = V_cmd − R·I − Ke·gr·ω */
    return (V_cmd - R * I - Ke * gr * omega) / L;
}


/* ── 설치/해제 ──────────────────────────────────────────────────── */

void install_electric_motor_callback(void)
{
    mjcb_act_dyn = electric_motor_act_dyn;
}

void uninstall_electric_motor_callback(void)
{
    mjcb_act_dyn = NULL;
}
