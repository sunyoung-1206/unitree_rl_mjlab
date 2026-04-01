# Unitree RL Mjlab

> **This repository is a fork of [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab).**
> See [Fork Modifications](#-fork-modifications) for a summary of changes added in this fork.

---

## ✳️ Overview
Unitree RL Mjlab is a reinforcement learning project built upon the
[mjlab](https://github.com/mujocolab/mjlab.git), using MuJoCo as its 
physics simulation backend, currently supporting Unitree Go2, A2, G1, H1_2 and R1.

Mjlab combines [Isaac Lab](https://github.com/isaac-sim/IsaacLab)'s proven API
with best-in-class [MuJoCo](https://github.com/google-deepmind/mujoco_warp)
physics to provide lightweight, modular abstractions for RL robotics research
and sim-to-real deployment.

<div align="center">

| <div align="center">  MuJoCo </div>                                                                                                                                           | <div align="center"> Physical </div>                                                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> | <div style="width:250px; height:150px; overflow:hidden;"><img src="doc/gif/g1-velocity-real.gif" style="width:100%; height:100%; object-fit:cover; object-position:center;"></div> |

</div>


## 📦 Installation and Configuration

Please refer to [setup.md](doc/setup_en.md) for installation and configuration steps.


## 🔁 Process Overview

The basic workflow for using reinforcement learning to achieve motion control is:

`Train` → `Play` → `Sim2Real`

- **Train**: The agent interacts with the MuJoCo simulation and optimizes policies through reward maximization.
- **Play**: Replay trained policies to verify expected behavior.
- **Sim2Real**: Deploy trained policies to physical Unitree robots for real-world execution.


## 🛠️ Usage Guide

### 1. Velocity Tracking Training

Run the following command to train a velocity tracking policy:

```bash
python scripts/train.py Unitree-G1-Flat --env.scene.num-envs=4096
```

Multi-GPU Training: Scale to multiple GPUs using --gpu-ids:

```bash
python scripts/train.py Unitree-G1-Flat \
  --gpu-ids 0 1 \
  --env.scene.num-envs=4096
```

- The first argument (e.g., Mjlab-Velocity-Flat-Unitree-G1) specifies the training task.
Available velocity tracking tasks:
  - Unitree-Go2-Flat
  - Unitree-Go2-Flat-Electric *(added in this fork)*
  - Unitree-Go2-Flat-Native-Electric *(added in this fork)*
  - Unitree-G1-Flat
  - Unitree-G1-23Dof-Flat
  - Unitree-H1_2-Flat
  - Unitree-A2-Flat
  - Unitree-R1-Flat

> [!NOTE]
> For more details, refer to the mjlab documentation:
> [mjlab documentation](https://mujocolab.github.io/mjlab/index.html).

### 2. Motion Imitation Training

Train a Unitree G1 to mimic reference motion sequences.

<div style="margin-left: 20px;">

#### 2.1 Prepare Motion Files

Prepare csv motion files in mjlab/motions/g1/ and convert them to npz format:

```bash
python scripts/csv_to_npz.py \
--input-file src/assets/motions/g1/dance1_subject2.csv \
--output-name dance1_subject2.npz \
--input-fps 30 \
--output-fps 50
```

**npz files will be stored at:**：`src/motions/g1/...`

#### 2.2 Training

After generating the NPZ file, launch imitation training:

```bash
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation --motion_file=src/assets/motions/g1/dance1_subject2.npz --env.scene.num-envs=4096
```

</div>

> [!NOTE]
> For detailed motion imitation instructions, refer to the BeyondMimic documentation:
> [BeyondMimic documentation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/README.md#motion-preprocessing--registry-setup).

#### ⚙️  Parameter Description
- `--env.scene`: simulation scene configuration (e.g., num_envs, dt, ground type, gravity, disturbances)
- `--env.observations`: observation space configuration (e.g., joint state, IMU, commands, etc.)
- `--env.rewards`: reward terms used for policy optimization
- `--env.commands`: task commands (e.g., velocity, pose, or motion targets)
- `--env.terminations`: termination conditions for each episode
- `--agent.seed`: random seed for reproducibility
- `--agent.resume`: resume from the last saved checkpoint when enabled
- `--agent.policy`: policy network architecture configuration
- `--agent.algorithm`: reinforcement learning algorithm configuration (PPO, hyperparameters, etc.)

**Training results are stored at**：`logs/rsl_rl/<robot>_(velocity | tracking)/<date_time>/model_<iteration>.pt`

### 3. Simulation Validation

To visualize policy behavior in MuJoCo:

Velocity tracking:
```bash
python scripts/play.py Unitree-G1-Flat --checkpoint_file=logs/rsl_rl/g1_velocity/2026-xx-xx_xx-xx-xx/model_xx.pt
```

Motion imitation:
```bash
python scripts/play.py Unitree-G1-Tracking --motion_file=src/assets/motions/g1/dance1_subject2.npz --checkpoint_file=logs/rsl_rl/g1_tracking/2026-xx-xx_xx-xx-xx/model_xx.pt
```

**Note**：

- During training, policy.onnx and policy.onnx.data are also exported for deployment onto physical robots.

**Visualization**：

| Go2                              | G1                             | H1_2                               | G1_mimic                          |
|----------------------------------|--------------------------------|------------------------------------|-----------------------------------|
| ![go2](doc/gif/go2-velocity.gif) | ![g1](doc/gif/g1-velocity.gif) | ![h1_2](doc/gif/h1_2-velocity.gif) | ![g1_mimic](doc/gif/g1-mimic.gif) |

### 4. Real Deployment

Before deployment, install the required communication tools:
- [cyclonedds](https://github.com/eclipse-cyclonedds/cyclonedds.git)
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2.git)

<div style="margin-left: 20px;">

#### 4.1 Power On the Robot
Start the robot in suspended state and wait until it enters `zero-torque` mode.

#### 4.2 Enable Debug Mode
While in `zero-torque` mode, press `L2 + R2` on the controller. The robot will enter `debug mode` with joint damping enabled.

#### 4.3 Connect to the Robot
Connect your PC to the robot via Ethernet. Configure the network as:
- Address：`192.168.123.222`
- Netmask：`255.255.255.0`

Use `ifconfig` to determine the Ethernet device name for deployment.

#### 4.4 Compilation

Example: Unitree G1 velocity control.
Place `policy.onnx` and `policy.onnx.data` into: `deploy/robots/g1/config/policy/velocity/v0/exported`.
Then compile:

```bash
cd deploy/robots/g1
mkdir build && cd build
cmake .. && make
```

#### 4.5 Deployment

## 4.5.1 Simulation Deployment

Before deploying on the real robot, it is recommended to perform simulation deployment using [unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
to prevent abnormal behaviors on the physical robot. This framework has already integrated it.

Build unitree_mujoco：

```bash
cd simulate
mkdir build && cd build
cmake .. && make -j8
```

Launch the simulator (note that a gamepad must be connected):

```bash
./simulate/build/unitree_mujoco
```

You can select the corresponding robot in `simulate/config`

Launch the simulation control program:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=lo
```

## 4.5.2 Real-Robot Deployment

Launch the control program on the real robot:

```bash
cd deploy/robots/g1/build
./g1_ctrl --network=enp5s0
```

**Arguments**：
- `network`: The network interface used to connect to the robot. Use `lo` for simulation deployment, and `enp5s0` for the real robot(You can check it using the `ifconfig` command) 

</div>

**Deployment Results**：

| Go2                                                    | G1                                                    | H1_2           | G1_mimic                                           |
|--------------------------------------------------------|-------------------------------------------------------|----------------|----------------------------------------------------|
| <img src="doc/gif/go2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-velocity-real.gif" width="300"/> | <img src="doc/gif/h1_2-velocity-real.gif" width="300"/> | <img src="doc/gif/g1-mimic-real.gif" width="300"/> |


---

## 🔧 Fork Modifications

> The following features were added in this fork on top of the original repository.

### 1. Electric Motor Actuator for Go2

Added an **electric motor ODE actuator** that models current-torque dynamics (dI/dt) of brushless DC motors, more physically accurate than a simple PD controller.

**New files:**
- `src/assets/robots/unitree_go2/electric_actuator.py` — `ElectricMotorActuatorCfg` class (Python Backward Euler solver)
- `src/assets/robots/unitree_go2/coupled_ode_solver.py` — 2×2 coupled electrical-mechanical ODE solver

**Modified files:**
- `src/assets/robots/unitree_go2/go2_constants.py` — added `GO2_ELECTRIC_*` configs
- `src/tasks/velocity/config/go2/env_cfgs.py` — added `unitree_go2_flat_electric_env_cfg()`
- `src/tasks/velocity/config/go2/__init__.py` — registered `Unitree-Go2-Flat-Electric` task

Training:
```bash
python scripts/train.py Unitree-Go2-Flat-Electric --env.scene.num-envs=4096
```

### 2. MuJoCo-Native Electric Motor (act/act_dot Integration)

Integrates motor current $I$ directly into MuJoCo's state vector (`d->act`) so the electrical ODE is solved **inside** MuJoCo's implicit solver — not in an external Python loop.

**Architecture:**

```
Policy (200Hz, 5ms)
  │ position_target
  ▼
┌──────────────────────────────────────────────────────┐
│  NativeElectricActuator (black-box sub-stepping)     │
│                                                       │
│  Sub 0:  τ_des = PD(target, q₀, v₀)  ← compute once │
│          I_des = τ_des / (Kt·gr)      ← ZOH cache    │
│                                                       │
│  Sub k:  ω_k ← latest MuJoCo velocity                │
│          V_k = R·I_des + Ke·gr·ω_k   ← back-EMF FF  │
│          ctrl_k → mj_step()                           │
│          ...                                          │
│  Sub 49: → (q, v) returned to policy                  │
└──────────────────────────────────────────────────────┘
```

**Key features:**
- `dyntype=filterexact`: MuJoCo analytically includes $\partial(\dot{I})/\partial I = -R/L$ in implicit solver matrix
- Sub-stepping: 50 × 0.1ms physics steps per 5ms policy step, with back-EMF updated every sub-step
- Zero-order hold: $\tau_{des}$ computed once per policy step, voltage controller updates every sub-step
- Bus voltage saturation: optional `V_bus` limit for realistic high-speed torque rolloff
- GPU compatible: uses MuJoCo built-in `filterexact` (no Python callbacks), works with `mujoco_warp`

**New files:**
- `src/assets/robots/unitree_go2/mj_native_electric_actuator.py` — `NativeElectricActuatorCfg` / `NativeElectricActuator`
- `src/assets/robots/unitree_go2/electric_motor_callback.c` — optional C callback for `dyntype=user` approach

| | Standard Go2 | Electric (Python ODE) | Native Electric (MuJoCo act) |
|---|---|---|---|
| Task ID | `Unitree-Go2-Flat` | `Unitree-Go2-Flat-Electric` | `Unitree-Go2-Flat-Native-Electric` |
| Actuator | PD control | Backward Euler 2×2 ODE | MuJoCo `filterexact` + sub-stepping |
| Current state | N/A | Python tensor | MuJoCo `d->act` (engine-integrated) |
| Implicit Jacobian | N/A | N/A | $\partial\dot{I}/\partial I = -R/L$ ✓ |
| GPU (mujoco_warp) | ✓ | ✓ | ✓ |

Training:
```bash
python scripts/train.py Unitree-Go2-Flat-Native-Electric --env.scene.num-envs=4096
```

### 3. Motor Tracking Visualization

`scripts/plot_electric_motor.py` generates per-joint plots for both Electric and Native-Electric actuators. Physics dt and decimation are auto-detected from the task config.

```bash
python scripts/plot_electric_motor.py <TASK_ID> \
    --checkpoint-file logs/rsl_rl/go2_velocity/<run>/model_XXXX.pt
```

| Argument | Default | Description |
|---|---|---|
| `--checkpoint-file` | (required) | Path to trained checkpoint `.pt` file |
| `--num-steps` | `50` | Policy steps to collect |
| `--joints` | all 12 | Comma-separated joint names, e.g. `FR_hip_joint,FR_thigh_joint,FR_calf_joint` |
| `--vx` | random | Forward velocity command [m/s] |
| `--vy` | random | Lateral velocity command [m/s] |
| `--wz` | random | Yaw rate command [rad/s] |
| `--tag` | `""` | Filename prefix (auto-generated from vx/vy/wz if empty) |
| `--out` | `motor_tracking` | Output directory |
| `--plots` | `1,2,3,4,5,6,7,8` | Plot numbers to generate |
| `--current-window-ms` | `60.0` | Time range for zoomed plots [ms] |

**Plot numbers:** 1=position, 2=torque, 3=torque zoomed, 4=torque residual, 5=current, 6=current zoomed, 7=current residual, 8=coupling verification (ω ↔ back-EMF ↔ I)

Example:
```bash
python scripts/plot_electric_motor.py Unitree-Go2-Flat-Native-Electric \
    --checkpoint-file logs/rsl_rl/go2_velocity/2026-04-01/model_5000.pt \
    --joints FR_hip_joint,FR_thigh_joint,FR_calf_joint \
    --vx 0.5 --vy 0.0 --wz 0.0 \
    --plots 2,5,8 --tag forward_05
```

### 4. Play Script Improvements

Added two options to `scripts/play.py`:

**Fixed velocity command** — lock `--vx/--vy/--wz` to a constant instead of random resampling:
```bash
python scripts/play.py Unitree-Go2-Flat --checkpoint_file=... --vx 0.5 --vy 0.0 --wz 0.0
```

**Step limit** — auto-stop after N policy steps:
```bash
python scripts/play.py Unitree-Go2-Flat --checkpoint_file=... --num-steps 200
```

---

## 🎉  Acknowledgements

This project would not be possible without the contributions of the following repositories:

- [mjlab](https://github.com/mujocolab/mjlab.git): training and execution framework
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking.git): versatile humanoid motion tracking framework
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): reinforcement learning algorithm implementation
- [mujoco_warp](https://github.com/google-deepmind/mujoco_warp.git): GPU-accelerated rendering and simulation interface
- [mujoco](https://github.com/google-deepmind/mujoco.git): high-fidelity rigid-body physics engine
