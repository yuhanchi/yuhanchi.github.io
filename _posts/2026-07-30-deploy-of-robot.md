---
title: "From an ONNX policy to a running robot"
date: "2026-07-30"
description: "A practical guide to running an ONNX policy on a robot with safe real-time control loops."
category: "tech"
tags: ["robotics"]
---
Robotics is an engineering discipline.

## What this deployment program does

This post assumes basic familiarity with C++ or machine learning, but not with
robot hardware. At the highest level, the program implements one path:

```text
robot sensor state
-> construct the neural-network inputs
-> run ONNX inference
-> convert the output into joint targets
-> apply safety and state-transition checks
-> send motor commands
```

Four terms appear throughout the code:

- **LowState** is the latest low-level sensor state received from the robot,
  including joint positions and IMU readings.
- A **policy** is the neural network that converts those readings into an
  action.
- **LowCmd** is the low-level command sent back to the motors.
- A **finite state machine (FSM)** decides which operating mode currently owns
  those commands: passive, fixed standing, velocity control, or mimic control.

The central design is **two loops running at different rates and exchanging the
latest action**. The policy loop runs at about 50 Hz and produces joint targets.
The FSM loop runs at 1 kHz, publishes commands over DDS, and checks whether the
robot should remain in or leave its current operating state. DDS is the
communication system used here to exchange low-level robot states and commands.

The faster loop must not wait for neural-network inference. Most of the code
structure exists to preserve that separation.

## A map of the repository

The five directories below are enough to orient a first reading:

```text
deploy/
├── include/FSM/       # Operating modes, transitions, and the 1 kHz loop
├── include/isaaclab/  # Build model inputs, run ONNX, interpret outputs
├── robots/g1/         # G1-specific code, configuration, and policies
├── thirdparty/        # ONNX Runtime and other external libraries
└── tools/             # Utilities for checking a model without a robot
```

Inside `robots/g1`, `main.cpp` is the program entrance. It loads configuration,
starts DDS communication, constructs the robot interface, and starts the FSM.
Shared control logic remains under `include/`; details that apply only to the G1
remain under `robots/g1/`.

The most relevant files are:

| Path | Responsibility |
|---|---|
| `include/FSM/CtrlFSM.h` | Run the fast control loop and change operating states |
| `include/FSM/State_RLBase.h` | Own the slower policy thread for an RL-controlled state |
| `include/isaaclab/envs/manager_based_rl_env.h` | Connect observation, inference, and action processing |
| `include/isaaclab/algorithms/algorithms.h` | Run the exported model through ONNX Runtime |
| `include/unitree_articulation.h` | Convert Unitree hardware data into policy-facing state |
| `robots/g1/config/config.yaml` | Select available FSM states and policy directories |
| `robots/g1/config/policy/` | Store versioned models and deployment parameters |
| `tools/verify_policy_onnx.cpp` | Check that a policy loads without real hardware |

The complete tree is retained below as a reference; it is not required for
understanding the runtime flow.

<details markdown="1">
<summary>Complete deployment structure</summary>

```text
deploy/
├── include/                          # Public libraries shared by different robots
│   ├── param.h                       # Configuration, parameters, paths, and policy selection
│   ├── unitree_articulation.h        # Convert Unitree LowState into what a policy needs
│   ├── unitree_joystick_dsl.hpp      # Convert joystick commands into state-switch expressions
│   ├── LinearInterpolator.h          # Smoothly move joints from one pose to another
│   │
│   │
│   ├── FSM/                          # ** IMPORTANT ** Finite State Machine
│   │   ├── BaseState.h               # A very clean template for a state and its registration
│   │   ├── FSMState.h                # Inherits BaseState and adapts it to the Unitree SDK
│   │   │
│   │   ├── State_Passive.h           # Some concrete "action choices"
│   │   ├── State_RLBase.h            # It also starts the separate policy thread
│   │   ├── State_FixStand.h
│   │   │
│   │   │                             # These .h files are not merely declarations:
│   │   │                             # much of their implementation is inline here.
│   │   │
│   │   └── CtrlFSM.h                 # The real Finite State Machine
│   │                                 # It owns the 1 kHz state loop and controls switches.
│   │                                 # Policy inference has another, slower thread.
│   │
│   │
│   └── isaaclab/                     # ** It is not a real simulator **
│       ├── envs/
│       │   ├── manager_based_rl_env.h            # A compact controller: observe -> infer -> act
|       |   |
│       │   └── mdp/                              # Atomic deployment functions
│       │       ├── observations/observations.h   # Atomic skill: construct observations
│       │       ├── actions/joint_actions.h       # Atomic skill: convert actions to joint targets
│       │       └── terminations.h                # Safety checks used by the outer FSM
│       │
│       ├── manager/
│       │   ├── observation_manager.h  # Manage observations + REGISTER_OBSERVATION macro
│       │   ├── action_manager.h       # Manage actions + REGISTER_ACTION macro
│       │   └── manager_term_cfg.h     # Common configuration for registered terms
│       │
│       ├── assets/
│       │   └── articulation/
│       │       └── articulation.h     # Abstract, policy-facing robot state
│       │
│       ├── algorithms/
│       │   └── algorithms.h           # ** IMPORTANT ** OrtRunner actually runs ONNX
│       │
│       ├── devices/
│       │   └── keyboard/
│       │       └── keyboard.h         # Optional keyboard input
│       │
│       └── utils/
│           └── utils.h
│
│
├── robots/
│   └── g1/                            # Exclusive code for G1
│       ├── CMakeLists.txt             # Build instructions; selects x64 or ARM64 ONNX Runtime
│       ├── main.cpp                   # ENTRANCE: configuration -> DDS -> robot -> FSM
│       │
│       ├── include/
│       │   ├── Types.h                # G1-specific LowState and LowCmd type aliases
│       │   └── State_Mimic.h          # Mimic-state declaration + MotionLoader
│       │
│       ├── src/
│       │   ├── State_RLBase.cpp       # Robot-specific realization of State_RLBase
│       │   └── State_Mimic.cpp        # Robot-specific realization of State_Mimic
│       │
│       │                              # This separation makes the shared FSM reusable,
│       │                              # while G1-specific details remain under robots/g1.
│       │
│       │
│       ├── config/
│       │   ├── config.yaml            # Which FSM states exist, how they switch,
│       │   │                          # and which policy directory each state uses
│       │   │
│       │   └── policy/
│       │       ├── velocity/
│       │       │   ├── v00/
│       │       │   │   ├── exported/
│       │       │   │   │   └── policy.onnx   # The neural network itself
│       │       │   │   └── params/
│       │       │   │       └── deploy.yaml    # Observation/action/PD contract
│       │       │   │
│       │       │   └── v01/                    # Another compatible version
│       │       │       ├── exported/
│       │       │       │   └── policy.onnx
│       │       │       └── params/
│       │       │           └── deploy.yaml
│       │       │
│       │       └── mimic/
│       │           └── dance1_subject2/
│       │               ├── exported/
│       │               │   ├── policy.onnx
│       │               │   └── policy.onnx.data
│       │               │       # Must stay beside policy.onnx when weights are external
│       │               └── params/
│       │                   ├── deploy.yaml
│       │                   └── dance1_subject2.npz
│       │                       # The reference motion used by this mimic state
│       │
│       └── build/
│           └── g1_ctrl                # The final compiled program
│
│
├── thirdparty/
│   ├── onnxruntime-linux-x64-1.22.0/          # For an x86_64 laptop/host
│   ├── onnxruntime-linux-aarch64-1.22.0/      # For an ARM64 onboard computer
│   └── cnpy/                                  # Read .npy/.npz motion files
│
│
└── tools/
    └── verify_policy_onnx.cpp         # Load and check a policy without the real robot
```

</details>




## Where an ONNX policy belongs

A normal G1 velocity policy should be installed as one self-contained version:

```text
deploy/robots/g1/config/policy/velocity/v02/
├── exported/
│   ├── policy.onnx
│   └── policy.onnx.data        # Include only if the ONNX model references it
└── params/
    └── deploy.yaml
```

The model and its `deploy.yaml` form one deployment artifact. They must not be
mixed across training runs.

`deploy.yaml` is the runtime **contract**. It defines:

- policy period (`step_dt`);
- robot-to-policy joint mapping;
- default joint pose;
- joint stiffness and damping;
- observation names, scales, clipping, and history lengths;
- action joint order, scale, offset, and clipping.

Copying only `policy.onnx` is safe only when the new policy was trained with
exactly the same contract as the existing `deploy.yaml`. (observation,
history length, joint ordering, action scale, default pose, PD gain, or policy
frequency)



## How the policy is loaded

`State_RLBase` resolves the selected policy directory and loads two files:

```cpp
auto policy_dir =
    param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
    YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
    std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
        FSMState::lowstate
    )
);

env->alg = std::make_unique<isaaclab::OrtRunner>(
    policy_dir / "exported" / "policy.onnx"
);
```

`ManagerBasedRLEnv` builds the observation and action managers from `deploy.yaml`. <br>
`OrtRunner` then reads the ONNX input names and shapes directly from the model. <br>
At every policy step, each ONNX input name must exist in the observation map; otherwise the runtime throws an error instead of silently
feeding the wrong tensor.<br>

> "Ort" refers to "ONNX Runtime"


## Runtime organization

The controller uses two periodic loops with different responsibilities:

| Loop | Typical rate | Responsibility |
|---|---:|---|
| Policy thread | 50 Hz | Read observations, run the network, and produce the latest joint targets |
| FSM thread | 1 kHz | Apply the latest targets, publish motor commands, and check transitions and safety conditions |

The policy thread belongs to an RL state: entering the state starts the thread,
and leaving the state stops and joins it. The FSM thread belongs to the whole
process and continues across state changes.

The detailed execution diagram follows. It is useful when tracing the code, but
the table above contains the main architectural point.

<details markdown="1">
<summary>Detailed two-thread execution diagram</summary>

```text
                             G1 LowState over DDS
                                    │
                                    ▼
                      Unitree articulation adapter
                     (env->robot, shared state source)
                                    │
  ┌─────────────────────────────────┼──────────────────────────────────┐
  │  FSM loop -- 1kHz, 1ms period, resident since process start,       │
  │              persists throughout all states' lifecycles            │
  │                                  │                                 │
  │                  currentState->pre_run()                           │
  │                                  │                                 │
  │   currentState->run()  ◄─────────────────────┐                     │
  │                                              │                     │
  │   If currentState == Passive/FixStand:       │                     │
  │    run() executes a deterministic controller │                     │
  │                                              │                     │
  │   ┌──────────────────────────────────────────┴──────────────────┐  │
  │   │ State_RLBase::enter() => spawn, ::exit() => join            │  │
  │   │                                                             │  │
  │   │   Policy thread -- 0.02s/50Hz, attached to this state obj,  │  │
  │   │   spawned and killed along with state transitions           │  │
  │   │                                                             │  │
  │   │   robot state update                                        │  │
  │   │        │                                                    │  │
  │   │   ObservationManager                                        │  │
  │   │        │                                                    │  │
  │   │   named ONNX input tensors                                  │  │
  │   │        │                                                    │  │
  │   │   OrtRunner / ONNX Runtime                                  │  │
  │   │        │                                                    │  │
  │   │   raw neural-network action                                 │  │
  │   │        │                                                    │  │
  │   │   ActionManager -> scaled joint position targets            │  │
  │   │        │                                                    │  │
  │   │   Write to env->action_manager->processed_actions()         │  │
  │   │   (latest-action handoff, written by the policy thread      │  │
  │   │    and read by the FSM thread)                               │  │
  │   └──────────────────────────────────────────┬──────────────────┘  │
  │                                              │                     │
  │   State_RLBase::run() reads processed_actions()                    │
  │                                  │                                 │
  │                                  │                                 │
  │   currentState->post_run() -> Publish G1 LowCmd over DDS           │
  │                                  │                                 │
  │   registered_checks (e.g. bad_orientation) -- evaluated every 1ms, │
  │   independent of Policy thread's status (alive or blocked).        │
  │   Triggers state transition if evaluated to true.                  │
  │   (Switching to Passive triggers current state exit() -> joins     │
  │    Policy thread)                                                  │
  └────────────────────────────────────────────────────────────────────┘
```

</details>

### The policy thread

When an RL state is entered, `State_RLBase::enter()` starts a thread with the
period specified by `step_dt`. Its compact loop is:

```text
read the latest robot state
-> construct named observations
-> run ONNX inference
-> scale/offset/clip the raw action
-> store the latest processed joint targets
```

This deployment environment does not calculate rewards, advance a physics
engine, or return training-style `(observation, reward, done)` values.

### The FSM thread

`CtrlFSM` runs independently at 1 kHz:

```text
pre_run()
-> run the active state
-> post_run()
-> evaluate transition and safety checks
```

In an RL state, `run()` copies the most recent processed policy action into the
corresponding motor command targets. `post_run()` handles low-level command
publication. The FSM is also responsible for transitions among safe passive
mode, fixed standing, velocity control, and mimic control.

The two-loop design allows the robot communication and safety state machine to
run faster than neural-network inference.

## Responsibilities of the main components

### `FSM/`

The FSM is the outer safety and operating-mode layer:

- `BaseState` defines `enter`, `pre_run`, `run`, `post_run`, and `exit`;
- `FSMState` adds Unitree state, command, keyboard, and transition support;
- `CtrlFSM` constructs configured states and dispatches transitions;
- `State_Passive` and `State_FixStand` do not require neural-network inference;
- `State_RLBase` and `State_Mimic` own policy-specific runtime logic.

### `isaaclab/envs/mdp/`

This directory contains deploy-time observation, action, and safety functions.
They are registered by name and instantiated from YAML configuration. It does
not contain the training reward functions of the Python Isaac Lab environment.

### `manager/`

The deployment runtime has an `ObservationManager` and an `ActionManager`:

- `ObservationManager` evaluates configured observation terms and maintains
  their histories;
- `ActionManager` converts raw policy outputs into physical joint position
  targets.

There is no deploy-time `RewardManager`. Termination-style functions are used
as FSM safety checks rather than as an RL training `done` signal.

### `algorithms/algorithms.h`

`OrtRunner` owns the ONNX Runtime environment and session. It:

1. loads `policy.onnx`;
2. obtains all model input names and tensor shapes;
3. maps the observation data to those named inputs;
4. executes the session;
5. copies the first output tensor into the shared action buffer.

### `unitree_articulation.h`

This adapter exposes Unitree `LowState` data in the form required by the
deployment observation functions, including joint state and IMU-derived
quantities. It is the main boundary between hardware communication types and
policy-facing state.


## Deployment checklist

Before running a new policy on hardware, verify:

1. The ONNX file and any external `.data` file are complete and hashed.
2. ONNX Runtime can load the model.
3. Every ONNX input name, shape, and semantic meaning matches the runtime.
4. PyTorch and ONNX outputs agree on representative observations.
5. Joint order, default pose, action scale, and output semantics match.
6. `step_dt`, stiffness, and damping match the training/deployment contract.
7. The policy passes simulator tests before hardware tests.
8. The selected policy version is pinned or logged.
9. Passive mode, fixed stand, emergency transition, and physical support are
   ready before entering the RL state.

The central lesson is simple: an ONNX file is only the neural network. A real
robot deployment is the network plus its observation contract, action contract,
control timing, state machine, hardware transport, and safety procedure.


---


## Advanced: adding terrain perception with the VideoMimic LiDAR pipeline

Everything above assumes the policy already has an observation vector sitting
in memory. For a **blind** (flat-ground) policy that vector comes entirely
from IMU + joint encoders. For a **terrain-aware** policy, one slice of that
vector has to come from a live map of the ground under and around the robot's
feet — and that map has to be built from a physical sensor, in real time, on
hardware that never saw a training simulator.

This section documents how one working, published reference does it:
**VideoMimic** (CoRL 2025), whose `sim2real/` code has run
this exact pipeline on a real Unitree G1. The structure is worth learning even
if you never touch this codebase, because it is the general pattern for
**bolting a sensor onto an existing RL deploy stack without touching the
control loop above**.

This section introduces ROS, LiDAR odometry, point clouds, and elevation maps.
It is independent of the basic deployment path above and can be skipped on a
first reading.

### Why a G1 needs an external sensor at all


```text
factory G1  =  motors + IMU + joint encoders   (no terrain sensing)
      +  Livox MID-360 LiDAR, rigidly mounted on the head/torso
      +  an onboard computer running a separate ROS1 stack
      =  a G1 that can see terrain
```

### Hardware topology

```text
┌───────────────────────────┐
│  G1 robot                 │
│  - motor controllers (DDS)│
│  - Livox MID-360 (bolted  │
│    on, has its own IMU)   │
└─────────────┬─────────────┘
              │ internal gigabit link
              ▼
┌────────────────────────────┐
│  onboard computer          │
│  (Jetson-class, ROS1       │
│   Noetic)                  │
│  - point-cloud driver      │
│  - odometry + mapping      │
│  - policy runtime          │
│  - DDS motor commands      │
└────────────────────────────┘
```

The LiDAR's raw point cloud **never reaches the policy directly**. It is
always pre-digested by a mapping layer first. That one design decision is why
the rest of this section reads as a pipeline, not a single function call.

### The four-layer software stack

| Layer | Component | Job | Talks over |
|---|---|---|---|
| 1 | LiDAR driver | Turn the sensor's private protocol into a standard point-cloud topic | ROS1 topic |
| 2 | LIO odometry (FAST-LIO) | Fuse LiDAR + its own IMU into a real-time pose estimate | ROS1 topic |
| 3 | Elevation mapping | Fuse point cloud + pose into a **robot-centric height map**, accumulated over time | ROS1 topic |
| 4 | Inference process | Subscribe to the height-map topic + DDS robot state, run the network, publish DDS motor commands | ROS1 (in) + DDS (out) |

Layers 1–3 are pure perception and know nothing about the policy. Layer 4 is
the only layer that mixes "ROS-world" and "DDS-world" — it is the seam between
the two ecosystems.

### Data flow: point cloud → motor command

```text
[1] Livox MID-360 scans the scene
        │  raw point cloud
        ▼
[2] LiDAR driver → ROS1 point-cloud topic
        ▼
[3] FAST-LIO
        │  produces a pose estimate ("odometry"),
        │  used only for mapping — never fed to the policy
        │  as a velocity/heading signal (see note below)
        ▼
[4] Elevation mapping
        │  fuses point cloud + pose into a height map,
        │  centered on the robot, updated continuously
        │  (occluded regions persist from earlier frames —
        │   this is a *time-accumulated* map, not a single-frame one)
        ▼
[5] Inference process
        │  a. transform the map into the robot's local frame
        │  b. crop an N×N grid around the robot
        │  c. fill any grid cell with no direct hit using
        │     k-nearest-neighbor + inverse-distance-weighted
        │     interpolation (a KD-tree query)
        ▼
[6] pack the grid into a fixed-shape tensor, e.g. [1, 11, 11]
        ▼
[7] feed it into the network alongside proprioception
        ▼
[8] network outputs a joint-angle delta
        ▼
[9] target = default standing pose + network delta
        ▼
[10] publish over DDS → motors move → LiDAR moves → back to [1]
```

**One rule worth memorizing**: the LIO odometry in step [3] is a *mapping*
signal, not a *control* signal. Feeding it back into the policy as an
estimated heading/velocity is a documented failure mode in this line of
research — small odometry drift compounds into a policy that is confidently
walking in the wrong direction. The odometry's only job is to let the map in
step [4] stay aligned as the robot moves.

### Format contract: what actually crosses the ROS↔policy seam

The inference process (layer 4) is where "a stream of ROS messages" turns into
"one fixed-shape tensor a network can consume." That translation has to be as
disciplined as the ONNX `deploy.yaml` contract described earlier in this post:

| Field | Meaning |
|---|---|
| grid shape | fixed, e.g. `11 × 11` cells |
| cell size | fixed physical spacing, e.g. `0.1 m` |
| alignment | which robot frame the grid is centered/rotated on |
| fill rule | what value a cell gets when no point lands in it (interpolate vs. flag-as-invalid — these are *not* interchangeable choices, see below) |
| update rate | how often the grid is rebuilt (mapping is slower than control) |

Two design choices are easy to get wrong when adapting this pipeline to a
different robot or a different policy:

1. **Mounting point changes the extrinsic calibration.** The transform from
   LiDAR frame to whatever body frame the grid is centered on is a physical
   measurement, not a constant you can copy from someone else's robot. Move
   the sensor from the head to the torso and the whole calibration is void.
2. **"No data" is not the same as "zero."** A cell with no LiDAR hit can be
   *interpolated* (treat it as probably-flat, as VideoMimic's KD-tree does)
   or *flagged invalid* (let the network know this reading is untrustworthy).
   These encode different assumptions about what the policy was trained to
   expect, and silently swapping one for the other changes what the network
   thinks it is looking at.

<!-- ### Safety state machine

The inference process runs its own small state machine, independent of (but
structurally identical in spirit to) the FSM described earlier in this post:

```cpp
enum class ControlState {
    INITIAL_HOLD,      // boot default: hold current pose, do not move
    GOING_TO_DEFAULT,   // transition to / hold the default stance
    RUNNING_POLICY,     // network is actually in control
    SHUTDOWN            // preparing to exit
};
```

The operator manually gates the transition out of `INITIAL_HOLD` once the
elevation map has had time to stabilize — the same "don't trust the network
before its inputs are trustworthy" instinct that motivates the FSM's
`registered_checks` in the rest of this post, just applied to the mapping
layer instead of joint state. -->

### How this connects back to the deploy/ picture above

Overlay this section onto the runtime diagram from earlier: the elevation-map
tensor is just one more named entry that has to show up in the
`ObservationManager`'s output before `OrtRunner` can run. Everything else —
the 50 Hz/1 kHz split, the shared-state handoff, the FSM owning the DDS
publish — is unchanged. Terrain perception does not add a third **control**
loop; it adds asynchronous perception and mapping processes that feed the policy
loop at their own rates.

A production system also needs an explicit failure policy for missing or stale
maps. Depending on how the policy was trained, that may mean holding the last
valid map for a bounded interval, switching to a separately tested blind policy,
or returning the FSM to a safe state. Reusing stale terrain indefinitely does
not by itself produce safe blind behavior.
