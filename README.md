# Flying Car Simulation

A MuJoCo simulation of a flying car with four gimbaled thrusters and an LQR controller for position and orientation tracking.

## Description

This project models a "flying car" - a rectangular prism body with four thrusters at the corners. Each thruster can pitch and roll independently via gimbal joints, enabling full 6-DOF control. The vehicle uses an LQR controller to track position (x, y, z) and yaw orientation.

## Demo

Run the simulation and press **spacebar** to set random target positions and orientations. The green sphere and arrow show the target pose.

## Files

- `flying_car.xml` - MuJoCo model definition
- `main.py` - Simulation loop and visualization
- `LQR.py` - Full-state discrete-time LQR controller

## Requirements

- Python 3.12+
- uv (for dependency management)

## Installation

```bash
git clone git@github.com:sschoedel/FlyingCar.git
cd FlyingCar
uv sync
```

## Running

```bash
uv run main.py
```

### Controls

- **Spacebar** - Set random target position and yaw

## Model

- **Chassis**: 40 kg rectangular body
- **Thrusters**: 4x gimbaled thrusters (9 kg each)
  - 2-DOF gimbal (pitch + roll)
  - 0-500 N thrust range
- **Total mass**: ~76 kg

## Controller

The LQR controller uses MuJoCo's `mjd_transitionFD()` for automatic linearization about hover equilibrium. Key features:

- Discrete-time Riccati equation solver
- Body-frame error transformation for yaw invariance
- Feedforward hover thrust + LQR feedback
