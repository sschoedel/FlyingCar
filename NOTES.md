# Flying Car Project Notes

## Project Overview

A MuJoCo simulation of a rectangular prism "flying car" with four gimbaled thrusters capable of pitch and roll control. Includes a working LQR controller for position and yaw tracking.

## File Structure

- `flying_car.xml` - MuJoCo model definition
- `main.py` - Simulation loop with visualization
- `LQR.py` - Working LQR controller (single full-state LQR)
- `controller.py` - Old cascaded LQR controller (deprecated, kept for comparison)
- `pyproject.toml` - UV project config with dependencies (mujoco, scipy, numpy)

## Model Description (`flying_car.xml`)

### Chassis
- Rectangular prism body (1.0m x 0.6m x 0.2m)
- Mass: 40 kg
- Inertia: diaginertia="1 2 2"
- Green arrow on top indicates forward (+X) direction
- Starts at position (0, 0, 2)

### Thrusters (4 total)
Each thruster is a two-body kinematic chain attached to the chassis:

1. **Pitch body** - attached to chassis with hinge joint (Y-axis rotation)
   - Mass: 4 kg each
   - Joint range: ±45 degrees

2. **Roll body** - attached to pitch body with hinge joint (X-axis rotation)
   - Mass: 5 kg each
   - Joint range: ±45 degrees
   - Contains the thruster site (force application point)

Thruster positions (relative to chassis center):
- Front-right: (0.55, -0.35, 0)
- Front-left: (0.55, 0.35, 0)
- Rear-right: (-0.55, -0.35, 0)
- Rear-left: (-0.55, 0.35, 0)

Total mass: 40 + 4×(4+5) = 76 kg

### Actuators (12 total)
1. **Thrust actuators** (4): Force in +Z direction of thruster site
   - Range: 0-500 N each
   - Names: thrust_fr, thrust_fl, thrust_rr, thrust_rl

2. **Gimbal actuators** (8): Position-controlled joints
   - Range: ±0.785 rad (±45 degrees)
   - Kp: 100
   - Names: pitch_fr, roll_fr, pitch_fl, roll_fl, pitch_rr, roll_rr, pitch_rl, roll_rl

### Collision Settings
- Thruster geoms have `contype="0" conaffinity="0"` (no collisions with chassis)

### Sensors
- Accelerometer and gyro at front-right thruster site
- Frame quaternion, position, and axis sensors for chassis

## Controller (`LQR.py`)

### Architecture
Single full-state discrete-time LQR controller.

### Linearization
- Uses `mujoco.mjd_transitionFD()` to compute discrete-time A and B matrices
- Linearized about hover equilibrium:
  - Position: (0, 0, 2)
  - Orientation: Identity quaternion (level)
  - Gimbals: All at zero
  - Thrust: total_mass × g / 4 per thruster
- State dimension: 2×nv (positions in tangent space + velocities)

### State Vector
```
[x, y, z, roll, pitch, yaw, gimbal_joints..., vx, vy, vz, wx, wy, wz, gimbal_vels...]
```

### LQR Weights
**Q matrix (state cost):**
- Position (x, y): 100
- Position (z): 200
- Orientation (roll, pitch, yaw): 50
- Gimbal positions: 1
- Linear velocities: 10
- Angular velocities: 5
- Gimbal velocities: 0.1

**R matrix (control cost):**
- Thrust actuators: 0.01
- Gimbal actuators: 0.1

### Key Implementation Details

**Body-frame error transformation:**
The controller was linearized at yaw=0, so when the robot is at a different yaw, the mapping between body-frame forces and world-frame motion changes. To handle this:
- Position error is transformed from world frame to body frame
- Linear velocity is transformed from world frame to body frame
- This way the controller always "sees" errors in its own reference frame

**Quaternion error:**
- Computed as `q_err = q_des^{-1} * q`
- Converted to axis-angle representation (approximately roll, pitch, yaw for small angles)
- Takes shortest path by flipping sign if w < 0

**Control computation:**
```
u = hover_ctrl + K @ (-x_error)
```

## Visualization (`main.py`)

### Thrust Arrows
- Orange arrows at each thruster site
- Point in -Z direction of thruster (downward when level)
- Length scaled by thrust magnitude (max 1.0m at full thrust)
- Only visible when thrust > 0.1 N

### Target Marker
- Green semi-transparent sphere at target position
- Green arrow showing target yaw direction
- Press **spacebar** to set random target position and yaw

### Real-time Simulation
- Simulation rate-limited to real-time using wall clock comparison
- Uses `mujoco.viewer.launch_passive` for interactive viewing
- Keyboard callback for interactive target setting

## Dependencies

```
mujoco>=3.4.0
scipy>=1.16.0
numpy>=2.4.0
```

## Running

```bash
uv run main.py
```

Press spacebar to set a new random target position and yaw orientation.

## Development History

### Initial Approach (controller.py - deprecated)
Cascaded LQR with separate position and attitude loops. Did not work well due to complexity and tuning difficulties.

### Working Approach (LQR.py)
Single full-state LQR using MuJoCo's finite-difference linearization. Key insights:
1. Use `mjd_transitionFD()` for automatic linearization - avoids manual derivation errors
2. Use discrete-time Riccati solver (`scipy.linalg.solve_discrete_are`) to match MuJoCo's discrete dynamics
3. Transform position/velocity errors to body frame to handle arbitrary yaw orientations
4. Hover control acts as feedforward, LQR provides feedback corrections
