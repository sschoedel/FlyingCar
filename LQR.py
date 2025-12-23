import json
import mujoco
import numpy as np
from scipy import linalg


def load_lqr_weights(filepath="lqr_weights.json"):
    """Load LQR weights from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_lqr_weights(weights, filepath="lqr_weights.json"):
    """Save LQR weights to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(weights, f, indent=2)


class FlyingCarLQR:
    """
    LQR controller for the flying car.

    Solves a single LQR problem for the full system state.
    """

    def __init__(self, model, weights=None):
        """
        Initialize the LQR controller.

        Args:
            model: MuJoCo model object
            weights: Optional dict with Q and R weights (from JSON config)
        """
        # Store model reference
        self.model = model
        self.weights = weights

        # Extract physical parameters from model
        self._extract_model_parameters()

        # Build linearized state-space model (A, B matrices)
        self._build_state_space()

        # Design LQR gain matrix
        self._solve_lqr()

    def update_weights(self, weights):
        """Update weights and recompute LQR gain."""
        self.weights = weights
        self._solve_lqr()

    def _extract_model_parameters(self):
        """Extract masses, inertias, geometry from the MuJoCo model."""
        # Get total mass
        self.total_mass = 0
        for i in range(self.model.nbody):
            self.total_mass += self.model.body_mass[i]

        self.gravity = 9.81

        # Dimensions
        self.nq = self.model.nq  # Number of position coordinates
        self.nv = self.model.nv  # Number of velocity coordinates (DOFs)
        self.nu = self.model.nu  # Number of actuators

        # Hover thrust per thruster
        self.hover_thrust = self.total_mass * self.gravity / 4

    def _build_state_space(self):
        """
        Build the linearized A and B matrices for the system.

        State vector: [qpos, qvel] - full MuJoCo state
        Control vector: [thrust_fr, thrust_fl, thrust_rr, thrust_rl,
                        pitch_fr, roll_fr, pitch_fl, roll_fl,
                        pitch_rr, roll_rr, pitch_rl, roll_rl]

        Linearized about hover equilibrium using MuJoCo's mjd_transitionFD.
        """
        # Create a temporary data object for linearization
        data = mujoco.MjData(self.model)

        # Set hover equilibrium state
        # qpos: [x, y, z, qw, qx, qy, qz, joint_angles...]
        mujoco.mj_resetData(self.model, data)

        # Position: hover at origin, identity quaternion (level)
        data.qpos[0:3] = [0, 0, 2]  # x, y, z
        data.qpos[3:7] = [1, 0, 0, 0]  # quaternion (w, x, y, z) - identity
        # All gimbal joints at zero (thrusters pointing down)
        data.qpos[7:] = 0

        # Zero velocity
        data.qvel[:] = 0

        # Hover control: equal thrust on all 4 thrusters, gimbals at zero
        data.ctrl[0:4] = self.hover_thrust  # Thrust actuators
        data.ctrl[4:] = 0  # Gimbal actuators (position targets = 0)

        # Store hover state and control for later use
        self.hover_qpos = data.qpos.copy()
        self.hover_qvel = data.qvel.copy()
        self.hover_ctrl = data.ctrl.copy()

        # Compute derivatives using finite differences
        # State dimension for mjd_transitionFD is 2*nv (qpos uses nv, not nq, due to quaternion)
        nx = 2 * self.nv

        # Allocate output matrices (discrete-time)
        self.A_d = np.zeros((nx, nx))
        self.B_d = np.zeros((nx, self.nu))

        # Finite difference parameters
        eps = 1e-6  # Perturbation size
        flg_centered = 1  # Use centered differences (more accurate)

        # Compute linearization
        mujoco.mjd_transitionFD(
            self.model, data, eps, flg_centered,
            self.A_d, self.B_d, None, None
        )

        # Store state dimension
        self.nx = nx

    def _solve_lqr(self):
        """
        Solve the discrete-time algebraic Riccati equation
        and compute the optimal gain matrix K.

        Minimizes: J = sum( x'Qx + u'Ru )
        Subject to: x_{k+1} = A_d x_k + B_d u_k
        """
        # State weighting matrix Q (nx x nx)
        # State: [qpos_tangent (nv), qvel (nv)]
        # qpos_tangent: [x, y, z, roll, pitch, yaw, gimbal_joints...]
        # qvel: [vx, vy, vz, wx, wy, wz, gimbal_joint_vels...]
        Q = np.eye(self.nx)

        # Get weights from config or use defaults
        if self.weights is not None:
            qw = self.weights.get("Q", {})
            pos = qw.get("position", {})
            ori = qw.get("orientation", {})
            vel = qw.get("velocity", {})
            ang = qw.get("angular_velocity", {})

            # Position weights
            Q[0, 0] = pos.get("x", 100.0)
            Q[1, 1] = pos.get("y", 100.0)
            Q[2, 2] = pos.get("z", 200.0)

            # Orientation weights
            Q[3, 3] = ori.get("roll", 50.0)
            Q[4, 4] = ori.get("pitch", 50.0)
            Q[5, 5] = ori.get("yaw", 50.0)

            # Gimbal joint positions
            gimbal_pos_weight = qw.get("gimbal_pos", 1.0)
            for i in range(6, self.nv):
                Q[i, i] = gimbal_pos_weight

            # Velocity weights
            Q[self.nv + 0, self.nv + 0] = vel.get("vx", 10.0)
            Q[self.nv + 1, self.nv + 1] = vel.get("vy", 10.0)
            Q[self.nv + 2, self.nv + 2] = vel.get("vz", 10.0)

            # Angular velocity weights
            Q[self.nv + 3, self.nv + 3] = ang.get("wx", 5.0)
            Q[self.nv + 4, self.nv + 4] = ang.get("wy", 5.0)
            Q[self.nv + 5, self.nv + 5] = ang.get("wz", 5.0)

            # Gimbal joint velocities
            gimbal_vel_weight = qw.get("gimbal_vel", 0.1)
            for i in range(6, self.nv):
                Q[self.nv + i, self.nv + i] = gimbal_vel_weight
        else:
            # Default weights
            Q[0, 0] = 100.0  # x
            Q[1, 1] = 100.0  # y
            Q[2, 2] = 200.0  # z
            Q[3, 3] = 50.0   # roll
            Q[4, 4] = 50.0   # pitch
            Q[5, 5] = 50.0   # yaw
            for i in range(6, self.nv):
                Q[i, i] = 1.0
            Q[self.nv + 0, self.nv + 0] = 10.0  # vx
            Q[self.nv + 1, self.nv + 1] = 10.0  # vy
            Q[self.nv + 2, self.nv + 2] = 10.0  # vz
            Q[self.nv + 3, self.nv + 3] = 5.0   # wx
            Q[self.nv + 4, self.nv + 4] = 5.0   # wy
            Q[self.nv + 5, self.nv + 5] = 5.0   # wz
            for i in range(6, self.nv):
                Q[self.nv + i, self.nv + i] = 0.1

        # Control weighting matrix R (nu x nu)
        R = np.eye(self.nu)

        if self.weights is not None:
            rw = self.weights.get("R", {})
            thrust_weight = rw.get("thrust", 0.01)
            gimbal_weight = rw.get("gimbal", 0.1)

            for i in range(4):
                R[i, i] = thrust_weight
            for i in range(4, self.nu):
                R[i, i] = gimbal_weight
        else:
            for i in range(4):
                R[i, i] = 0.01
            for i in range(4, self.nu):
                R[i, i] = 0.1

        # Solve discrete-time algebraic Riccati equation
        # P = A'PA - A'PB(R + B'PB)^{-1}B'PA + Q
        P = linalg.solve_discrete_are(self.A_d, self.B_d, Q, R)

        # Compute optimal gain matrix
        # K = (R + B'PB)^{-1} B'PA
        self.K = np.linalg.inv(R + self.B_d.T @ P @ self.B_d) @ self.B_d.T @ P @ self.A_d

        # Store Q and R for reference
        self.Q = Q
        self.R = R

    def compute_control(self, data, target_pos, target_quat=None):
        """
        Compute control inputs to fly to target position and orientation.

        Args:
            data: MuJoCo data object
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [w, x, y, z] (default: identity/level)

        Returns:
            ctrl: Control vector matching model.nu (number of actuators)
        """
        # Default to level orientation if not specified
        if target_quat is None:
            target_quat = np.array([1, 0, 0, 0])

        # Compute state error (current state - desired state)
        # For LQR: u = -K @ x_error

        # Current quaternion
        quat = data.qpos[3:7]

        # Get rotation matrix from body to world frame
        R_body_to_world = self._quat_to_rot_matrix(quat)
        R_world_to_body = R_body_to_world.T

        # Build current state in tangent space representation
        # Position error in world frame
        pos_error_world = data.qpos[0:3] - target_pos

        # Transform position error to body frame
        # This way the controller always sees errors in its own frame
        pos_error_body = R_world_to_body @ pos_error_world

        # Compute orientation error in tangent space (axis-angle representation)
        # For small angles, this approximates roll, pitch, yaw errors
        ori_error = self._quat_error(quat, target_quat)

        # Gimbal joint position errors (current - hover = current - 0)
        gimbal_pos_error = data.qpos[7:] - self.hover_qpos[7:]

        # Velocity errors
        # Linear velocity in world frame -> transform to body frame
        vel_world = data.qvel[0:3]
        vel_body = R_world_to_body @ vel_world

        # Angular velocity (already in body frame in MuJoCo)
        ang_vel = data.qvel[3:6]

        # Gimbal joint velocities
        gimbal_vel = data.qvel[6:]

        # Stack into full state error vector
        x_error = np.concatenate([
            pos_error_body,
            ori_error,
            gimbal_pos_error,
            vel_body,
            ang_vel,
            gimbal_vel
        ])

        # Compute control deviation from hover
        u_delta = -self.K @ x_error

        # Add hover control to get absolute control
        ctrl = self.hover_ctrl + u_delta

        # Clip to actuator limits
        for i in range(self.nu):
            ctrl[i] = np.clip(
                ctrl[i],
                self.model.actuator_ctrlrange[i, 0],
                self.model.actuator_ctrlrange[i, 1]
            )

        return ctrl

    def _quat_error(self, quat, quat_des):
        """
        Compute orientation error in tangent space (3D).

        Returns exact axis-angle error vector using atan2 formula.
        """
        # quat = [w, x, y, z]
        w, x, y, z = quat
        w_d, x_d, y_d, z_d = quat_des

        # Quaternion error: q_err = q_des^{-1} * q
        # For unit quaternions, inverse is conjugate
        # q_des^{-1} = [w_d, -x_d, -y_d, -z_d]
        # Quaternion multiplication
        w_e = w_d * w + x_d * x + y_d * y + z_d * z
        x_e = w_d * x - x_d * w - y_d * z + z_d * y
        y_e = w_d * y + x_d * z - y_d * w - z_d * x
        z_e = w_d * z - x_d * y + y_d * x - z_d * w

        # Take shorter path (ensure w_e >= 0)
        if w_e < 0:
            w_e = -w_e
            x_e, y_e, z_e = -x_e, -y_e, -z_e

        # Convert to axis-angle using exact atan2 formula
        # For quaternion [w, x, y, z]: sin(θ/2) = ||[x,y,z]||, cos(θ/2) = w
        axis = np.array([x_e, y_e, z_e])
        sin_half_angle = np.linalg.norm(axis)

        if sin_half_angle < 1e-10:
            # Near-zero rotation, avoid division by zero
            return np.zeros(3)

        # Normalize axis
        axis = axis / sin_half_angle

        # Compute angle: θ = 2 * atan2(sin(θ/2), cos(θ/2))
        angle = 2.0 * np.arctan2(sin_half_angle, w_e)

        return axis * angle

    def _quat_to_rot_matrix(self, quat):
        """
        Convert quaternion (w, x, y, z) to 3x3 rotation matrix.
        """
        w, x, y, z = quat

        R = np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

        return R

