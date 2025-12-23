"""
LQR controller for the flying car.

Solves a single LQR problem for the full system state,
linearized about hover equilibrium.
"""

import numpy as np
from scipy import linalg

from .base import FlyingCarControllerBase


class FlyingCarLQR(FlyingCarControllerBase):
    """
    LQR controller for the flying car.

    Solves a single LQR problem for the full system state,
    linearized about hover equilibrium.
    """

    def __init__(self, model, weights=None):
        """
        Initialize the LQR controller.

        Args:
            model: MuJoCo model object
            weights: Optional dict with Q and R weights (from JSON config)
        """
        super().__init__(model, weights)

        # Linearize about hover equilibrium (done once)
        self.A_d, self.B_d = self._linearize(
            self.hover_qpos, self.hover_qvel, self.hover_ctrl
        )

        # Design LQR gain matrix
        self._solve_lqr()

    def update_weights(self, weights):
        """Update weights and recompute LQR gain."""
        self.weights = weights
        self._solve_lqr()

    def _solve_lqr(self):
        """
        Solve the discrete-time algebraic Riccati equation
        and compute the optimal gain matrix K.

        Minimizes: J = sum( x'Qx + u'Ru )
        Subject to: x_{k+1} = A_d x_k + B_d u_k
        """
        # Build Q and R matrices from weights
        Q, R = self._build_cost_matrices()

        # Solve discrete-time algebraic Riccati equation
        # P = A'PA - A'PB(R + B'PB)^{-1}B'PA + Q
        P = linalg.solve_discrete_are(self.A_d, self.B_d, Q, R)

        # Compute optimal gain matrix
        # K = (R + B'PB)^{-1} B'PA
        self.K = np.linalg.inv(R + self.B_d.T @ P @ self.B_d) @ self.B_d.T @ P @ self.A_d

        # Store Q and R for reference
        self.Q = Q
        self.R = R

    def _build_cost_matrices(self):
        """
        Build Q and R cost matrices from weights config.

        Returns:
            tuple: (Q, R) cost matrices
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

        return Q, R

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
        # Build state error vector
        x_error = self._build_state_error(data, target_pos, target_quat)

        # Compute control deviation from hover: u = -K @ x_error
        u_delta = -self.K @ x_error

        # Add hover control to get absolute control
        ctrl = self.hover_ctrl + u_delta

        # Saturate to actuator limits
        return self._saturate_control(ctrl)
