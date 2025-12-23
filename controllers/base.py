"""
Base controller class for the flying car.

Provides shared functionality for model parameter extraction,
linearization, state error computation, and quaternion utilities.
"""

import json
from abc import ABC, abstractmethod

import mujoco
import numpy as np


def load_controller_weights(filepath="lqr_weights.json"):
    """Load controller weights from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_controller_weights(weights, filepath="lqr_weights.json"):
    """Save controller weights to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(weights, f, indent=2)


class FlyingCarControllerBase(ABC):
    """
    Base class for flying car controllers.

    Provides shared functionality for model parameter extraction,
    linearization, state error computation, and quaternion utilities.
    """

    def __init__(self, model, weights=None):
        """
        Initialize the controller base.

        Args:
            model: MuJoCo model object
            weights: Optional dict with controller weights (from JSON config)
        """
        self.model = model
        self.weights = weights

        # Extract physical parameters from model
        self._extract_model_parameters()

        # Compute and store hover equilibrium
        self.hover_qpos, self.hover_qvel, self.hover_ctrl = self._compute_hover_equilibrium()

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

        # State dimension in tangent space (for linearization)
        self.nx = 2 * self.nv

        # Hover thrust per thruster
        self.hover_thrust = self.total_mass * self.gravity / 4

    def _compute_hover_equilibrium(self):
        """
        Compute the hover equilibrium state and control.

        Returns:
            tuple: (qpos, qvel, ctrl) at hover equilibrium
        """
        data = mujoco.MjData(self.model)
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

        return data.qpos.copy(), data.qvel.copy(), data.ctrl.copy()

    def _linearize(self, qpos, qvel, ctrl):
        """
        Linearize the dynamics about a given operating point.

        Args:
            qpos: Position vector at linearization point
            qvel: Velocity vector at linearization point
            ctrl: Control vector at linearization point

        Returns:
            tuple: (A_d, B_d) discrete-time state-space matrices
        """
        # Create a temporary data object for linearization
        data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, data)

        # Set the operating point
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl

        # Allocate output matrices (discrete-time)
        A_d = np.zeros((self.nx, self.nx))
        B_d = np.zeros((self.nx, self.nu))

        # Finite difference parameters
        eps = 1e-6  # Perturbation size
        flg_centered = 1  # Use centered differences (more accurate)

        # Compute linearization
        mujoco.mjd_transitionFD(
            self.model, data, eps, flg_centered,
            A_d, B_d, None, None
        )

        return A_d, B_d

    def _build_state_error(self, data, target_pos, target_quat=None):
        """
        Build the state error vector in tangent space.

        Args:
            data: MuJoCo data object with current state
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [w, x, y, z] (default: identity/level)

        Returns:
            x_error: State error vector (nx,)
        """
        # Default to level orientation if not specified
        if target_quat is None:
            target_quat = np.array([1, 0, 0, 0])

        # Current quaternion
        quat = data.qpos[3:7]

        # Get rotation matrix from body to world frame
        R_body_to_world = self._quat_to_rot_matrix(quat)
        R_world_to_body = R_body_to_world.T

        # Position error in world frame, transformed to body frame
        pos_error_world = data.qpos[0:3] - target_pos
        pos_error_body = R_world_to_body @ pos_error_world

        # Orientation error in tangent space (axis-angle representation)
        ori_error = self._quat_error(quat, target_quat)

        # Gimbal joint position errors (current - hover = current - 0)
        gimbal_pos_error = data.qpos[7:] - self.hover_qpos[7:]

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

        return x_error

    def _saturate_control(self, ctrl):
        """
        Clip control to actuator limits.

        Args:
            ctrl: Control vector

        Returns:
            ctrl: Saturated control vector
        """
        ctrl = np.asarray(ctrl).flatten()
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
        # For quaternion [w, x, y, z]: sin(theta/2) = ||[x,y,z]||, cos(theta/2) = w
        axis = np.array([x_e, y_e, z_e])
        sin_half_angle = np.linalg.norm(axis)

        if sin_half_angle < 1e-10:
            # Near-zero rotation, avoid division by zero
            return np.zeros(3)

        # Normalize axis
        axis = axis / sin_half_angle

        # Compute angle: theta = 2 * atan2(sin(theta/2), cos(theta/2))
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

    @abstractmethod
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
        pass

    def update_weights(self, weights):
        """Update controller weights. Override in subclass if needed."""
        self.weights = weights
