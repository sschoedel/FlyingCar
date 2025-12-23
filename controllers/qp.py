"""
QP-based controller for the flying car.

Solves a finite-horizon quadratic program at each timestep,
re-linearizing about the current state. This allows for
additional constraints beyond what standard LQR can handle.
"""

import numpy as np
import cvxpy as cp
from scipy import linalg

from .base import FlyingCarControllerBase


class FlyingCarQP(FlyingCarControllerBase):
    """
    QP-based controller for the flying car.

    Solves a finite-horizon quadratic program at each timestep,
    re-linearizing about the current state. Supports:
    - Control input constraints (actuator limits)
    - State constraints (optional)
    - Terminal cost for stability

    The QP formulation is:
        min  sum_{k=0}^{N-1} (x_k'Qx_k + u_k'Ru_k) + x_N'P x_N
        s.t. x_{k+1} = A x_k + B u_k
             u_min <= u_k <= u_max
             (optional state constraints)
    """

    def __init__(self, model, weights=None, horizon=10):
        """
        Initialize the QP controller.

        Args:
            model: MuJoCo model object
            weights: Optional dict with Q and R weights (from JSON config)
            horizon: Prediction horizon (number of steps)
        """
        super().__init__(model, weights)

        self.horizon = horizon

        # Build cost matrices
        self.Q, self.R = self._build_cost_matrices()

        # Compute terminal cost matrix P using Riccati at hover
        # This ensures stability for the finite-horizon problem
        self._compute_terminal_cost()

        # Set up cvxpy problem structure for warm-starting
        self._setup_qp_problem()

    def update_weights(self, weights):
        """Update weights and rebuild QP problem."""
        self.weights = weights
        self.Q, self.R = self._build_cost_matrices()
        self._compute_terminal_cost()
        self._setup_qp_problem()

    def _compute_terminal_cost(self):
        """
        Compute terminal cost matrix P by solving Riccati at hover.

        This provides a stabilizing terminal cost for the finite-horizon QP.
        """
        # Linearize at hover for terminal cost computation
        A_hover, B_hover = self._linearize(
            self.hover_qpos, self.hover_qvel, self.hover_ctrl
        )

        try:
            self.P = linalg.solve_discrete_are(A_hover, B_hover, self.Q, self.R)
        except Exception:
            # Fall back to Q if Riccati fails
            self.P = self.Q.copy()

    def _setup_qp_problem(self):
        """
        Set up the cvxpy problem structure.

        Uses a sparse formulation with both state and control as variables,
        which allows for state constraints and is efficient with cvxpy.
        """
        N = self.horizon
        nx = self.nx
        nu = self.nu

        # Decision variables
        self.x_var = cp.Variable((N + 1, nx))  # States x_0 to x_N
        self.u_var = cp.Variable((N, nu))       # Controls u_0 to u_{N-1}

        # Parameters (updated each solve)
        self.x0_param = cp.Parameter(nx)        # Initial state
        self.A_param = cp.Parameter((nx, nx))   # Dynamics matrix A
        self.B_param = cp.Parameter((nx, nu))   # Dynamics matrix B

        # Build cost function
        cost = 0

        # Stage costs: sum_{k=0}^{N-1} (x_k'Qx_k + u_k'Ru_k)
        Q_sqrt = linalg.sqrtm(self.Q).real
        R_sqrt = linalg.sqrtm(self.R).real
        P_sqrt = linalg.sqrtm(self.P).real

        for k in range(N):
            cost += cp.sum_squares(Q_sqrt @ self.x_var[k])
            cost += cp.sum_squares(R_sqrt @ self.u_var[k])

        # Terminal cost: x_N'Px_N
        cost += cp.sum_squares(P_sqrt @ self.x_var[N])

        # Build constraints
        constraints = []

        # Initial state constraint
        constraints.append(self.x_var[0] == self.x0_param)

        # Dynamics constraints: x_{k+1} = A x_k + B u_k
        for k in range(N):
            constraints.append(
                self.x_var[k + 1] == self.A_param @ self.x_var[k] + self.B_param @ self.u_var[k]
            )

        # Control constraints (actuator limits)
        # Note: These are limits on u_delta (deviation from hover)
        # Actual limits: u_min <= u_hover + u_delta <= u_max
        # So: u_min - u_hover <= u_delta <= u_max - u_hover
        u_min = np.array([self.model.actuator_ctrlrange[i, 0] for i in range(nu)])
        u_max = np.array([self.model.actuator_ctrlrange[i, 1] for i in range(nu)])

        u_delta_min = u_min - self.hover_ctrl
        u_delta_max = u_max - self.hover_ctrl

        for k in range(N):
            constraints.append(self.u_var[k] >= u_delta_min)
            constraints.append(self.u_var[k] <= u_delta_max)

        # Create the problem
        self.problem = cp.Problem(cp.Minimize(cost), constraints)

        # Store for potential state constraints
        self.u_delta_min = u_delta_min
        self.u_delta_max = u_delta_max

    def _build_cost_matrices(self):
        """
        Build Q and R cost matrices from weights config.

        Returns:
            tuple: (Q, R) cost matrices
        """
        Q = np.eye(self.nx)

        if self.weights is not None:
            qw = self.weights.get("Q", {})
            pos = qw.get("position", {})
            ori = qw.get("orientation", {})
            vel = qw.get("velocity", {})
            ang = qw.get("angular_velocity", {})

            Q[0, 0] = pos.get("x", 100.0)
            Q[1, 1] = pos.get("y", 100.0)
            Q[2, 2] = pos.get("z", 200.0)
            Q[3, 3] = ori.get("roll", 50.0)
            Q[4, 4] = ori.get("pitch", 50.0)
            Q[5, 5] = ori.get("yaw", 50.0)

            gimbal_pos_weight = qw.get("gimbal_pos", 1.0)
            for i in range(6, self.nv):
                Q[i, i] = gimbal_pos_weight

            Q[self.nv + 0, self.nv + 0] = vel.get("vx", 10.0)
            Q[self.nv + 1, self.nv + 1] = vel.get("vy", 10.0)
            Q[self.nv + 2, self.nv + 2] = vel.get("vz", 10.0)
            Q[self.nv + 3, self.nv + 3] = ang.get("wx", 5.0)
            Q[self.nv + 4, self.nv + 4] = ang.get("wy", 5.0)
            Q[self.nv + 5, self.nv + 5] = ang.get("wz", 5.0)

            gimbal_vel_weight = qw.get("gimbal_vel", 0.1)
            for i in range(6, self.nv):
                Q[self.nv + i, self.nv + i] = gimbal_vel_weight
        else:
            Q[0, 0] = 100.0
            Q[1, 1] = 100.0
            Q[2, 2] = 200.0
            Q[3, 3] = 50.0
            Q[4, 4] = 50.0
            Q[5, 5] = 50.0
            for i in range(6, self.nv):
                Q[i, i] = 1.0
            Q[self.nv + 0, self.nv + 0] = 10.0
            Q[self.nv + 1, self.nv + 1] = 10.0
            Q[self.nv + 2, self.nv + 2] = 10.0
            Q[self.nv + 3, self.nv + 3] = 5.0
            Q[self.nv + 4, self.nv + 4] = 5.0
            Q[self.nv + 5, self.nv + 5] = 5.0
            for i in range(6, self.nv):
                Q[self.nv + i, self.nv + i] = 0.1

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
        Compute control inputs by solving a QP at the current state.

        Args:
            data: MuJoCo data object
            target_pos: Target position [x, y, z]
            target_quat: Target quaternion [w, x, y, z] (default: identity/level)

        Returns:
            ctrl: Control vector matching model.nu (number of actuators)
        """
        # Build state error vector (this is x_0 for the QP)
        x_error = self._build_state_error(data, target_pos, target_quat)

        # Re-linearize about current state
        A_d, B_d = self._linearize(data.qpos, data.qvel, data.ctrl)

        # Update parameters
        self.x0_param.value = x_error
        self.A_param.value = A_d
        self.B_param.value = B_d

        # Solve the QP
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)

            if self.problem.status in ['optimal', 'optimal_inaccurate']:
                # Extract first control action (receding horizon)
                u_delta = self.u_var.value[0]
            else:
                # Fallback to LQR-style if QP fails
                u_delta = self._fallback_control(A_d, B_d, x_error)
        except Exception:
            # Fallback to LQR-style if solver fails
            u_delta = self._fallback_control(A_d, B_d, x_error)

        # Add hover control to get absolute control
        ctrl = self.hover_ctrl + u_delta

        # Saturate to actuator limits (should already be satisfied by QP)
        return self._saturate_control(ctrl)

    def _fallback_control(self, A_d, B_d, x_error):
        """
        Fallback control using LQR at current linearization.

        Used when QP solver fails.
        """
        try:
            P = linalg.solve_discrete_are(A_d, B_d, self.Q, self.R)
            K = np.linalg.inv(self.R + B_d.T @ P @ B_d) @ B_d.T @ P @ A_d
            u_delta = -K @ x_error
        except Exception:
            # Last resort: zero control deviation
            u_delta = np.zeros(self.nu)

        # Clip to control limits
        u_delta = np.clip(u_delta, self.u_delta_min, self.u_delta_max)
        return u_delta

    def add_state_constraint(self, state_idx, lower=None, upper=None):
        """
        Add a state constraint to the QP.

        Args:
            state_idx: Index of the state to constrain
            lower: Lower bound (None for no lower bound)
            upper: Upper bound (None for no upper bound)

        Note: Call _setup_qp_problem() after adding all constraints,
        or modify this to rebuild incrementally.
        """
        # This is a placeholder for future constraint additions
        # Would need to rebuild the problem with new constraints
        raise NotImplementedError(
            "Dynamic constraint addition not yet implemented. "
            "Modify _setup_qp_problem() directly for custom constraints."
        )
