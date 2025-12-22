import numpy as np
from scipy import linalg


def quat_to_euler(quat):
    """Convert quaternion (w, x, y, z) to euler angles (roll, pitch, yaw)."""
    w, x, y, z = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


class FlyingCarLQR:
    """
    LQR controller for the flying car.

    Uses a cascaded control approach:
    1. Outer loop: Position control -> desired acceleration -> desired attitude
    2. Inner loop: Attitude control -> thruster gimbal angles
    3. Altitude control -> total thrust
    """

    def __init__(self, model):
        # Get total mass (chassis + all thruster bodies)
        self.total_mass = 0
        for i in range(model.nbody):
            self.total_mass += model.body_mass[i]

        self.gravity = 9.81

        # Thruster positions relative to chassis center (for moment arm calculations)
        self.thruster_positions = np.array([
            [0.55, -0.35, 0],   # front-right
            [0.55, 0.35, 0],    # front-left
            [-0.55, -0.35, 0],  # rear-right
            [-0.55, 0.35, 0],   # rear-left
        ])

        # Control limits
        self.max_thrust_per_thruster = 500.0
        self.max_gimbal_angle = 0.785  # 45 degrees

        # Design LQR gains for position control (x, y, z and their derivatives)
        self._design_position_lqr()

        # Design LQR gains for attitude control (roll, pitch, yaw and their derivatives)
        self._design_attitude_lqr()

    def _design_position_lqr(self):
        """Design LQR for position control (double integrator per axis)."""
        # State: [x, x_dot] for each axis
        # For a double integrator: x_ddot = u
        A = np.array([[0, 1],
                      [0, 0]])
        B = np.array([[0],
                      [1]])

        # Tune these weights for desired response
        Q = np.diag([10.0, 2.0])  # Position error, velocity error
        R = np.array([[1.0]])      # Control effort

        # Solve continuous-time algebraic Riccati equation
        P = linalg.solve_continuous_are(A, B, Q, R)
        self.K_pos = np.linalg.inv(R) @ B.T @ P

    def _design_attitude_lqr(self):
        """Design LQR for attitude control (double integrator per axis)."""
        A = np.array([[0, 1],
                      [0, 0]])
        B = np.array([[0],
                      [1]])

        # Tighter control on attitude
        Q = np.diag([50.0, 10.0])  # Angle error, angular velocity error
        R = np.array([[0.1]])       # Control effort

        P = linalg.solve_continuous_are(A, B, Q, R)
        self.K_att = np.linalg.inv(R) @ B.T @ P

    def compute_control(self, data, target_pos):
        """
        Compute control inputs to fly to target position.

        Args:
            data: MuJoCo data object
            target_pos: Target position [x, y, z]

        Returns:
            ctrl: Control vector [thrust_fr, thrust_fl, thrust_rr, thrust_rl,
                                  pitch_fr, roll_fr, pitch_fl, roll_fl,
                                  pitch_rr, roll_rr, pitch_rl, roll_rl]
        """
        # Get current state from sensors
        # qpos layout: [x, y, z, qw, qx, qy, qz, joint_angles...]
        pos = data.qpos[0:3].copy()
        quat = data.qpos[3:7].copy()
        vel = data.qvel[0:3].copy()
        ang_vel = data.qvel[3:6].copy()

        euler = quat_to_euler(quat)
        roll, pitch, yaw = euler

        # --- Outer Loop: Position Control ---
        pos_error = target_pos - pos

        # Compute desired accelerations using LQR
        ax_des = -self.K_pos @ np.array([pos_error[0], -vel[0]])
        ay_des = -self.K_pos @ np.array([pos_error[1], -vel[1]])
        az_des = -self.K_pos @ np.array([pos_error[2], -vel[2]])

        ax_des = ax_des[0]
        ay_des = ay_des[0]
        az_des = az_des[0]

        # Add gravity compensation to z
        az_des += self.gravity

        # --- Compute Desired Attitude from Desired Acceleration ---
        # For small angles: ax ≈ g * pitch_des, ay ≈ -g * roll_des
        # Desired pitch to achieve ax (positive pitch = nose down = positive x acceleration)
        pitch_des = np.clip(ax_des / self.gravity, -0.5, 0.5)
        # Desired roll to achieve ay (positive roll = right wing down = negative y acceleration)
        roll_des = np.clip(-ay_des / self.gravity, -0.5, 0.5)
        yaw_des = 0.0  # Keep yaw at zero for now

        # --- Inner Loop: Attitude Control ---
        roll_error = roll_des - roll
        pitch_error = pitch_des - pitch
        yaw_error = yaw_des - yaw

        # Wrap yaw error to [-pi, pi]
        yaw_error = np.arctan2(np.sin(yaw_error), np.cos(yaw_error))

        # Compute desired angular accelerations
        roll_ddot_des = -self.K_att @ np.array([roll_error, -ang_vel[0]])
        pitch_ddot_des = -self.K_att @ np.array([pitch_error, -ang_vel[1]])
        yaw_ddot_des = -self.K_att @ np.array([yaw_error, -ang_vel[2]])

        roll_ddot_des = roll_ddot_des[0]
        pitch_ddot_des = pitch_ddot_des[0]
        yaw_ddot_des = yaw_ddot_des[0]

        # --- Thrust Allocation ---
        # Total thrust needed to achieve desired vertical acceleration
        total_thrust = self.total_mass * az_des / np.cos(roll) / np.cos(pitch)
        total_thrust = np.clip(total_thrust, 0, 4 * self.max_thrust_per_thruster)

        # Base thrust per thruster
        base_thrust = total_thrust / 4

        # Differential thrust for roll and pitch control of the chassis
        # Positive pitch_ddot -> need more thrust in rear, less in front
        # Positive roll_ddot -> need more thrust on left, less on right
        Ix = 2.0  # Approximate moment of inertia around x (roll)
        Iy = 1.0  # Approximate moment of inertia around y (pitch)

        pitch_moment = Iy * pitch_ddot_des
        roll_moment = Ix * roll_ddot_des

        # Moment arms
        arm_x = 0.55  # Distance from center to front/rear thrusters
        arm_y = 0.35  # Distance from center to left/right thrusters

        # Thrust differentials
        delta_thrust_pitch = pitch_moment / (2 * arm_x)  # positive = more rear thrust
        delta_thrust_roll = roll_moment / (2 * arm_y)    # positive = more left thrust

        # Individual thrust values
        thrust_fr = base_thrust - delta_thrust_pitch / 2 - delta_thrust_roll / 2
        thrust_fl = base_thrust - delta_thrust_pitch / 2 + delta_thrust_roll / 2
        thrust_rr = base_thrust + delta_thrust_pitch / 2 - delta_thrust_roll / 2
        thrust_rl = base_thrust + delta_thrust_pitch / 2 + delta_thrust_roll / 2

        # Clip thrust values
        thrusts = np.array([thrust_fr, thrust_fl, thrust_rr, thrust_rl])
        thrusts = np.clip(thrusts, 0, self.max_thrust_per_thruster)

        # --- Gimbal Angles ---
        # Tilt thrusters to provide horizontal force for position control
        # All thrusters tilt together to maintain balance
        # Positive pitch gimbal -> thrust vector tilts forward -> +x force
        # Positive roll gimbal -> thrust vector tilts right -> -y force

        gimbal_pitch = np.clip(-pitch_des, -self.max_gimbal_angle, self.max_gimbal_angle)
        gimbal_roll = np.clip(-roll_des, -self.max_gimbal_angle, self.max_gimbal_angle)

        # All gimbals get the same command (could be differentiated for yaw control)
        gimbal_angles = np.array([
            gimbal_pitch, gimbal_roll,  # front-right
            gimbal_pitch, gimbal_roll,  # front-left
            gimbal_pitch, gimbal_roll,  # rear-right
            gimbal_pitch, gimbal_roll,  # rear-left
        ])

        # Combine into control vector
        ctrl = np.concatenate([thrusts, gimbal_angles])

        return ctrl
