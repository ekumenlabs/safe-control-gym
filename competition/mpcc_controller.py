# Copyright (c) 2022 Ekumen, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import acados_template as ad

import casadi as cs
import math as m
import numpy as np

from acados_template import AcadosModel

from typing import Dict, Any

from safe_control_gym.math_and_models.transformations import csRotXYZ


class MPCCController():

    def __init__(self,
                 dt,
                 mpcc_horizon_len=10,
                 mass=0.027,
                 ixx=1.4e-5,
                 iyy=1.4e-5,
                 izz=2.17e-5,
                 arm_len=0.0397,
                 gravity=9.8,
                 kf=3.16e-10,
                 km=7.94e-12,
                 ):

        # control system sampling interval
        self.PARAM_DT = dt

        #
        # nominal drone parameters can be found in the URDF file

        self.PARAM_DRONE_MASS = mass
        self.PARAM_J_MATRIX_DIA = [ixx, iyy, izz]
        self.PARAM_ARM_LEN = arm_len
        self.PARAM_GRAVITY = gravity
        self.PARAM_KF = kf
        self.PARAM_KM = km

        #
        # MPCC optimizer parameters

        self.MPCC_HORIZON_LEN = mpcc_horizon_len

        def deg_to_rad(d): return d * 2 * m.pi / 180.0

        self.MPCC_CONTOUR_ERROR_GAUSSIAN_SIGMA = 2.5 # m

        self.MPCC_PROGRESS_INCENTIVE_PEAK_WEIGHT = 1.5
        self.MPCC_LAG_ERROR_WEIGHT = 45.0
        self.MPCC_CONTOUR_ERROR_WEIGHT_MAX = 65.0
        self.MPCC_CONTOUR_ERROR_WEIGHT_MIN = 25.0
        self.MPCC_BODY_ORIENTATION_RATE_WEIGHT_DIAG = 0.2
        self.MPCC_CONTOUR_RATE_CHANGE_WEIGHT = 0.08
        self.MPCC_RATE_BOUNDED_THRUST_WEIGHT = 0.001

        self.MPCC_CONSTRAINT_RATE_BOUNDED_THRUST_MAX_MODULE = 3.0  # N
        self.MPCC_CONSTRAINT_RATE_BOUNDED_THRUST_MIN_MODULE = 0.0  # N
        self.MPCC_CONSTRAINT_DELTA_RATE_BOUNDED_THRUST_MAX_MODULE = 2. # N / s
        self.MPCC_CONSTRAINT_DELTA_RATE_BOUNDED_THRUST_MIN_MODULE = 0. # N / s
        self.MPCC_CONSTRAINT_MAX_INCLINATION_MODULE = deg_to_rad(60)  # deg
        self.MPCC_CONSTRAINT_BODY_ORIENTATION_RATE_MAX_MODULE = \
            deg_to_rad(450)  # deg/sec
        self.MPCC_CONSTRAINT_CONTOUR_PARAM_VEL_MAX = 0.8  # m/s
        self.MPCC_CONSTRAINT_CONTOUR_PARAM_ACC_MAX_MODULE = 5.  # m/(s*s)

        # Initialize the solver state
        self.solver = None
        self.reset()

    def _build_discrete_system_model(self):
        """
        Creates symbolic (CasADi) models for dynamics of the quadrotor system dynamics, observation and cost function terms.

        Based on the model described in  Ch. 2 of Luis, Carlos, and Jérôme Le Ny.
        "Design of a trajectory tracking controller for a nanoquadcopter." arXiv preprint arXiv:1608.05786 (2016) and
        its implementation in the Safe Gym simulator.

        Refactored from model system in Quadrotor.py
        """

        # model parameters
        m, g, l = self.PARAM_DRONE_MASS, self.PARAM_GRAVITY, self.PARAM_ARM_LEN

        #
        # Dynamic model parameters

        # Inertia matrix main diagonal elements
        Ixx, Iyy, Izz = self.PARAM_J_MATRIX_DIA

        # Assuming the quadrotor is symmetric around the axes, as per [1], the
        # simplified inertia matrix turns out to be pure diagonal
        J = cs.blockcat([[Ixx, 0.0, 0.0],
                         [0.0, Iyy, 0.0],
                         [0.0, 0.0, Izz]])

        # Precalculated inverse inertial matrix
        Jinv = cs.blockcat([[1.0/Ixx, 0.0, 0.0],
                            [0.0, 1.0/Iyy, 0.0],
                            [0.0, 0.0, 1.0/Izz]])

        # thrust to squared rotor speed const (TODO: this needs to be confirmed)
        gamma = self.PARAM_KM/self.PARAM_KF

        #
        # Dynamic model states

        # quadrotor pose in world frame
        x = cs.MX.sym('x')
        z = cs.MX.sym('z')
        y = cs.MX.sym('y')

        # quadrotor speed in world frame
        x_dot = cs.MX.sym('x_dot')
        z_dot = cs.MX.sym('z_dot')
        y_dot = cs.MX.sym('y_dot')

        # quadrotor orientation in world frame
        phi = cs.MX.sym('phi')      # Roll
        theta = cs.MX.sym('theta')  # Pitch
        psi = cs.MX.sym('psi')      # Yaw

        # quadrotor orientation change rate in body frame
        p = cs.MX.sym('p')  # Body frame roll rate
        q = cs.MX.sym('q')  # body frame pitch rate
        r = cs.MX.sym('r')  # Body frame yaw rate
        #
        # Dynamic model inputs

        # individual thrusts
        f1 = cs.MX.sym('f1')
        f2 = cs.MX.sym('f2')
        f3 = cs.MX.sym('f3')
        f4 = cs.MX.sym('f4')

        #
        # Auxiliar helper expresions

        # rotation matrix transforming a vector in the body frame to the world frame.
        # PyBullet Euler angles use the SDFormat for rotation matrices.
        Rob = csRotXYZ(phi, theta, psi)

        # aggregated thrust, in body frame
        fT = cs.vertcat(0, 0, f1 + f2 + f3 + f4)

        # gravity force, in world frame
        gT = cs.vertcat(0, 0, g)

        Mb = cs.vertcat(
            l/cs.sqrt(2.0)*(f1+f2-f3-f4),
            l/cs.sqrt(2.0)*(-f1+f2+f3-f4),
            gamma*(f1-f2+f3-f4)
        )

        #
        # System dynamics function.

        # We are using the velocity of the base wrt to the world frame expressed in the world frame.
        # Note that the reference expresses this in the body frame.
        oVdot_cg_o = Rob @ fT / m - gT

        # Velocity rate of change
        pos_ddot = oVdot_cg_o

        # Position rate of change
        pos_dot = cs.vertcat(x_dot, y_dot, z_dot)

        # Orientation rate of change
        ang_dot = cs.blockcat([[1, cs.sin(phi)*cs.tan(theta), cs.cos(phi)*cs.tan(theta)],
                               [0, cs.cos(phi), -cs.sin(phi)],
                               [0, cs.sin(phi)/cs.cos(theta), cs.cos(phi)/cs.cos(theta)]]) @ cs.vertcat(p, q, r)

        # Orientation rate rate of change
        rate_dot = Jinv @ (Mb - (cs.skew(cs.vertcat(p, q, r))
                           @ J @ cs.vertcat(p, q, r)))

        #
        # Aggregated state and input vectors

        # state and input vectors
        X = cs.vertcat(x, y, z, x_dot, y_dot, z_dot, phi, theta, psi, p, q, r)
        U = cs.vertcat(f1, f2, f3, f4)

        # dynamic function f(X, U)
        X_dot = cs.vertcat(pos_dot, pos_ddot, ang_dot, rate_dot)

        continuous_system_model_function = cs.Function(
            'continuous_system_model_function', [X, U], [X_dot], ['X', 'U'], ['X_dot'])

        #
        # System discretization

        X_dot_discretization = self._runge_kutta_discretization(
            continuous_system_model_function, X.size(1), U.size(1), self.PARAM_DT)

        X_next = X_dot_discretization(X, U)

        #
        # Generate system model function

        discrete_system_model_function = cs.Function(
            'discrete_system_model_function', [X, U], [X_next], ['X', 'U'], ['X_next'])

        return discrete_system_model_function.expand()

    def _runge_kutta_discretization(self, f, n, m, dt):
        """
        Runge-Kutta discretization of a continuous DAE
        Taken from mpc_utils.py
        """
        X = cs.MX.sym('X', n)
        U = cs.MX.sym('U', m)

        # Runge-Kutta 4 integration
        k1 = f(X,         U)
        k2 = f(X+dt/2*k1, U)
        k3 = f(X+dt/2*k2, U)
        k4 = f(X+dt*k3,   U)
        X_next = X + dt/6*(k1+2*k2+2*k3+k4)

        # Runge-Kutta 2 integration
        # k1 = f(X, U)
        # k2 = f(X + k1*dt, U)
        # X_next = X + dt*(k1+k2)/2

        system_model_disc = cs.Function(
            'system_model_disc', [X, U], [X_next], ['X', 'U'], ['X_next'])

        return system_model_disc.expand()

    def _build_extended_mpcc_system(self):
        """
        Build a symbolic function system for the extended states used by MPCC
        """
        dt = self.PARAM_DT

        #
        # Dynamic model states

        # rate bounded actuator thrusts
        rbf1 = cs.MX.sym('rbf1')
        rbf2 = cs.MX.sym('rbf2')
        rbf3 = cs.MX.sym('rbf3')
        rbf4 = cs.MX.sym('rbf4')

        contour_param_pos = cs.MX.sym(
            'contour_param_pos')  # progress along the curve
        # progress rate along the curve
        contour_param_vel = cs.MX.sym('contour_param_vel')

        #
        # Dynamic model inputs

        # individual thrusts deltas
        delta_rbt_1 = cs.MX.sym('delta_rbt_1')
        delta_rbt_2 = cs.MX.sym('delta_rbt_2')
        delta_rbt_3 = cs.MX.sym('delta_rbt_3')
        delta_rbt_4 = cs.MX.sym('delta_rbt_4')

        # progress acceleration
        contour_param_acc = cs.MX.sym('contour_param_acc')

        #
        # Auxiliar helper expresions

        # rate bounded thrusts vector
        rate_bounded_thrust = cs.vertcat(rbf1, rbf2, rbf3, rbf4)

        # rate bounded delta thrusts vector
        delta_rate_bounded_thrust = cs.vertcat(
            delta_rbt_1, delta_rbt_2, delta_rbt_3, delta_rbt_4)

        #
        # System dynamics function.

        rate_bounded_thrust_next = rate_bounded_thrust + delta_rate_bounded_thrust * dt
        contour_pos_next = contour_param_pos + contour_param_vel * dt
        contour_vel_next = contour_param_vel + contour_param_acc * dt

        #
        # Aggregated state and input vectors

        # dynamic function f(X, U)
        X_next = cs.vertcat(rate_bounded_thrust_next,
                            contour_pos_next, contour_vel_next)

        # state and input vectors
        X = cs.vertcat(rate_bounded_thrust,
                       contour_param_pos, contour_param_vel)
        U = cs.vertcat(delta_rate_bounded_thrust, contour_param_acc)

        #
        # Build reusable function

        extended_mpcc_function = cs.Function(
            'X_next_function', [X, U], [X_next, rate_bounded_thrust], ['X', 'U'], ['X_next', 'rate_bounded_thrust'])

        return extended_mpcc_function.expand()


    def _build_mpcc_cost_term_function(self):
        """
        Build a symbolic cost function term function.
        """
        lag_error_weight = self.MPCC_LAG_ERROR_WEIGHT
        body_rate_weight = self.MPCC_BODY_ORIENTATION_RATE_WEIGHT_DIAG
        countour_rate_change_weight = self.MPCC_CONTOUR_RATE_CHANGE_WEIGHT
        rbt_weight = self.MPCC_RATE_BOUNDED_THRUST_WEIGHT
        progress_incentive_weight = self.MPCC_PROGRESS_INCENTIVE_PEAK_WEIGHT

        #
        # Inputs

        # lag error
        lag_error = cs.MX.sym('lag_error')

        # contour error
        contour_error = cs.MX.sym('contour_error', 3)

        # contour error weight
        contour_error_weight = cs.MX.sym('contour_error_weight')

        # orientation rate of change
        body_orientation_rate = cs.MX.sym('body_orientation_rate', 3)

        # progress rate along the curve
        contour_param_vel = cs.MX.sym('contour_param_vel')

        # progress acceleration
        contour_param_acc = cs.MX.sym('contour_param_acc')

        # rate bounded actuators
        delta_rate_bounded_thrust = cs.MX.sym('delta_rate_bounded_thrust', 4)

        # Current location of the drone
        current_vehicle_pos = cs.MX.sym('current_vehicle_pos', 3)

        #
        # Cost term expresion

        lag_error_cost = lag_error_weight * (lag_error * lag_error)

        contour_error_cost = contour_error_weight * \
            cs.dot(contour_error, contour_error)

        orientation_rate_cost = body_rate_weight * \
            cs.dot(body_orientation_rate, body_orientation_rate)

        contour_rate_change_cost = countour_rate_change_weight * \
            (contour_param_acc * contour_param_acc)

        rate_bounded_thrust_change_cost = rbt_weight * \
            cs.dot(delta_rate_bounded_thrust, delta_rate_bounded_thrust)

        progress_incentive = progress_incentive_weight * contour_param_vel

        J_mpcc_k = lag_error_cost + contour_error_cost + orientation_rate_cost + \
            contour_rate_change_cost + rate_bounded_thrust_change_cost - progress_incentive

        #
        # Build reusable function

        cost_term_function = cs.Function(
            'cost_term_function',
            [lag_error, contour_error, contour_error_weight, body_orientation_rate,
                contour_param_acc, delta_rate_bounded_thrust, contour_param_vel,
                current_vehicle_pos],
            [J_mpcc_k],
            ['lag_error', 'contour_error', 'contour_error_weight', 'body_orientation_rate',
                'contour_param_acc', 'delta_rate_bounded_thrust', 'contour_param_vel',
                'current_vehicle_pos'],
            ['J_mpcc_k'])

        return cost_term_function.expand()

    def _build_mpcc_contour_error_functions(self):
        """
        Build a symbolic error function pair to calculate lag and contour error
        """
        #
        # inputs

        # current coordinates of the vehicle
        position = cs.MX.sym('position', 3)

        # coordinates of the contour curve at theta
        contour_curve = cs.MX.sym('contour_curve', 3)

        # tangent of the contour curve at theta
        contour_tangent = cs.MX.sym('contour_tangent', 3)

        #
        # build reusable function

        error = position - contour_curve

        lag_error = cs.dot(contour_tangent, error)
        contour_error = error - lag_error * contour_tangent

        error_function = cs.Function(
            'lag_error_function',
            [position, contour_curve, contour_tangent],
            [lag_error, contour_error],
            ['position', 'contour_curve', 'contour_tangent'],
            ['lag_error', 'contour_error']
        )

        return error_function.expand()

    def _build_mpcc_build_contour_interpolant_functions(self, waypoints: list, contour_poses: list):
        """
        Build interpolant spline curve through waypoints.
        Returns a symbolic function pair used to coordiantes and tangents along the curve.
        """
        #
        # inputs

        theta = cs.MX.sym('theta')

        #
        # contour curve interpolation

        spline_support = np.linspace(
            np.min(contour_poses), np.max(contour_poses), len(waypoints))

        waypoints_array = np.array(waypoints)

        waypoints_x = waypoints_array[:, 0]
        waypoints_y = waypoints_array[:, 1]
        waypoints_z = waypoints_array[:, 2]

        contour_f_x = cs.interpolant(
            'contour_f_x', 'bspline', [spline_support], waypoints_x)
        contour_f_y = cs.interpolant(
            'contour_f_y', 'bspline', [spline_support], waypoints_y)
        contour_f_z = cs.interpolant(
            'contour_f_z', 'bspline', [spline_support], waypoints_z)

        contour_curve = cs.vertcat(contour_f_x(
            theta), contour_f_y(theta), contour_f_z(theta))

        contour_curve_derivative = cs.jacobian(contour_curve, theta)

        # TODO: I'd rather not do this assumption but dividing by the norm
        # causes Casadi to trip when differentiating the expresion around trivial
        # initialization values (all zeros).
        # Since the curve is an arc-len parameterized curve, then contour_curve_derivative
        # should already approximate contour_tangent

        # contour_tangent = contour_curve_derivative / \
        #     cs.norm_2(contour_curve_derivative)

        contour_tangent = contour_curve_derivative  # READ ABOVE

        #
        # build reusable function

        contour_curve_function = cs.Function(
            'contour_curve_function',
            [theta],
            [contour_curve, contour_tangent],
            ['theta'],
            ['contour_curve', 'contour_tangent'])

        # This function cannot be expanded to SX
        return contour_curve_function

    def _build_contour_error_weight_function(self, gate_list, obstacle_list):
        """
        Build a function to compute the contour error weight based on the desired position
        of the platform. When we are close to a waypoint, we give more importance to the contour
        error, and everywhere else to the progress.
        """
        contour_curve = cs.MX.sym('contour_curve', 3)

        gaussian_amplitude = self.MPCC_CONTOUR_ERROR_WEIGHT_MAX - \
            self.MPCC_CONTOUR_ERROR_WEIGHT_MIN

        contour_error_weight = self.MPCC_CONTOUR_ERROR_WEIGHT_MIN

        # distance to gates is calculated in 3D
        for gate_pos in gate_list:
            distance_to_kernel = contour_curve - gate_pos
            contour_error_weight += gaussian_amplitude * \
                cs.exp(-1.0 / (2 * self.MPCC_CONTOUR_ERROR_GAUSSIAN_SIGMA**2)
                       * cs.dot(distance_to_kernel, distance_to_kernel))

        # distance to obstacles is measured in 2D
        for obstacle_pos in obstacle_list:
            distance_to_kernel = (contour_curve - obstacle_pos)
            distance_to_kernel_2d = distance_to_kernel[0:2]
            contour_error_weight += gaussian_amplitude * \
                cs.exp(-1.0 / (2 * self.MPCC_CONTOUR_ERROR_GAUSSIAN_SIGMA**2)
                       * cs.dot(distance_to_kernel_2d, distance_to_kernel_2d))

        contour_error_weight_function = cs.Function(
            'contour_error_weight',
            [contour_curve], [contour_error_weight],
            ['contour_curve'], ['contour_error_weight'])

        return contour_error_weight_function.expand()

    def _build_full_mpcc_target_model(self):
        """
        Combine the drone system model and the extended mpcc equation system in a single combined system used by the optimizer
        """
        discrete_system_model = self._build_discrete_system_model()

        extended_system_model = self._build_extended_mpcc_system()

        #
        # Optimizer inputs

        state_size_model = discrete_system_model.size1_in("X")
        state_size_mpcc_ext = extended_system_model.size1_in("X")

        input_size_model = discrete_system_model.size1_in("U")
        input_size_mpcc_ext = extended_system_model.size1_in("U")

        # state is the combined state of both
        X = cs.MX.sym('X', state_size_model + state_size_mpcc_ext)

        # notice that the model input is connected to a mpcc ext system output, and is not exposed outwards
        U = cs.MX.sym('U', input_size_mpcc_ext)

        #
        # Plumbing work

        model_x, mpcc_x = cs.vertsplit(X, [0, state_size_model, X.size(1)])

        mpcc_xnext = extended_system_model(
            X=mpcc_x, U=U)["X_next"]

        mpcc_rate_bounded_thrust = extended_system_model(
            X=mpcc_x, U=U
        )["rate_bounded_thrust"]

        model_xnext = discrete_system_model(
            X=model_x, U=mpcc_rate_bounded_thrust
        )['X_next']

        X_next = cs.vertcat(model_xnext, mpcc_xnext)

        full_mpcc_system_function = cs.Function(
            'full_mpcc_system_function', [X, U], [X_next], ['X', 'U'], ['X_next']).expand()

        X_dot = cs.MX.sym('X_dot', X.size(1))

        model = ad.AcadosModel()
        model.disc_dyn_expr = full_mpcc_system_function(X, U)
        model.x = X
        model.u = U
        model.xdot = X_dot
        model.z = []
        model.p = []
        model.name = 'crazyflie_contouring_control_model'
        return model

    def _split_actions(self, x_next):
        """
        Extract pos, vel, rpy and body rates triads from the full system state vector
        """
        pos = np.array(x_next[0:3])
        vel = np.array(x_next[3:6])
        rpy = np.array(x_next[6:9])
        body_rates = np.array(x_next[9:12])
        contour_param_pos = x_next[16]
        return pos, vel, rpy, body_rates, contour_param_pos

    def setup(self,
              waypoints,
              contour_poses,
              landmarks):
        """
        Setup optimizer ie. cost function and constraints.
        """
        self.contour_poses = contour_poses

        # gate landmarks are listed multiple times, and that distorts cost map
        # No time to understand and fix at source, so...
        # Update: see https://github.com/ekumenlabs/RnD/pull/150#discussion_r991218783
        groups = [(position, asoc_landmarks) for position, asoc_landmarks in zip(
            waypoints, landmarks) if asoc_landmarks]
        landmarks_with_no_duplicates = []
        for position, asoc_landmarks in groups:
            matcher = {}
            for landmark in asoc_landmarks:
                matcher[landmark.kind] = landmark.location
            for kind, location in matcher.items():
                landmarks_with_no_duplicates.append((location, position, kind))

        gates = [position for _, position,
                 kind in landmarks_with_no_duplicates if kind == 'waypoint']
        obstacles = [position for _, position,
                     kind in landmarks_with_no_duplicates if kind == 'obstacle']

        self.interpolant_functions = self._build_mpcc_build_contour_interpolant_functions(
            waypoints, contour_poses)
        contour_error_functions = self._build_mpcc_contour_error_functions()
        contour_error_weight_functions = self._build_contour_error_weight_function(
            gates, obstacles)
        mpcc_cost_term_function = self._build_mpcc_cost_term_function()

        # Define optimizer and variables.
        self.ocp = ad.AcadosOcp()
        self.ocp.model = self._build_full_mpcc_target_model()
        full_system_state_vector_len = self.ocp.model.x.size(1)
        full_system_input_vector_len = self.ocp.model.u.size(1)
        self.ocp.dims.N = self.MPCC_HORIZON_LEN

        #
        # Setup MPCC cost function
        position, _, _, body_orientation_rate, rate_bounded_thrust, contour_param_pos, contour_param_vel = cs.vertsplit(
            self.ocp.model.x, [0, 3, 6, 9, 12, 16, 17, full_system_state_vector_len])
        delta_rate_bounded_thrust, contour_param_acc = cs.vertsplit(
            self.ocp.model.u, [0, 4, full_system_input_vector_len])

        interpolant_functions_sym = self.interpolant_functions(
            theta=contour_param_pos,
        )
        contour_curve = interpolant_functions_sym["contour_curve"]
        contour_tangent = interpolant_functions_sym["contour_tangent"]
        contour_error_functions_sym = contour_error_functions(
            position=position,
            contour_curve=contour_curve,
            contour_tangent=contour_tangent
        )
        lag_error = contour_error_functions_sym["lag_error"]
        contour_error = contour_error_functions_sym["contour_error"]
        contour_error_weight_sym = contour_error_weight_functions(
            contour_curve=contour_curve,
        )
        contour_error_weight = contour_error_weight_sym["contour_error_weight"]

        self.ocp.cost.cost_type = "EXTERNAL"
        self.ocp.model.cost_expr_ext_cost = mpcc_cost_term_function(
            lag_error=lag_error,
            contour_error=contour_error,
            contour_error_weight=contour_error_weight,
            body_orientation_rate=body_orientation_rate,
            contour_param_acc=contour_param_acc,
            delta_rate_bounded_thrust=delta_rate_bounded_thrust,
            contour_param_vel=contour_param_vel,
            current_vehicle_pos=position,
        )["J_mpcc_k"]

        # Note: the terminal cost term in MPCC cost function
        # is the same as of intermediate shooting nodes

        #
        # Constraints

        # The first state in the sequence is constrained at the origin.
        # Initial states for subsequent horizons are constrained via
        # linear state constraints.

        # Initial state constraint
        self.ocp.constraints.x0 = np.zeros(full_system_state_vector_len)
        self.ocp.constraints.Jbx_0 = np.eye(full_system_state_vector_len)
        self.ocp.constraints.lbx_0 = np.zeros(full_system_state_vector_len)
        self.ocp.constraints.ubx_0 = np.zeros(full_system_state_vector_len)

        # TODO: some of these might better the constrained individually and not as
        # the module of the vector (e.g. rate_bounded_thrust)

        # Linear constraints on state, namely:
        # - |roll, pitch| <= inclination_max
        # - |p, q, r| <= w_max
        # - 0 <= contour_param_vel < contour_param_vel_max
        ilx = np.array([6, 7, 9, 10, 11, 17])
        C = np.zeros((
            full_system_state_vector_len,
            full_system_state_vector_len))
        C[ilx, ilx] = 1
        C = C[ilx, :]
        self.ocp.constraints.C = C
        self.ocp.constraints.C_e = C
        self.ocp.constraints.D = np.zeros_like(C)
        self.ocp.constraints.lg = np.array([
            -self.MPCC_CONSTRAINT_MAX_INCLINATION_MODULE,
            -self.MPCC_CONSTRAINT_MAX_INCLINATION_MODULE,
            -self.MPCC_CONSTRAINT_BODY_ORIENTATION_RATE_MAX_MODULE,
            -self.MPCC_CONSTRAINT_BODY_ORIENTATION_RATE_MAX_MODULE,
            -self.MPCC_CONSTRAINT_BODY_ORIENTATION_RATE_MAX_MODULE,
            0.
        ])
        self.ocp.constraints.lg_e = self.ocp.constraints.lg
        self.ocp.constraints.ug = np.array([
            self.MPCC_CONSTRAINT_MAX_INCLINATION_MODULE,
            self.MPCC_CONSTRAINT_MAX_INCLINATION_MODULE,
            self.MPCC_CONSTRAINT_BODY_ORIENTATION_RATE_MAX_MODULE,
            self.MPCC_CONSTRAINT_BODY_ORIENTATION_RATE_MAX_MODULE,
            self.MPCC_CONSTRAINT_BODY_ORIENTATION_RATE_MAX_MODULE,
            self.MPCC_CONSTRAINT_CONTOUR_PARAM_VEL_MAX,
        ])
        self.ocp.constraints.ug_e = self.ocp.constraints.ug

        # Linear constraints on input, namely:
        # - |contour_curve_acc| <= contour_curve_acc_max
        Jbu = np.zeros((1, full_system_input_vector_len))
        Jbu[0, -1] = 1
        self.ocp.constraints.Jbu = Jbu

        self.ocp.constraints.lbu = np.array([
            -self.MPCC_CONSTRAINT_CONTOUR_PARAM_ACC_MAX_MODULE])
        self.ocp.constraints.ubu = np.array([
            self.MPCC_CONSTRAINT_CONTOUR_PARAM_ACC_MAX_MODULE])

        # Nonlinear constraints on input and state, namely:
        # - |f| <= f_max
        # - |df| <= df_max
        self.ocp.model.con_h_expr = cs.vertcat(
            cs.dot(rate_bounded_thrust, rate_bounded_thrust),
        #    cs.dot(delta_rate_bounded_thrust, delta_rate_bounded_thrust),
        )
        self.ocp.constraints.lh = np.array([
            self.MPCC_CONSTRAINT_RATE_BOUNDED_THRUST_MIN_MODULE**2,
        #    self.MPCC_CONSTRAINT_DELTA_RATE_BOUNDED_THRUST_MIN_MODULE**2,
        ])
        self.ocp.constraints.uh = np.array([
            self.MPCC_CONSTRAINT_RATE_BOUNDED_THRUST_MAX_MODULE**2,
        #    self.MPCC_CONSTRAINT_DELTA_RATE_BOUNDED_THRUST_MAX_MODULE**2,
        ])
        self.ocp.model.con_h_expr_e = cs.dot(rate_bounded_thrust, rate_bounded_thrust)
        self.ocp.constraints.lh_e = np.array([
            self.MPCC_CONSTRAINT_RATE_BOUNDED_THRUST_MIN_MODULE**2])
        self.ocp.constraints.uh_e = np.array([
            self.MPCC_CONSTRAINT_RATE_BOUNDED_THRUST_MAX_MODULE**2])

        # Complete solver initializtion
        self.ocp.solver_options.tf = self.MPCC_HORIZON_LEN * self.PARAM_DT
        self.ocp.solver_options.qp_solver = "FULL_CONDENSING_QPOASES"
        self.ocp.solver_options.qp_solver_cond_N = self.MPCC_HORIZON_LEN
        self.ocp.solver_options.qp_solver_warm_start = 1
        self.ocp.solver_options.qp_tol = 1e-2
        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.tol = 1e-2
        self.ocp.solver_options.hessian_approx = "EXACT"
        #self.ocp.solver_options.nlp_solver_ext_qp_res = 1
        #self.ocp.solver_options.regularize_method = "PROJECT"
        self.ocp.solver_options.levenberg_marquardt = 3.
        self.ocp.solver_options.integrator_type = "DISCRETE"

        self.solver = ad.AcadosOcpSolver(self.ocp, json_file="mpcc.json")

        self.warm_started = False

    def solve(self,
              current_pos: np.ndarray,
              current_vel: np.ndarray,
              current_rpy: np.ndarray,
              current_pqr: np.ndarray):
        # Initialize solver, if we have any information

        # TODO: we need to initialize the optimizer with a state guestimation
        # If we have no other information probably we can use the same contour
        # curve to initialize x and v, contour_curve_pos and contour_curve_vel
        # and assume the rest of the values remain unchanged for a good enough
        # initial guess.
        if self.warm_started:
            # Use the previous solution to initialize the virtual initial values
            self.virtual_init_state = self.solver.get(1, "x")
        else:
            # pre-set thrust for buoyancy
            self.virtual_init_state[12:16] = \
               self.PARAM_DRONE_MASS * self.PARAM_GRAVITY / 4.0

        # build the initialization vector from a mixture of observable states
        # and the previous iteration's state estimations
        self.virtual_init_state[0:3] = current_pos
        self.virtual_init_state[3:6] = current_vel
        self.virtual_init_state[6:9] = current_rpy
        self.virtual_init_state[9:12] = current_pqr

        # set the value of the initialization vector parameter
        self.solver.constraints_set(0, "lbx", self.virtual_init_state)
        self.solver.constraints_set(0, "ubx", self.virtual_init_state)

        # Magic here
        status = self.solver.solve()

        if status != 0:
            print("The solver did not return a solution, so we are making up our own!")
            self.solver.print_statistics()
            self.solver.reset()
            self.warm_started = False
            fallback_velocity = 0.2
            self.virtual_init_state[16] += self.PARAM_DT * fallback_velocity
            interpolant_functions_sym = self.interpolant_functions(
                theta=self.virtual_init_state[16],
            )
            contour_param_pos = self.virtual_init_state[16]
            pos = np.array(interpolant_functions_sym["contour_curve"].T)[0, :]
            vel = fallback_velocity * \
                np.array(interpolant_functions_sym["contour_tangent"].T)[0, :]
            rpy = body_rates = np.zeros(3)
        else:
            # recover actionable values
            pos, vel, rpy, body_rates, contour_param_pos = \
                self._split_actions(self.solver.get(1, "x"))
            self.warm_started = True

        contour_param_pos = self.contour_poses[-1] if (
            self.contour_poses[-1] < contour_param_pos) else contour_param_pos
        current_carrot_pos = np.squeeze(np.array(self.interpolant_functions(
            theta=contour_param_pos)["contour_curve"]))

        #
        # START OF DEBUG CODE
        #
        vector_to_be_logged = current_carrot_pos
        #
        # END OF DEBUG CODE
        #
        print(self.solver.get(1, "x"))
        return pos, vel, rpy, body_rates, current_carrot_pos, vector_to_be_logged

    def reset(self):
        self.warm_started = False
        if self.solver is not None:
            self.virtual_init_state = np.zeros(
                self.solver.acados_ocp.model.x.size(1))


def generate_mpcc_controller():
    from planning import plan_time_optimal_trajectory_through_gates, State, Limits, Cylinder, to_pose

    START_POSE = [-0.9, -2.9, 0.03]
    TAKE_OFF_HEIGHT = 0.4
    EVENLY_SPACED_SEGMENTS = 90
    ARC_PARAMETRIZATION_TOLERANCE = 1e-4
    GOAL_POSE = [-0.5, 2.9, 0.75]

    GATE_DIMENSIONS = {
        "tall": {
            "shape": "square",
            "height": 1.,
            "edge": 0.45
        },
        "low": {
            "shape": "square",
            "height": 0.525,
            "edge": 0.45
        }
    }
    GATE_POSES = [
       [0.5, -2.5, 0, 0, 0, -1.57, 0],
       [2, -1.5, 0, 0, 0, 0, 1],
       [0, 0.2, 0, 0, 0, 1.57, 1],
       [-0.5, 1.5, 0, 0, 0, 0, 0]]
    OBSTACLE_DIMENSIONS = {
        "shape": "cylinder",
        "height": 1.05,
        "radius": 0.05
    }
    OBSTACLE_POSES = [
      [0.5, -1, 0, 0, 0, 0],
      [1.5, 0, 0, 0, 0, 0],
      [-1, 0, 0, 0, 0, 0]
    ]

    # Determine waypoints
    air_start_pos = (START_POSE[0], START_POSE[1], TAKE_OFF_HEIGHT)

    def calculate_gate_pos(x, y, yaw, type):
        tall_gate_height = GATE_DIMENSIONS["tall"]["height"]
        low_gate_height = GATE_DIMENSIONS["low"]["height"]
        height = tall_gate_height if type == 0 else low_gate_height
        return (x, y, height, 0, 0, yaw)

    gates = []
    rotation_offset = 1.57
    for gate in GATE_POSES:
        gates.append(
            calculate_gate_pos(
                x=gate[0],
                y=gate[1],
                    yaw=gate[5] + rotation_offset,
                    type=gate[6]),
            )

    obstacle_height = OBSTACLE_DIMENSIONS["height"]
    obstacle_radius = OBSTACLE_DIMENSIONS["radius"]

    obstacles = []
    for obstacle in OBSTACLE_POSES:
        obstacles.append(
            Cylinder(obstacle[0:3], radius=obstacle_radius, height=obstacle_height))

    print("Calculating best path through gates, may take a few seconds...")

    path = plan_time_optimal_trajectory_through_gates(
        initial_state=State(
            position=np.array(air_start_pos),
            velocity=np.zeros(3)),
        final_state=State(
            position=np.array(GOAL_POSE),
            velocity=np.zeros(3)),
        gate_poses=list(map(to_pose, gates)),
        acceleration_limits=Limits(
            lower=-1 * np.ones(3),
            upper=1 * np.ones(3),
        ),
        velocity_limits=Limits(
            lower=np.array([0.05, -np.pi/6, -np.pi/6]),
            upper=np.array([2.00, np.pi/6, np.pi/6]),
        ),
        num_cone_samples=3,
        obstacles=obstacles,
    )

    waypoint_pos = []
    waypoint_arg = []
    waypoint_marks = []
    for length, position, landmarks in path.evenly_spaced_points(
        EVENLY_SPACED_SEGMENTS, ARC_PARAMETRIZATION_TOLERANCE
    ):
        waypoint_pos.append(position)
        waypoint_arg.append(length)
        waypoint_marks.append(landmarks)

    waypoint_arg = np.array(waypoint_arg)
    waypoint_pos = np.array(waypoint_pos)

    controller = MPCCController(
        dt=1/30,
        mass=0.027,
        ixx=1.4e-5,
        iyy=1.4e-5,
        izz=2.17e-5,
        mpcc_horizon_len=20
    )

    controller.setup(
        waypoints=waypoint_pos,
        contour_poses=waypoint_arg,
        landmarks=waypoint_marks,
    )


if __name__ == '__main__':
    generate_mpcc_controller()

    # uut = MPCCController(0.1)
    # uut.reset()

    # print(uut._build_discrete_system_model())
    # print(uut._build_extended_mpcc_system())
    # print(uut._build_mpcc_cost_term_function())
    # print(uut._build_full_mpcc_target_system())

    # waypoints = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (1, 2, 3)]
    # contour_poses = [-1, 0, 1, 2]

    # print(uut._build_mpcc_build_contour_interpolant_functions(
    #     waypoints, contour_poses
    # ))

    # obs = [0] * 12

    # uut.setup(waypoints, contour_poses)

    # print(uut.solve(obs))
