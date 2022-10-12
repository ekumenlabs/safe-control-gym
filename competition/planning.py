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

import collections
from abc import ABC, abstractmethod
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation

from trajectory import ConstantAccelerationTrajectory
from trajectory import PiecewiseTrajectory
from trajectory import Trajectory


def spherical2cartesian(vector):
    vector = np.asarray(vector)
    if vector.ndim == 1:
        r, theta, phi = vector
        return np.array([
            r * np.cos(theta),
            -r * np.sin(theta) * np.sin(phi),
            r * np.sin(theta) * np.cos(phi)
        ])
    r, theta, phi = vector.T
    return np.column_stack((
        r * np.cos(theta),
        -r * np.sin(theta) * np.sin(phi),
        r * np.sin(theta) * np.cos(phi)
    ))

def cartesian2spherical(vector):
    x, y, z = np.asarray(vector).T
    r = np.sqrt(x**2 + y**2 + z**2)
    if r > 0:
        theta = np.arccos(z / r)
        if x > 0:
            phi = np.arctan(y / x)
        elif x < 0 and y >= 0:
            phi = np.arctan(y / x) + np.pi
        elif x < 0 and y < 0:
            phi = np.arctan(y / x) - np.pi
        else:
            phi = np.sign(y) * np.pi / 2
    else:
        theta = phi = 0
    return np.array([r, theta, phi])


def real_roots(a, b, c):
    r = b * b - 4. * a * c
    if r < 0:
        return np.array([])
    return (-b + np.sqrt(r) * np.array([-1., 1.])) / (2. * a)


def scalar_pmm_bang_bang_control_policy_time(p0, v0, p2, v2, u0, u2):
    if u2 == 0 and u0 == 0:
        if p0 == p2 and v0 == v2:
            return 0
        return np.inf
    if u2 == 0:
        t1 = (v2 - v0) / u0
        p1 = p0 + v0 * t1 + (u0 / 2) * t1**2
        if v2 == 0:
            if p2 != p1:
                t2 = np.inf
            else:
                t2 = 0
        else:
            t2 = (p2 - p1) / v2
        return t1 + t2
    if u0 == 0:
        t2 = (v2 - v0) / u2
        p1 = p2 - v0 * t2 - (u2 / 2) * t2**2
        if v0 == 0:
            if p1 != p0:
                t1 = np.inf
            else:
                t1 = 0
        else:
            t1 = (p1 - p0) / v0
        return t1 + t2

    # Then both u2 and u0 are nonzero
    gamma = u0 / u2
    beta = (v2 - v0) / u2

    a = (u0 / 2) * (1 - gamma)
    b = v0 * (1 - gamma)
    c = beta * (v2 + v0) / 2. + (p0 - p2)

    t1 = real_roots(a, b, c)
    if t1.size == 0:
        return np.inf
    t1 = t1[t1 >= 0]
    T = (1 - gamma) * t1 + beta
    T = T[T >= t1]
    if T.size == 0:
        return np.inf
    return np.max(T).item()


def pmm_bang_bang_control_policy_minimum_time(p0, v0, p2, v2, u_lower, u_upper):
    Ta = np.array([
        scalar_pmm_bang_bang_control_policy_time(*args)
        for args in zip(p0, v0, p2, v2, u_upper, u_lower)
    ])
    Tb = np.array([
        scalar_pmm_bang_bang_control_policy_time(*args)
        for args in zip(p0, v0, p2, v2, u_lower, u_upper)
    ])
    return np.max(np.min([Ta, Tb], axis=0))


def balance_sum(c, a, b, epsilon=1e-14):
    a = np.array(a)
    b = np.array(b)
    a_zeros = np.abs(a) < epsilon
    b_zeros = np.abs(b) < epsilon
    a[a_zeros] = 0
    b[a_zeros] = c
    a[b_zeros] = c
    b[b_zeros] = 0
    return a, b


def scalar_pmm_bang_bang_control_policy(p0, v0, p2, v2, u_lower, u_upper, T):
    assert u_upper != 0
    gamma = u_lower / u_upper
    beta = (v2 - v0) / u_upper

    a = ((u_lower / 2) * T**2) / (1 - gamma)
    b = v0 * T - (u_lower * beta * T) / (1 - gamma) + (p0 - p2)
    c = ((u_upper * beta**2) / 2) / (1 - gamma)

    alphas = real_roots(a, b, c)
    alphas = alphas[alphas != 0]
    if alphas.size == 0:
        return T, 0
    t1s = (T - beta / alphas) / (1 - gamma)
    t1s, t2s = balance_sum(T, t1s, T - t1s)
    alphas = alphas[np.logical_and(t1s >= 0, t2s >= 0)]
    alpha = alphas[np.argmax(np.abs(alphas))].item()
    t1 = (T - beta / alpha) / (1 - gamma)
    t1, t2 = balance_sum(T, t1, T - t1)
    return t1, alpha


def pmm_bang_bang_control_policy(p0, v0, p2, v2, u_lower, u_upper, T):
    t1, alpha = np.array([
        scalar_pmm_bang_bang_control_policy(*args, T)
        for args in zip(p0, v0, p2, v2, u_lower, u_upper)]).T
    if np.any(np.abs(alpha) > 1):
        # Slowing down the axes requires overshooting the target state
        # the first time to hit it on the way back. Rescale accelerations
        # to stay within limits and increase the trajectory time.
        alpha = alpha / np.max(np.abs(alpha))
        T = pmm_bang_bang_control_policy_minimum_time(
            p0, v0, p2, v2, alpha * u_lower, alpha * u_upper
        )
    dt = np.diff(np.hstack((0, np.sort(t1), T)))
    u = [u_lower] + [None] * 3
    for i, coord in enumerate(np.argsort(t1), start=1):
        u[i] = np.array(u[i - 1])
        u[i][coord] = u_upper[coord]
    u = alpha * np.array(u)
    return dt, u


def pmm_time_optimal_trajectory(p0, v0, p2, v2, u_lower, u_upper):
    T = pmm_bang_bang_control_policy_minimum_time(
        p0, v0, p2, v2, u_lower, u_upper
    )
    if not np.isfinite(T):
        return None

    p, v = p0, v0
    segments = []
    for dt, u in zip(*pmm_bang_bang_control_policy(
        p0, v0, p2, v2, u_lower, u_upper, T
    )):
        if dt == 0:
            continue
        segments.append(
            ConstantAccelerationTrajectory(p, v, u, dt))
        p = segments[-1].position(dt)
        v = segments[-1].velocity(dt)
    return PiecewiseTrajectory(segments)


class Obstacle(ABC):

    @abstractmethod
    def closest_point(
        self, trajectory: Trajectory
    ) -> Tuple[float, Tuple[float, float, float], float]:
        """
        Returns the time, position, and distance of the point
        along the trajectory that is closest to this obstacle.
        """
        pass

class Cylinder(Obstacle):

    def __init__(
        self,
        position: Tuple[float, float, float],
        radius: float, height: float
    ):
        self.position = position
        self.radius = radius
        self.height = height

    def closest_point(
        self, trajectory: Trajectory
    ) -> Tuple[float, Tuple[float, float, float], float]:
        time, point, distance = trajectory.closest_point_to_line(
            self.position, np.array([0, 0, 1]), (0, self.height))
        return time, point, np.clip(distance - self.radius, 0, None)


class State(NamedTuple):
    position: np.ndarray
    velocity: np.ndarray


class Pose(NamedTuple):
    position: np.ndarray
    rotation: Rotation


def to_pose(vec: np.ndarray) -> Pose:
    return Pose(position=vec[:3], rotation=Rotation.from_euler('xyz', vec[3:]))


class Limits(NamedTuple):
    lower: np.ndarray
    upper: np.ndarray


def plan_pmm_time_optimal_trajectory(
    initial_state: State,
    final_state: State,
    acceleration_limits: Limits,
    intermediate_regions: Optional[List[List[State]]] = None,
    obstacles: Optional[List[Obstacle]] = None,
    safe_obstacle_distance: float = 1.,
) -> Tuple[List[State], Trajectory]:
    state_graph = nx.DiGraph()
    states = collections.defaultdict(dict)
    states[0][0] = initial_state

    intermediate_regions = intermediate_regions or []
    regions = intermediate_regions + [[final_state]]
    for i, region in enumerate(regions, start=1):
        for j, prev_state in states[i - 1].items():
            for k, next_state in enumerate(region):
                trajectory = pmm_time_optimal_trajectory(
                    prev_state.position, prev_state.velocity,
                    next_state.position, next_state.velocity,
                    *acceleration_limits
                )
                if not trajectory:
                    continue
                if obstacles:
                    in_collision = False
                    for obstacle in obstacles:
                        time, point, distance = \
                            obstacle.closest_point(trajectory)
                        if distance <= 0:
                            in_collision = True
                            break
                        if distance < safe_obstacle_distance:
                            trajectory.add_landmark('obstacle', time)
                    if in_collision:
                        continue
                states[i][k] = next_state
                if i > 1:  # first trajectory segment starts from initial state
                    trajectory.add_landmark('waypoint', trajectory.start_time)
                if i < len(regions) - 1:  # last trajectory segment ends in goal state
                    trajectory.add_landmark('waypoint', trajectory.end_time)
                state_graph.add_edge((i - 1, j), (i, k), trajectory=trajectory)

    shortest_path_nodes = nx.shortest_path(
        state_graph, (0, 0), (len(regions), 0),
        weight=lambda u, v, d: d['trajectory'].duration)
    shortest_path_edges = zip(shortest_path_nodes[:-1], shortest_path_nodes[1:])
    trajectory = PiecewiseTrajectory([
        state_graph[u][v]['trajectory'] for u, v in shortest_path_edges])
    relevant_states = [states[i][j] for i, j in shortest_path_nodes]

    return relevant_states, trajectory


def linspace_product(start, end, n):
    return np.array(np.meshgrid(
        *np.linspace(start, end, n).T
    )).T.reshape(-1, 3)


def plan_time_optimal_trajectory_through_gates(
    initial_state: State,
    final_state: State,
    gate_poses: List[Pose],
    acceleration_limits: Limits,
    velocity_limits: Limits,
    max_iterations: int = 5,
    num_cone_samples: int = 3,
    cone_refocusing_factor: float = 0.8,
    convergence_epsilon: float = 1.,
    obstacles: Optional[List[Obstacle]] = None,
    safe_obstacle_distance: float = 1.
):
    # TODO: cone refocusing often fails to find a feasible
    # trajectory to refine. Perhaps we should do an exhaustive
    # or randomized search for feasibility before refinement.
    assert max_iterations > 0
    best_time = np.inf
    velocity_limits = [velocity_limits] * len(gate_poses)
    for k in range(1, max_iterations + 1):
        gate_cones = [[
            State(position, rotation.apply(velocity))
            for velocity in spherical2cartesian(
                linspace_product(*velocity_limits[i],
                                 num_cone_samples))
        ] for i, (position, rotation) in enumerate(gate_poses)]

        states, trajectory = plan_pmm_time_optimal_trajectory(
            initial_state, final_state, acceleration_limits,
            gate_cones, obstacles, safe_obstacle_distance)
        if abs(best_time - trajectory.duration) < convergence_epsilon:
            break
        best_time = trajectory.duration

        gate_velocities = (state.velocity for state in states[1:-1])
        gate_states = zip(gate_poses, gate_velocities)
        for i, ((_, rotation), velocity) in enumerate(gate_states):
            # Scale limits velocity search around the optimal velocity
            rotated_velocity = cartesian2spherical(
                rotation.inv().apply(velocity))
            velocity_limits[i] = Limits(
                lower=(1 - cone_refocusing_factor**(1/k)) * rotated_velocity,
                upper=(1 + cone_refocusing_factor**(1/k)) * rotated_velocity)
    else:
        print(f'Trajectory search did not converge after {max_iterations} iterations: {best_time}')

    return trajectory

def main():
    path = plan_time_optimal_trajectory_through_gates(
        initial_state=State(
            position=np.array(
                [-0.9, -2.9, 0.03]),
            velocity=np.zeros(3)),
        final_state=State(
            position=np.array(
                [0.5, -0.4, 0.8]),
            velocity=np.zeros(3)),
        gate_poses=list(map(to_pose, [
            # x, y, z, _, _, yaw
            [-0.5, -0.4, 0.8, 0, 0, 0],
            [0.5, 0.4, 0.8, 0, 0, -1.57],
            [0.5, -0.4, 0.8, 0, 0, -1.57],
            # [-0.5, -0.4, 0.8, 0, 0, 1.57],
            # [0.5, 0.4, 0.8, 0, 0, -1.57],
            # [0.5, -0.4, 0.8, 0, 0, -1.57],
        ])),
        #obstacles=[Cylinder([0, 0, 0], 0.3, 2.)],
        acceleration_limits=Limits(
            lower=-2 * np.ones(3),
            upper=2 * np.ones(3),
        ),
        velocity_limits=Limits(
            lower=np.array([0.1, -np.pi/6, -np.pi/6]),
            upper=np.array([5., np.pi/6, np.pi/6]),
        ),
        num_cone_samples=3
    )

    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')

    curve_samples = path.evenly_spaced_points(100, tolerance=1e-4)
    reparameterized_points = [pos for _, pos, _ in curve_samples]

    x, y, z = zip(*reparameterized_points)
    ax.plot(x, y, z, marker='x')

    plt.show()


if __name__ == "__main__":
    main()
