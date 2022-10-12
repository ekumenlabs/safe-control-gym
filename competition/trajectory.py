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

import math
import numpy as np

from abc import ABC, abstractmethod
from functools import lru_cache
from matplotlib import pyplot as plt
from scipy import optimize
from typing import Generator, Iterable, List, NamedTuple, Optional, Tuple


def fastpolytrim(p, tol=1e-16):
    for i in reversed(range(len(p))):
        if abs(p[i]) > tol:
            break
    return p[:i + 1]

def fastpolyadd(p, q):
    if len(p) < len(q):
        p, q = q, p
    r = tuple(p[i] + q[i] for i in range(len(q)))
    if len(p) > len(q):
        r += tuple(p[len(q):])
    return fastpolytrim(r)

def fastpolyneg(p):
    return tuple(-pv for pv in p)

def fastpolymul(p, q):
    if len(p) < len(q):
        p, q = q, p
    r = (0,)
    for i in reversed(range(len(p))):
        s = tuple(p[i] * q[j] for j in range(len(q)))
        if i > 0:
            s = (0,) * i + s
        r = fastpolyadd(s, r)
    return fastpolytrim(r)

def fastpolypow(p, n):
    r = (1,)
    for _ in range(n):
        r = fastpolymul(r, p)
    return r

def fastpolyder(p):
    return tuple(pv * n for n, pv in enumerate(p[1:], start=1))

def fastpolyroots(p):
    order = len(p) - 1
    if order == 3:
        d, c, b, a = map(float, p)
        a2, a1, a0 = b/a, c/a, d/a
        w = (-1 + 1j * 3**(1/2)) / 2
        if a2 == 0 and a1 == 0:
            r = (-a0)**(1/3)
            return r, r * w, r * w**2
        P = a1 - (a2**2) / 3
        Q = -a0 + a1 * a2 / 3 - 2 * (a2**3) / 27
        D = (Q / 2)**2 + (P / 3)**3
        B = (-Q / 2 - D**(1/2))**(1/3)
        if B == 0:
            B = (-Q / 2 + D**(1/2))**(1/3)
        A = P / (3 * B)
        return (
            -a2 / 3 + A - B,
            -a2 / 3 + A * w**2 - B * w,
            -a2 / 3 + A * w - B * w**2)
    if order == 2:
        c, b, a = map(float, p)
        r = b * b - 4. * a * c
        return ((-b + r**(1/2)) / (2 * a),
                (-b - r**(1/2)) / (2 * a))
    if order == 1:
        b, a = map(float, p)
        return (-b / a,)
    return tuple(np.roots(reversed(self.coeffs)))

def fastpolyeval(p, x):
    y = 0
    for c in reversed(p):
        y = y * x + c
    return y

class fastpoly:

    def __init__(self, *coeffs):
        assert len(coeffs) > 0
        self.coeffs = coeffs

    def __call__(self, x: float) -> float:
        return fastpolyeval(self.coeffs, x)

    def __add__(self, other: 'fastpoly') -> 'fastpoly':
        return fastpoly(*fastpolyadd(self.coeffs, other.coeffs))

    def __sub__(self, other: 'fastpoly') -> 'fastpoly':
        return fastpoly(*fastpolyadd(
            self.coeffs, fastpolyneg(other.coeffs)))

    def __mul__(self, other: 'fastpoly') -> 'fastpoly':
        return fastpoly(*fastpolymul(self.coeffs, other.coeffs))

    def __pow__(self, n: int) -> 'fastpoly':
        return fastpoly(*fastpolypow(self.coeffs, n))

    def __neg__(self) -> 'fastpoly':
        return fastpoly(*fastpolyneg(self.coeffs))

    @property
    def order(self) -> int:
        return len(self.coeffs) - 1

    @property
    def deriv(self) -> 'fastpoly':
        return fastpoly(*fastpolyder(self.coeffs))

    @property
    def roots(self) -> Tuple[float]:
        return fastpolyroots(self.coeffs)

def only_reals(values, tol=1e-14):
    return tuple(
        getattr(x, 'real', x) for x in values
        if abs(getattr(x, 'imag', 0.)) < tol
    )

def clip(x, amin, amax):
    return min(max(x, amin), amax)

class ParametricCurve:

    def __init__(
        self,
        x: fastpoly,
        y: fastpoly,
        z: fastpoly,
        domain: Tuple[float, float],
    ):
        self._x = x
        self._y = y
        self._z = z
        self._x_dot = x.deriv
        self._y_dot = y.deriv
        self._z_dot = z.deriv
        self._arclength_antiderivative = None

        self.domain = domain

    def point(self, time: float) -> Tuple[float, float, float]:
        """
        Returns the position vector at a given point in time.
        """
        return (self._x(time), self._y(time), self._z(time))

    def tangent(self, time: float) -> Tuple[float, float, float]:
        """
        Returns the velocity vector at a given point in time.
        """
        return (self._x_dot(time), self._y_dot(time), self._z_dot(time))

    def closest_point_to_line(
        self,
        origin: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        domain: Tuple[float, float]
    ) -> Tuple[float, Tuple[float, float, float]]:
        """
        Returns minimum distance between curve and point as well as the time at which it is achieved.
        """
        ox, oy, oz = origin
        dx, dy, dz = direction
        assert (dx**2 + dy**2 + dz**2) == 1
        distance_deriv_num = (
            self._x_dot * (self._x - fastpoly(ox)) +
            self._y_dot * (self._y - fastpoly(oy))
        )
        time = None
        shortest_distance = np.inf
        #print(distance_deriv_num.roots)
        for u in only_reals(distance_deriv_num.roots):
            u = clip(u, *self.domain)
            x_u, y_u, z_u = self.point(u)
            v = clip(
                dx * (x_u - ox) +
                dy * (y_u - oy) +
                dz * (z_u - oz),
                *domain
            )
            distance = math.sqrt(
                (x_u - dx * v - ox)**2 +
                (y_u - dy * v - oy)**2 +
                (z_u - dz * v - oz)**2)
            if shortest_distance > distance:
                shortest_distance = distance
                time = u
        return time, self.point(time), shortest_distance

    @lru_cache(maxsize=10000)
    def interval_arclength(self, start: float, end: float) -> float:
        """
        Returns the arclength of a given interval.
        """
        if self._arclength_antiderivative is None:
            # Compute the arclength antiderivative analytically
            squared_velocity = self._x_dot**2 + self._y_dot**2 + self._z_dot**2

            if squared_velocity.order == 2:
                c, b, a = squared_velocity.coeffs
                def antiderivative(time: float):
                    # See expressions (121), (122), and (125) in
                    # https://link.springer.com/content/pdf/bbm:978-1-4612-1520-2/1
                    velocity = math.sqrt(clip(squared_velocity(time), 0, np.inf))
                    if a > 0:
                        log_argument = 2*a * time + b + 2 * math.sqrt(a) * velocity
                        term = math.log(log_argument) / math.sqrt(a) if log_argument > 0 else 0
                    else:
                        term = math.asin((-2*a * time - b) / math.sqrt(b**2 - 4*a*c)) / math.sqrt(-a)
                    return ((2*a * time + b) / (4*a)) * velocity + ((4*a*c - b**2) / (8*a)) * term
                self._arclength_antiderivative = antiderivative

            elif squared_velocity.order == 0:
                velocity = math.sqrt(squared_velocity.coeffs[0])
                def antiderivative(time: float):
                    return velocity * time
                self._arclength_antiderivative = antiderivative

            else:
                raise RuntimeError(
                    "Parametric equations must be constant acceleration parabolas (degree 2) "
                    "or constant velocity lines (degree 1). "
                    f"Current equations are:\n{np.poly1d(x)}\n{np.poly1d(y)}\n{np.poly1d(z)}"
                )
        return self._arclength_antiderivative(end) - self._arclength_antiderivative(start)

    def arclength(self) -> float:
        """
        Returns the total arclength of the curve.
        """
        return self.interval_arclength(*self.domain)

    def points(self, num_points: int) -> Generator[Tuple[float, float, float], None, None]:
        """
        Returns the points resulting from applying a linspace to the curve.
        """
        for time in np.linspace(*self.domain, num_points):
            yield self.point(time)


class Landmark(NamedTuple):
    """A relevant entity along a trajectory."""
    location: float
    kind: str


class Trajectory(ABC):

    @property
    @abstractmethod
    def start_time(self):
        """
        Start time of the trajectory.
        """
        pass

    @property
    @abstractmethod
    def end_time(self):
        """
        End time of the trajectory.
        """
        pass

    @property
    def duration(self):
        """
        Duration of the trajectory.
        """
        return self.end_time - self.start_time

    @abstractmethod
    def arclength(self, time: Optional[float] = None) -> float:
        """
        Returns the arclength of the path up to a given time.

        If no time is specified, the total arclength is returned.
        """
        pass

    @abstractmethod
    def position(self, time: float) -> Tuple[float, float, float]:
        """
        Returns the position vector at a given point in time.
        """
        pass

    @abstractmethod
    def velocity(self, time: float) -> Tuple[float, float, float]:
        """
        Returns the velocity vector at a given point in time.
        """
        pass

    @property
    @abstractmethod
    def landmarks(self) -> Iterable[Landmark]:
        """
        Returns an iterable over the landmarks along the trajectory.
        """
        pass

    @abstractmethod
    def add_landmark(self, kind: str, time: float):
        """
        Adds a landmark at a given time (and space).
        """
        pass

    @abstractmethod
    def points(self, num_points: int) -> Generator[Tuple[float, float, float], None, None]:
        """
        Returns the points resulting from applying a linspace to every segment in the path.
        """
        pass

    @abstractmethod
    def closest_point_to_line(
        self,
        point: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        interval: Tuple[float, float]
    ) -> Tuple[float, Tuple[float, float, float], float]:
        """
        Returns the time, position, and distance of the point along the trajectory
        that is closest to the specified line segment.
        """
        pass


class ConstantAccelerationTrajectory(Trajectory):
    """A constant acceleration trajectory in 3D space."""

    def __init__(
        self,
        p0: Tuple[float, float, float],
        v0: Tuple[float, float, float],
        u: Tuple[float, float, float],
        t: float
    ):
        super().__init__()
        assert t > 0, t
        self._curve = ParametricCurve(
            fastpoly(p0[0], v0[0], u[0] / 2.),
            fastpoly(p0[1], v0[1], u[1] / 2.),
            fastpoly(p0[2], v0[2], u[2] / 2.),
            (0, t)
        )
        self._landmarks = []

    @property
    def start_time(self):
        return self._curve.domain[0]

    @property
    def end_time(self):
        return self._curve.domain[1]

    @property
    def landmarks(self):
        return self._landmarks

    def add_landmark(self, kind: str, time: float):
        self._landmarks.append(Landmark(
            self.arclength(time), kind))

    def position(self, time: float) -> Tuple[float, float, float]:
        return self._curve.point(time)

    def velocity(self, time: float) -> Tuple[float, float, float]:
        return self._curve.tangent(time)

    @lru_cache(maxsize=10000)
    def arclength(self, time: Optional[float] = None) -> float:
        if time is None:
            time = self._curve.domain[1]
        return self._curve.interval_arclength(self._curve.domain[0], time)

    def points(self, num_points: int) -> Generator[Tuple[float, float, float], None, None]:
        return self._curve.points(num_points)

    def closest_point_to_line(
        self,
        point: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        interval: Tuple[float, float]
    ) -> Tuple[float, Tuple[float, float, float], float]:
        return self._curve.closest_point_to_line(point, direction, interval)


class PiecewiseTrajectory(Trajectory):
    """
    A piecewise trajectory in 3D space.

    To ensure compact support, time over the [0, duration] interval is mapped
    to each segment [start_time, end_time] interval as necessary.
    """

    def __init__(self, segments: List[Trajectory]):
        """"""
        super().__init__()
        assert segments
        self._segments = []

        for segment in segments:
            if isinstance(segment, PiecewiseTrajectory):
                # Avoid unnecessary nesting if possible
                self._segments.extend(segment._segments)
            else:
                self._segments.append(segment)

    @property
    def start_time(self):
        return 0.

    @property
    def end_time(self):
        return sum(segment.duration for segment in self._segments)

    @property
    def landmarks(self):
        offset = 0.
        for segment in self._segments:
            for landmark in segment.landmarks:
                yield Landmark(
                    landmark.location + offset,
                    landmark.kind
                )
            offset += segment.arclength()

    def _get_segment_index(self, time: float) -> Tuple[int, float]:
        """
        Returns the index to the segment spanning the interval that contains the given time.

        Time is also returned, corrected to lay in the corresponding segment time interval.
        """
        offset = 0.
        for i, segment in enumerate(self._segments):
            if segment.duration >= time - offset:
                return i, time - offset + segment.start_time
            offset += segment.duration
        if time == offset:
            return len(self._segments) - 1, self._segments[-1].end_time
        raise ValueError(f'{time} not in [{self.start_time}, {self.end_time}]')

    def arclength(self, time: Optional[float] = None) -> float:
        if time is None:
            return sum(segment.arclength() for segment in self._segments)
        index, time = self._get_segment_index(time)
        return sum(
            segment.arclength() for segment in self._segments[:index]
        ) + self._segments[index].arclength(time)

    def position(self, time: float) -> Tuple[float, float, float]:
        index, time = self._get_segment_index(time)
        return self._segments[index].position(time)

    def velocity(self, time: float) -> Tuple[float, float, float]:
        index, time = self._get_segment_index(time)
        return self._segments[index].velocity(time)

    def add_landmark(self, kind: str, time: float):
        index, time = self._get_segment_index(time)
        return self._segments[index].add_landmark(kind, time)

    def closest_point_to_line(
        self,
        point: Tuple[float, float, float],
        direction: Tuple[float, float, float],
        interval: Tuple[float, float]
    ) -> Tuple[float, Tuple[float, float, float], float]:
        offset = 0.
        time_of_closest_point = None
        closest_point = None
        shortest_distance = np.inf
        for segment in self._segments:
            time, position, distance = \
                segment.closest_point_to_line(point, direction, interval)
            if shortest_distance > distance:
                time_of_closest_point = \
                    time - segment.start_time + offset
                closest_point = position
                shortest_distance = distance
            offset += segment.duration
        return time_of_closest_point, closest_point, shortest_distance

    def points(self, num_points: int) -> Generator[Tuple[float, float, float], None, None]:
        num_points_per_segment = int(num_points / len(self._segments))
        for segment in self._segments:
            yield from segment.points(num_points_per_segment)

    def evenly_spaced_points(
        self, num_points: int, tolerance: float
    ) -> Generator[Tuple[float, Tuple[float, float, float], List[str]], None, None]:
        """
        Returns evenly spaced points along the curve, plus nearby landmarks if any.
        Useful for (re)parameterization with respect to arclength.
        """
        def make_segment_generator() -> Generator[Tuple[ParametricCurve, float], float, None]:
            """
            Generator method that, for a given arclength in the curve, returns the
            corresponding segment and delta arclength needed to reach it.
            Assumes that the input lengths increase monotonically.
            """
            lower_bound = 0  # Lower bound, sum of the lengths of the previous segments
            input_length: float = yield
            for segment in self._segments:
                upper_bound = lower_bound + segment.arclength()
                while lower_bound <= input_length <= upper_bound:
                    input_length: float = yield segment, input_length - lower_bound
                lower_bound = upper_bound

        def make_landmark_generator(
            max_distance: float
        ) -> Generator[List[Landmark], float, None]:
            """
            Generator method that, for a given arclength in the curve, returns the
            landmarks that are within the specified maximum distance.
            Assumes that the input lengths increase monotonically.
            """
            current = end = 0
            landmarks = list(self.landmarks)
            landmarks.sort(key=lambda landmark : landmark.location)
            input_length: float = yield
            while current < len(landmarks):
                if abs(landmarks[current].location - input_length) > max_distance:
                    # The input lenght is far from the current landmark,
                    # wait for next length and check this again.
                    input_length: float = yield []
                    continue

                # Initial limit found, find remaining landmarks close to the input length.
                end = current + 1
                while end < len(landmarks) and abs(
                    landmarks[end].location - input_length
                ) <= max_distance:
                    end += 1

                input_length: float = yield landmarks[current:end]
                current = end

            while True:
                # No landmarks left.
                yield []

        segment_generator = make_segment_generator()
        next(segment_generator)  # Start execution of the generator
        landmark_generator = make_landmark_generator(
            self.arclength() / (num_points - 1))
        next(landmark_generator)  # Start execution of the generator
        for length in np.linspace(0, self.arclength(), num_points):
            segment, target_length = segment_generator.send(length)
            if target_length < tolerance:
                target_time = segment.start_time
            elif abs(target_length - segment.arclength()) < tolerance:
                target_time = segment.end_time
            else:
                def objective(time):
                    length = segment.arclength(time)
                    return target_length - length
                target_time = optimize.bisect(
                    objective,
                    a=segment.start_time,
                    b=segment.end_time,
                    xtol=tolerance
                )
            position = segment.position(target_time)
            landmarks = landmark_generator.send(length)
            yield length, position, landmarks

def main():
    path = PiecewiseTrajectory([
        ConstantAccelerationTrajectory(
            p0=np.zeros(3), v0=np.zeros(3),
            u=np.array([0.2, 1., -0.4]),
            t=2.
        ),
        ConstantAccelerationTrajectory(
            p0=np.array([0.4, 2, -0.8]),
            v0=np.array([0.4, 2., -0.8]),
            u=np.array([0, -1., 0.4]),
            t=5.
        ),
        ConstantAccelerationTrajectory(
            p0=np.array([2.4, -0.5, 0.2]),
            v0=np.array([0.4, -3., 1.2]),
            u=np.array([2., 0, 0]),
            t=2.
        )
    ])

    fig = plt.figure(figsize=(4,4))
    ax = fig.add_subplot(111, projection='3d')

    points = path.points(50)
    x, y, z = zip(*points)
    ax.plot(x, y, z, marker='o')

    curve_samples = path.evenly_spaced_points(50, 1e-4)
    reparameterized_points = [pos for _, pos, _ in curve_samples]

    x, y, z = zip(*reparameterized_points)
    ax.plot(x, y, z, marker='x')

    plt.show()


if __name__ == '__main__':
    main()
