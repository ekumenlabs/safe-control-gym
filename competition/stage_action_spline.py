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

from typing import Tuple

import numpy as np

from competition_utils import Command
from mpcc_controller import MPCCController


class StageActionSpline():
    """
    Experimental MPCC controller execution stage.
    Further details TBD.
    """

    def __init__(self,
                 dt: float,
                 waypoints_pos: np.ndarray,
                 waypoints_arg: np.ndarray,
                 arc_vel: float
                 ):
        self._dt = dt
        self._arc_vel = arc_vel
        self._waypoints_pos = waypoints_pos
        self._waypoints_arg = waypoints_arg

        self._end_of_curve_arg = max(waypoints_arg)

        controller = MPCCController(dt=dt)
        self._spline_func = controller._build_mpcc_build_contour_interpolant_functions(
            waypoints=waypoints_pos,
            contour_poses=waypoints_arg
        )

    def run(self, global_iteration, stage_iteration, pos, vel, rpy, pqr, corrections) -> Tuple[bool, Command, list]:
        done = False
        theta = self._arc_vel * stage_iteration * self._dt
        if theta > self._end_of_curve_arg:
            theta = self._end_of_curve_arg
            done = True

        pos, vel = self._next_pos(theta=theta, arc_vel=self._arc_vel)

        target_pos = np.array(pos)
        target_vel = np.array(vel)
        target_acc = np.zeros(3)
        target_rpy_rates = np.zeros(3)
        target_yaw = 0

        command_type = Command(1)  # cmdFullState.
        args = [target_pos, target_vel, target_acc,
                target_yaw, target_rpy_rates]

        return done, command_type, args

    def reset(self):
        self._theta = 0

    def _next_pos(self, theta: float, arc_vel: float):
        results_matrix = self._spline_func(theta=theta)
        contour_curve_pos = np.array(results_matrix["contour_curve"].T)[0, :]
        contour_tangent = np.array(results_matrix["contour_tangent"].T)[0, :]
        return contour_curve_pos, contour_tangent * arc_vel


if __name__ == '__main__':
    pass
