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

import time
from typing import Tuple

import numpy as np

from competition_utils import Command
from mpcc_controller import MPCCController


class StageActionMPCC():
    """
    Experimental MPCC controller execution stage.
    Further details TBD.
    """

    def __init__(self,
                 dt: float,
                 mass: float,
                 ixx: float,
                 iyy: float,
                 izz: float,
                 mpcc_horizon_len: int,
                 waypoints_pos: np.ndarray,
                 waypoints_arg: np.ndarray,
                 waypoints_marks: np.ndarray,
                 ):
        self._dt = dt
        self._waypoints_pos = waypoints_pos
        self._waypoints_arg = waypoints_arg
        self._waypoints_marks = waypoints_marks
        self._controller = MPCCController(
            dt=dt,
            mass=mass,
            ixx=ixx,
            iyy=iyy,
            izz=izz,
            mpcc_horizon_len=mpcc_horizon_len,
        )

        print("Setting up MPCC, will take a few seconds to run JIT compiler...")

        self._controller.setup(
            waypoints=self._waypoints_pos,
            contour_poses=self._waypoints_arg,
            landmarks=self._waypoints_marks,
        )
        self._goal_tolerance = 0.9

        self._start_time = time.time()

    def run(self, global_iteration, stage_iteration, pos, vel, rpy, pqr) -> Tuple[bool, Command, list]:
        if (stage_iteration == 0):
            self._start_time = time.time()

        solver_output = self._controller.solve(
            current_pos=pos,
            current_vel=vel,
            current_rpy=rpy,
            current_pqr=pqr,
        )

        if solver_output is None:
            # In case of failure, send a "do nothing" command instead of throwing an exception.
            # This allows the evaluation script to collect metrics from the episode.
            # TODO: Replace this with a recovery plan. Ideally, try to follow the curve at a slower
            # pace until the optimizer picks up again.
            print("The MPCC controller failed to converge to a solution")
            return True, Command.NONE, None

        target_pos = np.array(solver_output[0])
        target_vel = np.array(solver_output[1])
        target_acc = np.zeros(3)
        target_yaw = solver_output[2][-1]
        target_rpy_rates = np.array(solver_output[3])

        current_carrot_pos = solver_output[4]

        # command_type = Command(1)  # cmdFullState.
        # args = [target_pos, target_vel, target_acc,
        #         target_yaw, target_rpy_rates]

        command_type = Command(7)  # cmdCurve.
        args = [None]

        # are we there yet?
        done = False
        vector_to_goal = (current_carrot_pos - self._waypoints_pos[-1])
        horizontal_distance_to_goal = np.sqrt(
            np.dot(vector_to_goal[0:2], vector_to_goal[0:2]))
        if (horizontal_distance_to_goal < self._goal_tolerance):
            done = True
            simulated_time = stage_iteration * self._dt
            wall_time = time.time() - self._start_time
            rt_factor = simulated_time / \
                (wall_time + 1e-5)  # don't divide by zero
            print("MPCC Stage time stats:")
            print(" - Stage wall-clock duration: {:.2f} sec".format(wall_time))
            print(" - Stage simulated duration:  {:.2f} sec".format(simulated_time))
            print(" - Real-time factor:          {:.2f}".format(rt_factor))

        return done, command_type, args

    def reset(self):
        self._controller.reset()


if __name__ == '__main__':
    pass
