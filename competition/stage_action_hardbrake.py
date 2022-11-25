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


class StageActionHardBrake():
    """
    Brakes hard using the mellinger controller to stabilize on the goal
    """

    def __init__(self,
                 dt: float,
                 goal_pose: np.ndarray,
                 duration: float,
                 ):
        self._dt = dt
        self._goal_pose = np.array(goal_pose)
        self._prev_pos = None
        self._duration = duration

        self._goal_distance_tolerance = 0.1  # m
        self._goal_velocity_tolerance = 0.1  # m/s

        self._min_delta_v = 0.8  # m/s

        # If the distance to the goal is greater than this value,
        # try to stop in place.
        self._goal_distance_max = 1.5  # m

        self.reset()

    def run(self, global_iteration, stage_iteration, pos, vel, rpy, pqr) -> Tuple[bool, Command, list]:

        distance_vector = pos - self._goal_pose

        current_distance = np.sqrt(np.dot(distance_vector, distance_vector))
        current_velocity = np.sqrt(np.dot(vel, vel))

        if stage_iteration == 0:
            # set a minimum approach vel to avoid crawling ot the goal if we were arriving slowly when we
            # transitioned into this stage
            self._delta_v = max(current_velocity, self._min_delta_v)
            self._delta_d = current_distance
            # this calculation comes from stating: whatever my initial velocity and distance,
            # I want to land on the goal pose assuming linear deceleration.
            #   delta_d = delta_v * delta_t - 0.5 * a * delta_t**2]
            #   delta_v = a * delta_t
            self._delta_time = 2.0 * self._delta_d / self._delta_v
            self._brake_acc = self._delta_v / self._delta_time
            # Unbiased asymptotic unit vector estimator for the approach direction
            self._approach_normal = distance_vector / \
                (current_distance + 0.001)  # cheap anti zero-div protection

            current_horizontal_distance = np.sqrt(
                np.dot(distance_vector[0:2], distance_vector[0:2]))
            if current_horizontal_distance > self._goal_distance_max:
                # If we are too far away from the goal, update the goal pose for
                # the drone to make its best effort at stopping at the current
                # position. Do this only in the first iteration.
                self._goal_pose = pos

        if current_distance < self._goal_distance_tolerance \
                and current_velocity < self._goal_velocity_tolerance:
            self._stabilized_counter += 1
        else:
            self._stabilized_counter = 0

        remaining_time = max(self._delta_time -
                             stage_iteration * self._dt, 0.0)
        brake_time = self._delta_time - remaining_time
        remaining_distance = max(
            self._delta_d - (self._delta_v * brake_time - 0.5 * self._brake_acc * brake_time**2), 0.0)

        target_pos = self._goal_pose + self._approach_normal * remaining_distance
        target_vel = -1.0 * self._approach_normal * \
            (remaining_time * self._brake_acc)
        target_acc = self._approach_normal * \
            (self._brake_acc if remaining_time > 0.0 else 0.0)
        target_rpy_rates = np.zeros(3)
        target_yaw = 0

        # Did we converge?
        done = self._stabilized_counter * self._dt >= self._duration
        command_type = Command.FULLSTATE
        args = [target_pos, target_vel, target_acc,
                target_yaw, target_rpy_rates]

        return done, command_type, args

    def reset(self):
        self._stabilized_counter = 0


if __name__ == '__main__':
    pass
