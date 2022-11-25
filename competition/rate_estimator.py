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


class RateEstimator():

    def __init__(self, dt: float):
        self._dt = dt

        self.PARAM_IIR_ALPHA = 0.8

        self.reset()

    def estimate(self, current_pos: np.ndarray, current_rpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._prev_pos is None:
            self._prev_pos = current_pos
        if self._prev_rpy is None:
            self._prev_rpy = current_rpy

        estimated_velocity = self._estimate_velocity(
            current_pos, self._prev_pos)
        estimated_body_rates = self._estimate_body_rates(
            current_rpy, self._prev_rpy)

        self._prev_pos = current_pos
        self._prev_rpy = current_rpy

        return estimated_velocity, estimated_body_rates

    def reset(self):
        self._prev_pos = None
        self._prev_rpy = None
        self.filtered_euler_angle_rates = 0.0

    def _estimate_velocity(self, current_pos, prev_pos):
        estimated_velocity = (current_pos - prev_pos) / self._dt
        return estimated_velocity

    def _estimate_body_rates(self, current_rpy, prev_rpy):
        """
        Reference:
        https://aviation.stackexchange.com/questions/83993/the-relation-between-euler-angle-rate-and-body-axis-rates
        """
        alpha = self.PARAM_IIR_ALPHA

        euler_angle_rates = (current_rpy - prev_rpy) / self._dt
        euler_angle_rates[2] = 0.0

        self.filtered_euler_angle_rates = alpha * \
            self.filtered_euler_angle_rates + (1.0 - alpha) * euler_angle_rates

        phi, theta, psi = current_rpy

        rotation_matrix = np.array([
            [1.0,          0.0,               -np.sin(theta)],
            [0.0,  np.cos(phi),  np.sin(phi) * np.cos(theta)],
            [0.0, -np.sin(phi),  np.cos(phi) * np.cos(theta)]
        ])

        pqr_body_rates = rotation_matrix @ self.filtered_euler_angle_rates

        return pqr_body_rates  # TODO disabled because it destabilizes the control


if __name__ == '__main__':
    pass
