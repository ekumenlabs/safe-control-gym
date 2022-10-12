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

from dataclasses import dataclass


@dataclass()
class EkControllerConfig:
    # control loop interval and frequency
    ctrl_timestep: float
    ctrl_freq: float
    # using firmware controller or using simple PID software only controller
    use_firmware: bool
    # initial values of [x, x_dot y, y_dot, z, z_dot, roll, pitch, yaw, r, p q]
    initial_obs: list
    # gate dimensions data
    gate_dimensions: dict
    # obstacle dimensions data
    obstacle_dimensions: dict
    # nominal_gates_pos_and_type
    nominal_gates_pose_and_type: dict
    # nominal_obstacles_pos
    nominal_obstacles_pos: dict
    # target pose after the trajectory
    x_reference: tuple
    # Physical variables
    mass: float
    ixx: float
    iyy: float
    izz: float
