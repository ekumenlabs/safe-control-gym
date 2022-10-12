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

from competition_utils import Command


class StageActionGotoXY():
    """
    Execution stage that generates a GOTOXY command and then waits until it has completed execution
    """

    def __init__(self,
                 dt: float,
                 x: float,
                 y: float,
                 z: float,
                 yaw: float,
                 duration: float,
                 ):
        self._dt = dt
        self.TARGET_POSE = (x, y, z)
        self.TARGET_YAW = yaw
        self._duration = duration

    def run(self, global_iteration, stage_iteration, pos, vel, rpy, pqr, corrections) -> Tuple[bool, Command, list]:
        command_type = Command(0)  # None.
        args = []
        done = False

        if stage_iteration == 0:
            command_type = Command(5)  # GoTo.
            args = [[*self.TARGET_POSE],
                    self.TARGET_YAW, self._duration, False]

        # once time has ran off, move on to the next stage
        if stage_iteration * self._dt >= self._duration:
            done = True

        return done, command_type, args

    def reset(self):
        pass


if __name__ == '__main__':
    uut = StageActionGotoXY(0.1, x=1., y=2., z=3., yaw=4., duration=2.)
    dummy_obs = [0]*12
    for i in range(30):
        print(uut.run(dummy_obs, 100 + i, i))
