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


class StageActionLand():
    """
    Execution stage that will generate a LAND command in the first iteration
    and will then wait until the command has been completed before moving on.
    """

    def __init__(self,
                 dt,
                 duration,
                 height
                 ):
        self._dt = dt
        self._height = height
        self._duration = duration

    def run(self, global_iteration, stage_iteration, pos, vel, rpy, pqr, corrections) -> Tuple[bool, Command, list]:
        # default to no command
        command_type = Command(0)  # None.
        args = []
        done = False

        # on the first iteration of the stage execution, send a land command
        if stage_iteration == 0:
            command_type = Command(3)  # Land.
            args = [self._height, self._duration]

        # once time has ran off, move on to the next stage
        if stage_iteration * self._dt >= self._duration:
            done = True

        return done, command_type, args

    def reset(self):
        pass


if __name__ == '__main__':
    uut = StageActionLand(0.1, 1.0, 2.0)
    dummy_obs = [0]*12
    for i in range(20):
        print(uut.run(dummy_obs, 100 + i, i))
