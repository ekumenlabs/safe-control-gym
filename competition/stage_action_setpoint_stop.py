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


class StageActionSetPointStop():
    """
    Execution stage that will generate a SetPointStop command whenever it's called, and will
    move on to the next stage right away. Intended to be used after a low-level command is sent
    to the drone, to re-enable the higher level controller's (GOTO, LAND, TAKEOFF) state.
    """

    def __init__(self):
        pass

    def run(self, global_iteration, stage_iteration, pos, vel, rpy, pqr) -> Tuple[bool, Command, list]:
        command_type = Command(6)  # notify setpoint stop.
        args = []
        done = True
        return done, command_type, args

    def reset(self):
        pass


if __name__ == '__main__':
    uut = StageActionSetPointStop()
    dummy_obs = [0] * 12
    for i in range(3):
        print(uut.run(dummy_obs, 100 + i, i))
