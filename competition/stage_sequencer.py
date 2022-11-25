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

from typing import Any, Dict, Tuple

import numpy as np

from competition_utils import Command
from stage_action_none import StageActionNone


class StageSequencer():
    """
    Simple execution stage sequencer. Will execute the stages one after another until there are no more stages left.
    If reset, it will reset all the stages and will then start from the start.
    """

    def __init__(self, stages: list):
        self._stages = stages
        self.reset()

    def reset(self):
        for stage in self._stages:
            stage.reset()
        self._stage_index = 0
        self._stage_iteration = 0
        self._log_transition()

    def run(self,
            global_iteration: int,
            cur_pos: np.ndarray,
            est_vel: np.ndarray,
            cur_rpy: np.ndarray,
            est_pqr: np.ndarray,
            ) -> Tuple[bool, Command, list]:
        if self._stage_index >= len(self._stages):
            # Warn us, but create a none command
            print(
                "WARNING! Stage sequencer continues to execute after running out of _stages")

        done, command, args = self._current_stage().run(
            global_iteration, self._stage_iteration, cur_pos, est_vel, cur_rpy, est_pqr)

        # move on to the next stage
        if done:
            self._stage_index = self._stage_index + 1
            self._stage_iteration = 0
            self._log_transition()
        else:
            self._stage_iteration = self._stage_iteration + 1

        return done, command, args

    def stage_index(self) -> int:
        return self._stage_index

    def stage_iteration(self) -> int:
        return self._stage_iteration

    def _log_transition(self):
        print("Transitioning to {}".format(
            self._current_stage().__class__.__name__))

    def _current_stage(self):
        if self._stage_index < len(self._stages):
            return self._stages[self._stage_index]
        return StageActionNone()


if __name__ == '__main__':
    from stage_action_take_off import StageActionTakeOff
    from stage_action_goto_xy import StageActionGotoXY
    from stage_action_land import StageActionLand
    from stage_action_none import StageActionNone

    stages = [
        StageActionTakeOff(0.1, 0.2),
        StageActionGotoXY(0.1, 1, 2, 3, 4, 0.4),
        StageActionNone(),
        StageActionGotoXY(0.1, 1, 2, 3, 4, 0.4),
        StageActionLand(0.1, 0.2),
    ]

    uut = StageSequencer(stages)

    dummy_obs = [0]*12

    for i in range(20):
        done, command, args = uut.run(dummy_obs, 100 + i)
        print("  - Iteration: global={}, stage={}".format(
            uut.stage_index(), uut.stage_iteration()))
        print("  - command(args): {}({}) ".format(command, args))
        print()
