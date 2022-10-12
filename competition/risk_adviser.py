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

from enum import Enum

import numpy as np


class RiskProfile(Enum):
    CONSERVATIVE = 0
    RECKLESS = 1


class RiskAdviser():

    def __init__(self):
        self._episode_count = 1
        self._episodes_completed = []
        self._priori_gate_locations = []
        self._exact_gate_locations = []

    def episode_advice(self):
        if self._episode_count == 1:
            # we're still collecting data
            return self._pack_advise(RiskProfile.CONSERVATIVE)
        elif self._episode_count == 2:
            # we're still collecting data
            return self._pack_advise(RiskProfile.CONSERVATIVE)
        elif self._episode_count == 3:
            if self._scene_is_random_between_episodes():
                # level 3
                return self._pack_advise(RiskProfile.CONSERVATIVE)
            else:
                # level 0, 1, 2 or sim2real
                return self._pack_advise(RiskProfile.RECKLESS)
        elif self._episode_count == 4:
            # This is the one that counts!!!!
            if self._scene_is_random_between_episodes() or self._we_crashed_the_latest_episode():
                # level 3, or we are out of luck
                return self._pack_advise(RiskProfile.CONSERVATIVE)
            else:
                # level 0, 1, 2 or sim2real
                return self._pack_advise(RiskProfile.RECKLESS)
        else:
            print("Something wrong! Too many episodes for our strategy!")
            return self._pack_advise(RiskProfile.CONSERVATIVE)

    def episode_results(self, we_completed_the_episode, priori_gate_locations, exact_gate_locations):
        self._episodes_completed.append(we_completed_the_episode)
        self._priori_gate_locations.append(priori_gate_locations)
        self._exact_gate_locations.append(exact_gate_locations)
        self._episode_count = self._episode_count + 1

    def _we_crashed_the_latest_episode(self):
        return not self._episodes_completed[-1]

    def _scene_is_random_between_episodes(self):
        diff_between_prev_and_exact_in_ep_0 = not self._gate_data_is_eq(
            self._priori_gate_locations[0],
            self._exact_gate_locations[0],
        )
        diff_between_exact_in_ep_0_and_1 = not self._gate_data_is_eq(
            self._exact_gate_locations[0],
            self._exact_gate_locations[1],
        )
        if diff_between_prev_and_exact_in_ep_0 and diff_between_exact_in_ep_0_and_1:
            return True
        return False

    def _known_gate_data(self):
        return self._exact_gate_locations[0]

    def _pack_advise(self, profile):
        # in the case of RECKLESS, return our estimate of what the actual pose of the gates is
        if profile == RiskProfile.RECKLESS:
            return profile, self._known_gate_data()
        return profile, {}

    @staticmethod
    def _gate_data_is_eq(gate_data_1, gate_data_2):
        error_threshold = 0.005  # arbitrary equality threshold
        if len(gate_data_1) != len(gate_data_2):
            print(
                "Different gate data! Did we crash somewhere along the path? We have incomplete data!")
            return False
        for id, pos_1 in gate_data_1.items():
            if id not in gate_data_2:
                print(
                    "Different gate data! Did we crash somewhere along the path? We have incomplete data!")
                return False
            pos_2 = gate_data_2[id]
            np_pos_1 = np.array(pos_1[0:3])
            np_pos_2 = np.array(pos_2[0:3])
            np_error = np_pos_1 - np_pos_2
            error = np.sqrt(np.dot(np_error, np_error))
            if error > error_threshold:
                return False
        return True


if __name__ == '__main__':

    delta = 0.01

    gate_data_1 = {
        1: [1, 2, 3, 0, 0, 0, 0],
        2: [1, 2, 3, 0, 0, 0, 0],
        3: [1, 2, 3, 0, 0, 0, 0],
    }

    gate_data_2 = {
        1: [1, 2, 3, 0, 0, 0, 0],
        2: [1, 2 + delta, 3, 0, 0, 0, 0],
        3: [1, 2, 3, 0, 0, 0, 0],
    }

    gate_data_3 = {
        1: [1, 2, 3, 0, 0, 0, 0],
        2: [1, 2, 3, 0, 0, 0, 0],
        3: [1, 2, 3 + delta, 0, 0, 0, 0],
    }

    gate_data_4 = {
        1: [1, 2, 3, 0, 0, 0, 0],
        2: [1, 2, 3, 0, 0, 0, 0],
        3: [1, 2, 3, 0, 0, 0, 0],
    }

    #
    # we need this function to test the rest, so it gets special treatment
    #
    assert (RiskAdviser._gate_data_is_eq(gate_data_1, gate_data_1) == True)
    assert (RiskAdviser._gate_data_is_eq(gate_data_1, gate_data_2) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_1, gate_data_3) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_1, gate_data_4) == True)

    assert (RiskAdviser._gate_data_is_eq(gate_data_2, gate_data_1) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_2, gate_data_2) == True)
    assert (RiskAdviser._gate_data_is_eq(gate_data_2, gate_data_3) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_2, gate_data_4) == False)

    assert (RiskAdviser._gate_data_is_eq(gate_data_3, gate_data_1) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_3, gate_data_2) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_3, gate_data_3) == True)
    assert (RiskAdviser._gate_data_is_eq(gate_data_3, gate_data_4) == False)

    assert (RiskAdviser._gate_data_is_eq(gate_data_4, gate_data_1) == True)
    assert (RiskAdviser._gate_data_is_eq(gate_data_4, gate_data_2) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_4, gate_data_3) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_4, gate_data_4) == True)

    assert (RiskAdviser._gate_data_is_eq(gate_data_4, {}) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_4, {}) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_4, {}) == False)
    assert (RiskAdviser._gate_data_is_eq(gate_data_4, {}) == False)

    def run_test_case(episode_results):
        uut = RiskAdviser()
        advice = []
        for results in episode_results:
            advice.append(uut.episode_advice())
            uut.episode_results(*results)
        return advice

    def test_case_0_and_1():
        # vanilla level0
        results = run_test_case(
            [
                (True, gate_data_1, gate_data_1),
                (True, gate_data_1, gate_data_1),
                (True, gate_data_1, gate_data_1),
                (True, gate_data_1, gate_data_1),
            ]
        )
        advice, g_hints = zip(*results)
        assert (advice[0] == RiskProfile.CONSERVATIVE)
        assert (advice[1] == RiskProfile.CONSERVATIVE)
        assert (advice[2] == RiskProfile.RECKLESS)
        assert (advice[3] == RiskProfile.RECKLESS)
        assert (RiskAdviser._gate_data_is_eq(g_hints[0], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[1], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[2], gate_data_1) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[3], gate_data_1) == True)
    test_case_0_and_1()

    def test_case_0_and_1_in_bad_day():
        # vanilla level0
        results = run_test_case(
            [
                (False, gate_data_1, gate_data_1),
                (False, gate_data_1, gate_data_1),
                (False, gate_data_1, gate_data_1),
                (False, gate_data_1, gate_data_1),
            ]
        )
        advice, g_hints = zip(*results)
        assert (advice[0] == RiskProfile.CONSERVATIVE)
        assert (advice[1] == RiskProfile.CONSERVATIVE)
        assert (advice[2] == RiskProfile.RECKLESS)
        assert (advice[3] == RiskProfile.CONSERVATIVE)
        assert (RiskAdviser._gate_data_is_eq(g_hints[0], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[1], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[2], gate_data_1) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[3], {}) == True)
    test_case_0_and_1_in_bad_day()

    def test_case_level2():
        results = run_test_case(
            [
                (True, gate_data_1, gate_data_2),
                (True, gate_data_1, gate_data_2),
                (True, gate_data_1, gate_data_2),
                (True, gate_data_1, gate_data_2),
            ]
        )
        advice, g_hints = zip(*results)
        assert (advice[0] == RiskProfile.CONSERVATIVE)
        assert (advice[1] == RiskProfile.CONSERVATIVE)
        assert (advice[2] == RiskProfile.RECKLESS)
        assert (advice[3] == RiskProfile.RECKLESS)
        assert (RiskAdviser._gate_data_is_eq(g_hints[0], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[1], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[2], gate_data_2) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[3], gate_data_2) == True)
    test_case_level2()

    def test_case_level2_with_crash_on_ep_3():
        results = run_test_case(
            [
                (True, gate_data_1, gate_data_2),
                (True, gate_data_1, gate_data_2),
                (False, gate_data_1, gate_data_2),
                (True, gate_data_1, gate_data_2),
            ]
        )
        advice, g_hints = zip(*results)
        assert (advice[0] == RiskProfile.CONSERVATIVE)
        assert (advice[1] == RiskProfile.CONSERVATIVE)
        assert (advice[2] == RiskProfile.RECKLESS)
        assert (advice[3] == RiskProfile.CONSERVATIVE)
        assert (RiskAdviser._gate_data_is_eq(g_hints[0], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[1], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[2], gate_data_2) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[3], {}) == True)
    test_case_level2_with_crash_on_ep_3()

    def test_case_level3():
        results = run_test_case(
            [
                (True, gate_data_1, gate_data_2),
                (True, gate_data_1, gate_data_3),
                (True, gate_data_1, gate_data_4),
                (True, gate_data_1, gate_data_3),
            ]
        )
        advice, g_hints = zip(*results)
        assert (advice[0] == RiskProfile.CONSERVATIVE)
        assert (advice[1] == RiskProfile.CONSERVATIVE)
        assert (advice[2] == RiskProfile.CONSERVATIVE)
        assert (advice[3] == RiskProfile.CONSERVATIVE)
        assert (RiskAdviser._gate_data_is_eq(g_hints[0], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[1], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[2], {}) == True)
        assert (RiskAdviser._gate_data_is_eq(g_hints[3], {}) == True)
    test_case_level3()

    print("Tests are OK!!!!")
