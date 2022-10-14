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
from scipy. interpolate import CubicSpline

from competition_utils import Command
from stage_action_take_off import StageActionTakeOff
from stage_action_land import StageActionLand
from stage_action_mpcc import StageActionMPCC
from stage_action_setpoint_stop import StageActionSetPointStop
from stage_action_hardbrake import StageActionHardBrake
from stage_action_finished import StageActionFinished
from stage_sequencer import StageSequencer
from ek_controller_config import EkControllerConfig
from planning import plan_time_optimal_trajectory_through_gates, State, Limits, Cylinder, to_pose
from rate_estimator import RateEstimator
from risk_adviser import RiskAdviser, RiskProfile


class EkControllerImpl:

    def __init__(self,
                 config: EkControllerConfig,
                 ):
        self._arc_parametrization_tolerance = 1e-4
        self._evenly_spaced_segments = 90
        self._take_off_height = 0.4
        self._gate_waypoint_offset = 0.2

        self._config = config

        self._dt = self._config.ctrl_timestep

        self._rate_estimator = RateEstimator(self._dt)

        # No path recompilation for us this round
        self._risk_adviser = RiskAdviser(forced_conservative_mode = True)

        self._start_pos = (
            self._config.initial_obs[0],
            self._config.initial_obs[2],
            self._config.initial_obs[4]
        )
        self._start_yaw = self._config.initial_obs[8]

        self._goal_pos = (
            self._config.x_reference[0],
            self._config.x_reference[2],
            self._config.x_reference[4]
        )
        self._goal_yaw = self._start_yaw

        self._flight_plans_cache = {}

        self.reset_episode()

    def _configure_mode(self, risk_profile, gate_poses):
        if risk_profile not in self._flight_plans_cache:
            print("Optimizing new flight path")
            waypoints_arg, waypoints_pos, landmarks = self._calculate_waypoints(
                gate_poses)
            ref_x, ref_y, ref_z = self._calculate_reference_trajectory(
                waypoints_pos, waypoints_arg)
            sequencer = self._build_flight_sequence(
                waypoints_pos=waypoints_pos, waypoints_arg=waypoints_arg, landmarks=landmarks)
            self._flight_plans_cache[risk_profile] = (
                waypoints_arg, waypoints_pos, landmarks, ref_x, ref_y, ref_z, sequencer)
        else:
            print("Using cached risk profile")

        self._waypoints_arg, self._waypoints_pos, self._landmarks, self._ref_x, self._ref_y, self._ref_z, self._stage_sequencer = self._flight_plans_cache[
            risk_profile]

    def get_waypoints(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._waypoints_arg, self._waypoints_pos

    def get_reference_trajectory(self) -> Tuple:
        return self._ref_x, self._ref_y, self._ref_z

    def cmd_firmware_impl(self,
                          time: float,
                          info: Dict[str, Any],
                          obs: list,
                          iteration: int
                          ) -> Tuple[Command, list]:
        corrections = self._process_gate_correction_data(info)
        cur_pos, cur_rpy = self._split_observations(obs)
        est_vel, est_pqr = self._estimate_missing_states(cur_pos, cur_rpy)
        _, command_type, args = self._stage_sequencer.run(
            iteration, cur_pos, est_vel, cur_rpy, est_pqr, corrections)
        return command_type, args

    def reset_episode(self):
        self._gate_nominal_poses = {}
        self._gate_corrected_poses = {}
        self._prev_gate_id = None
        self._next_gate_id = None

        risk_profile, most_likely_gate_poses = self._risk_adviser.episode_advice()

        if risk_profile == RiskProfile.RECKLESS:
            self._gate_nominal_poses = most_likely_gate_poses
            self._gate_corrected_poses = most_likely_gate_poses
            self._configure_mode(risk_profile, most_likely_gate_poses.values())
        else:
            self._configure_mode(
                risk_profile, self._config.nominal_gates_pose_and_type)

        self._rate_estimator.reset()
        self._stage_sequencer.reset()

    def learn_from_episode(self, episode_info):
        episode_completed = episode_info['task_completed']
        self._risk_adviser.episode_results(
            episode_completed, self._gate_nominal_poses, self._gate_corrected_poses)

    def _split_observations(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        current_pos = np.array([obs[0], obs[2], obs[4]])
        current_rpy = np.array(obs[6:9])
        return current_pos, current_rpy

    def _estimate_missing_states(self, current_pos: np.ndarray, current_rpy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        estimated_velocity, estimated_body_rates = self._rate_estimator.estimate(
            current_pos, current_rpy)
        return estimated_velocity, estimated_body_rates

    def _calculate_reference_trajectory(self, waypoints_pos, waypoints_arg):
        fx = CubicSpline(waypoints_arg, waypoints_pos[:, 0])
        fy = CubicSpline(waypoints_arg, waypoints_pos[:, 1])
        fz = CubicSpline(waypoints_arg, waypoints_pos[:, 2])
        t_scaled = np.linspace(
            waypoints_arg[0],
            waypoints_arg[-1],
            100)
        ref_x = fx(t_scaled)
        ref_y = fy(t_scaled)
        ref_z = fz(t_scaled)
        return ref_x, ref_y, ref_z

    def _calculate_gate_pos(self, x, y, yaw, type):
        tall_gate_height = self._config.gate_dimensions["tall"]["height"]
        low_gate_height = self._config.gate_dimensions["low"]["height"]
        height = tall_gate_height if type == 0 else low_gate_height
        return (x, y, height, 0, 0, yaw)

    def _calculate_waypoints(self, gate_poses) -> Tuple[np.ndarray, np.ndarray]:
        # Determine waypoints
        air_start_pos = (
            self._start_pos[0], self._start_pos[1], self._take_off_height)

        assert self._config.gate_dimensions["tall"]["shape"] == "square"
        assert self._config.gate_dimensions["low"]["shape"] == "square"

        rotation_offset = 1.57
        gates = []
        for gate in gate_poses:
            gates.append(
                self._calculate_gate_pos(
                    x=gate[0],
                    y=gate[1],
                    yaw=gate[5] + rotation_offset,
                    type=gate[6]),
            )

        assert self._config.obstacle_dimensions["shape"] == "cylinder"
        obstacle_height = self._config.obstacle_dimensions["height"]
        obstacle_radius = self._config.obstacle_dimensions["radius"]

        obstacles = []
        for obstacle in self._config.nominal_obstacles_pos:
            obstacles.append(
                Cylinder(obstacle[0:3], radius=obstacle_radius, height=obstacle_height))

        print("Calculating best path through gates, may take a few seconds...")

        path = plan_time_optimal_trajectory_through_gates(
            initial_state=State(
                position=np.array(air_start_pos),
                velocity=np.zeros(3)),
            final_state=State(
                position=np.array(self._goal_pos),
                velocity=np.zeros(3)),
            gate_poses=list(map(to_pose, gates)),
            acceleration_limits=Limits(
                lower=-1 * np.ones(3),
                upper=1 * np.ones(3),
            ),
            velocity_limits=Limits(
                lower=np.array([0.05, -np.pi/6, -np.pi/6]),
                upper=np.array([2.00, np.pi/6, np.pi/6]),
            ),
            num_cone_samples=3,
            obstacles=obstacles,
        )

        waypoint_pos = []
        waypoint_arg = []
        waypoint_marks = []
        for length, position, landmarks in path.evenly_spaced_points(
            self._evenly_spaced_segments, self._arc_parametrization_tolerance
        ):
            waypoint_pos.append(position)
            waypoint_arg.append(length)
            waypoint_marks.append(landmarks)

        return np.array(waypoint_arg), np.array(waypoint_pos), waypoint_marks

    def _process_gate_correction_data(self, info):
        try:
            gate_id = info['current_target_gate_id']
            gate_type = info['current_target_gate_type']
            gate_in_range = info['current_target_gate_in_range']
            gate_pos = info['current_target_gate_pos']

            if gate_id != self._next_gate_id:
                self._prev_gate_id = self._next_gate_id
                self._next_gate_id = gate_id

            corrected_gate_pose = self._calculate_gate_pos(
                x=gate_pos[0],
                y=gate_pos[1],
                yaw=gate_pos[5],
                type=gate_type)

            if gate_id not in self._gate_nominal_poses and not gate_in_range:
                self._gate_nominal_poses[gate_id] = \
                    *corrected_gate_pose, gate_type
                print("Received a nominal gate pose for id {}: {}".format(
                    gate_id, corrected_gate_pose))

            if gate_id not in self._gate_corrected_poses and gate_in_range:
                self._gate_corrected_poses[gate_id] = \
                    *corrected_gate_pose, gate_type
                print("Received a corrected gate pose for id {}: {}".format(
                    gate_id, corrected_gate_pose))
        except:
            pass

        corrections = {}

        corrections["prev_gate_location"] = self._get_location_for(
            self._prev_gate_id)
        corrections["prev_gate_correction"] = self._get_correction_for(
            self._prev_gate_id)

        corrections["next_gate_location"] = self._get_location_for(
            self._next_gate_id)
        corrections["next_gate_correction"] = self._get_correction_for(
            self._next_gate_id)

        corrections['next_gate_location_is_fuzzy'] = self._gate_location_is_fuzzy(
            self._next_gate_id)

        return corrections

    def _gate_location_is_fuzzy(self, id):
        return id not in self._gate_corrected_poses

    def _get_location_for(self, id):
        """if the nominal pose is uknown, return some position far away"""
        return self._gate_nominal_poses[id][0:3] if id in self._gate_nominal_poses else np.ones(3) * 99

    def _get_correction_for(self, id):
        """if the nominal pose is uknown, return all zeros for the correction"""
        if id in self._gate_nominal_poses and id in self._gate_corrected_poses:
            nominal = self._gate_nominal_poses[id]
            actual = self._gate_corrected_poses[id]
            gate_correction = np.array(
                actual[0:3]) - np.array(nominal[0:3])
            return gate_correction
        return np.zeros(3)

    def _build_flight_sequence(self, waypoints_arg, waypoints_pos, landmarks):
        take_off_duration = 1.0
        take_off_height = self._take_off_height

        # Duration time to stop the drone and stabilize it in the goal position
        # The evaluation script will stop execution once it detects the drone
        # has been stable for 2 seconds
        stop_duration = 3.0
        land_duration = 2.0

        # Build stage sequencer
        stages = [
            StageActionTakeOff(
                dt=self._dt,
                duration=take_off_duration,
                height=take_off_height),
            StageActionMPCC(
                dt=self._dt,
                mass=self._config.mass,
                ixx=self._config.ixx,
                iyy=self._config.iyy,
                izz=self._config.izz,
                mpcc_horizon_len=10,
                waypoints_arg=waypoints_arg,
                waypoints_pos=waypoints_pos,
                waypoints_marks=landmarks,
            ),
            StageActionHardBrake(
                dt=self._dt,
                goal_pose=waypoints_pos[-1],
                duration=stop_duration),
            StageActionSetPointStop(),
            # If we get to this point the episode will fail. Try to land.
            StageActionLand(
                dt=self._dt,
                duration=land_duration,
                # Usually should be a few centimeters above the initial position
                # to ensure that the controller does not try to penetrate the floor
                # if the mocap coordinate origin is not perfect.
                height=0.02),
            StageActionFinished(),
        ]
        return StageSequencer(stages)
