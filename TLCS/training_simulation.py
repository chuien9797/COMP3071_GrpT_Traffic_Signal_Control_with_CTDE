import os
import timeit
import traci
import numpy as np
import random
from datetime import datetime
import math
import matplotlib.pyplot as plt

from emergency_handler import check_emergency
import intersection_config as int_config

# Constants for fault injection and fault curriculum settings
RECOVERY_DELAY = 15               # Steps to recover faulty signals
FAULT_REWARD_SCALE = 0.5          # Scale reward if fault occurs
EPISODE_FAULT_START = 25          # Start injecting faults only from this episode


class Simulation:
    def __init__(self,
                 Model,          # PPO model wrapper instance
                 sumo_cmd,
                 gamma,
                 max_steps,
                 green_duration,
                 yellow_duration,
                 num_actions,
                 training_epochs,  # For logging/number of update epochs (used in PPO update below)
                 intersection_type="cross",
                 ppo_clip_ratio=0.2,
                 ppo_update_epochs=10,
                 gae_lambda=0.95,
                 signal_fault_prob=0.1):
        self._Model = Model
        self._sumo_cmd = sumo_cmd
        self._gamma = gamma
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_actions = num_actions
        self._training_epochs = training_epochs
        self.intersection_type = intersection_type

        # PPO trajectory buffer for on-policy data
        self.trajectory = []

        # PPO hyperparameters
        self.ppo_clip_ratio = ppo_clip_ratio
        self.ppo_update_epochs = ppo_update_epochs
        self.gae_lambda = gae_lambda

        # Simulation statistics and logs
        self._step = 0
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []

        # Fault injection and emergency handling
        self.faulty_lights = set()
        self.fault_injected_this_episode = False
        self.skip_fault_this_episode = False
        self._green_durations_log = []
        self.fault_details = []   # List of tuples: (step, tlid, original state, modified state)
        self._action_counts = np.zeros(self._num_actions, dtype=int)
        self._teleport_count = 0
        self._sum_reward = 0.0
        self._sum_queue_length = 0.0
        self._sum_waiting_time = 0.0
        self._waiting_times = {}
        self.signal_fault_prob = signal_fault_prob
        self.manual_override = False
        self.recovery_queue = {}  # Holds faults scheduled for recovery

        # Load intersection configuration – if not found, raise an error.
        if self.intersection_type not in int_config.INTERSECTION_CONFIGS:
            raise ValueError("Intersection type '{}' not found in config.".format(self.intersection_type))
        self.int_conf = int_config.INTERSECTION_CONFIGS[self.intersection_type]

    def run(self, episode):
        """
        Runs one full PPO simulation episode:
          - Collects an on-policy trajectory: (state, action, log_prob, value, reward)
          - At the end of the episode, computes returns and advantages using GAE.
          - Then updates the PPO model.
        """
        # Reset trajectory and simulation variables for this episode
        self.trajectory = []
        self._step = 0
        self._sum_reward = 0.0
        self._sum_queue_length = 0.0
        self._sum_waiting_time = 0.0
        self._waiting_times = {}
        self.fault_injected_this_episode = False
        self.skip_fault_this_episode = (episode < EPISODE_FAULT_START) or (random.random() < 0.5)

        start_time = timeit.default_timer()
        traci.start(self._sumo_cmd)
        print("Simulating Episode {} on {}".format(episode, self.intersection_type))

        # Get the initial state from the simulation.
        state = self._get_state()

        # Main simulation loop: run until max_steps reached.
        while self._step < self._max_steps:
            if check_emergency(self):
                continue

            # In PPO, actions are sampled stochastically from the policy (no epsilon-greedy)
            action, log_prob, value = self._Model.get_action(state)
            self._action_counts[action] += 1

            # If not the first step, simulate a yellow phase (if changing phases)
            if self._step != 0:
                self._set_yellow_phase(action)
                self._simulate(self._yellow_duration)

            # Set the green phase for the chosen action.
            self._set_green_phase(action)
            # Compute an adaptive green duration (this can be tuned per your simulation needs)
            adaptive_green = self._compute_adaptive_green_duration(state)
            self._green_durations_log.append(adaptive_green)
            # Uncomment below for debugging if desired:
            # print("[Step {}] Adaptive green duration: {}".format(self._step, adaptive_green))
            self._simulate(adaptive_green)

            # Get the next state and compute waiting time (used for reward)
            next_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            # Example reward: negative total waiting time (modify if needed)
            reward = -current_total_wait
            self._sum_reward += reward

            # Save the transition in the trajectory buffer
            self.trajectory.append((state, action, log_prob, value, reward))

            state = next_state

        traci.close()
        sim_time = round(timeit.default_timer() - start_time, 1)

        # Process the trajectory to compute returns and advantages using GAE.
        states, actions, log_probs, values, returns, advantages = self._process_trajectory()

        # Perform PPO updates using the collected trajectory.
        train_start = timeit.default_timer()
        self._Model.train(states, actions, log_probs, returns, advantages,
                          clip_ratio=self.ppo_clip_ratio,
                          update_epochs=self.ppo_update_epochs)
        train_time = round(timeit.default_timer() - train_start, 1)

        self._reward_store.append(self._sum_reward)
        self._save_episode_stats()
        self._write_summary_log(episode, sim_time)

        return sim_time, train_time, self._sum_reward

    def _process_trajectory(self):
        """
        Processes the collected trajectory.
        Computes cumulative discounted returns and advantages using Generalized Advantage Estimation (GAE).
        Returns:
            states: np.array of shape (N, num_lanes, lane_feature_dim)
            actions: np.array of shape (N,)
            log_probs: np.array of shape (N,)
            values: np.array of shape (N,)
            returns: np.array of shape (N,)
            advantages: np.array of shape (N,)
        """
        trajectory = self.trajectory
        N = len(trajectory)
        states, actions, log_probs, values, rewards = zip(*trajectory)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        log_probs = np.array(log_probs, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)

        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        running_return = 0
        running_advantage = 0
        next_value = 0

        # Calculate returns and advantage estimates in reverse order.
        for t in reversed(range(N)):
            running_return = rewards[t] + self._gamma * running_return
            td_error = rewards[t] + self._gamma * next_value - values[t]
            running_advantage = td_error + self._gamma * self.gae_lambda * running_advantage
            returns[t] = running_return
            advantages[t] = running_advantage
            next_value = values[t]

        # Normalize advantages (commonly improves training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return states, actions, log_probs, values, returns, advantages

    def _write_summary_log(self, episode, sim_time):
        """
        Writes a summary log for the episode to a file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_filename = "logs/episode_{}_summary.log".format(episode)
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)
        with open(log_filename, "w", encoding="utf-8") as f:
            f.write("Timestamp: {}\n".format(timestamp))
            f.write("Intersection: {}\n".format(self.intersection_type))
            f.write("Total reward: {:.2f}\n".format(self._sum_reward))
            f.write("Simulation duration: {}s\n".format(sim_time))
            avg_queue = self._sum_queue_length / self._max_steps if self._max_steps > 0 else 0
            f.write("Avg queue length: {:.2f}\n".format(avg_queue))
            f.write("Cumulative wait time: {:.2f}\n".format(self._sum_waiting_time))
            f.write("Fault injected: {}\n".format("Yes" if self.fault_injected_this_episode else "No"))
            if self.fault_details:
                f.write("\nFault Details:\n")
                for step, tlid, original, modified in self.fault_details:
                    f.write("Step {} | TLID: {} | Orig: {} -> Mod: {}\n".format(step, tlid, original, modified))
            f.write("\nAction Distribution:\n")
            for i, count in enumerate(self._action_counts):
                f.write("Action {}: {} times\n".format(i, count))

    def _compute_adaptive_green_duration(self, state):
        """
        Computes an adaptive green phase duration based on state information.
        Example: uses average waiting time and queue lengths.
        """
        avg_wait = np.mean(state[:, 1])
        queue_length = np.sum(state[:, 3])
        base = self._green_duration
        wait_factor = int(avg_wait // 2)
        queue_factor = int(queue_length // 5)
        adaptive_extension = min(wait_factor + queue_factor, 10)
        return base + adaptive_extension

    def _simulate(self, steps_todo):
        """
        Advances the SUMO simulation by a specified number of steps.
        Also collects statistics such as queue lengths and waiting times.
        """
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            self._inject_signal_faults()
            self._recover_faults_if_due()
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length  # This can be replaced or refined with actual waiting times
            self._teleport_count += traci.simulation.getStartingTeleportNumber()

    def _inject_signal_faults(self):
        """
        Randomly injects faults into traffic signals.
        Uses the deprecated getCompleteRedYellowGreenDefinition for compatibility with your configuration.
        """
        self.manual_override = False
        if self.skip_fault_this_episode or self.fault_injected_this_episode:
            return

        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            # Here we use the original method for cross intersections; you may later update to getAllProgramLogics if needed.
            try:
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tlid)[0]
            except Exception as e:
                print(f"Warning: getCompleteRedYellowGreenDefinition failed for {tlid}: {e}")
                continue

            current_phase = traci.trafficlight.getPhase(tlid)
            current_state = logic.phases[current_phase].state

            new_state = list(current_state)
            flipped_indices = []
            g_indices = [i for i, s in enumerate(current_state) if s == 'G']
            if g_indices:
                flip_idx = random.choice(g_indices)
                new_state[flip_idx] = 'r'
                flipped_indices.append(flip_idx)
            new_state_str = ''.join(new_state)
            if new_state_str != current_state:
                try:
                    traci.trafficlight.setRedYellowGreenState(tlid, new_state_str)
                except Exception as e:
                    print("Failed to set fault state for {}: {}".format(tlid, e))
                self.manual_override = True
                self.fault_injected_this_episode = True
                self.fault_details.append((self._step, tlid, current_state, new_state_str))
                print("[Step {}] Fault injected at TL={}, phase={}, flipped indices: {}".format(self._step, tlid, current_phase, flipped_indices))
                return

    def _recover_faults_if_due(self):
        """
        Recovers any traffic signal faults if the designated recovery delay has passed.
        """
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            key = (tlid, self._step)
            if key in self.recovery_queue:
                original_state = self.recovery_queue[key]
                try:
                    traci.trafficlight.setRedYellowGreenState(tlid, original_state)
                    print("[Step {}] Signal recovered at TL={}".format(self._step, tlid))
                    del self.recovery_queue[key]
                    self.manual_override = False
                except traci.exceptions.TraCIException:
                    pass

    def _collect_waiting_times(self):
        """
        Collects waiting times for vehicles on incoming lanes.
        """
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            lane_id = traci.vehicle.getLaneID(car_id)
            if lane_id in incoming_lane_ids:
                self._waiting_times[car_id] = traci.vehicle.getAccumulatedWaitingTime(car_id)
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        return float(sum(self._waiting_times.values()))


    def _get_queue_length(self):
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lane_ids)

    def _set_green_phase(self, action_number):
        """
        Sets the traffic light to the green phase associated with the given action.
        Uses getAllProgramLogics to obtain the traffic light configuration so that even complex maps are handled.
        """
        if self.manual_override:
            return

        phase_map = self.int_conf.get("phase_mapping", {})
        if action_number not in phase_map:
            print(f"Action number {action_number} not found in phase mapping for green phase.")
            return
        green_phase = phase_map[action_number].get("green")
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tlid)
                if not logics:
                    print(f"No traffic light logics available for {tlid}.")
                    continue
                num_phases = len(logics[0].phases)
                if green_phase >= num_phases:
                    print(f"Invalid green phase {green_phase} for TL {tlid} (only {num_phases} phases available).")
                    continue
                # Reset to default logic before setting the phase.
                traci.trafficlight.setProgram(tlid, "0")
                traci.trafficlight.setPhase(tlid, green_phase)
            except Exception as e:
                print(f"Error setting green phase for TL {tlid}: {e}")

    def _set_yellow_phase(self, action_number):
        """
        Sets the traffic light to the yellow phase associated with the given action.
        Uses getAllProgramLogics and checks that the selected yellow phase is valid.
        """
        phase_map = self.int_conf.get("phase_mapping", {})
        if action_number not in phase_map:
            print(f"Action number {action_number} not found in phase mapping for yellow phase.")
            return
        yellow_phase = phase_map[action_number].get("yellow")
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tlid)
                if not logics:
                    print(f"No traffic light logics available for {tlid}.")
                    continue
                num_phases = len(logics[0].phases)
                if yellow_phase >= num_phases:
                    print(f"Invalid yellow phase {yellow_phase} for TL {tlid} (only {num_phases} phases available).")
                    continue
                # Reset program to default and then set the yellow phase.
                traci.trafficlight.setProgram(tlid, "0")
                traci.trafficlight.setPhase(tlid, yellow_phase)
            except Exception as e:
                print(f"Error setting yellow phase for TL {tlid}: {e}")

    def _get_state(self):
        """
        Retrieves the current state from the simulation.
        For each incoming lane, the state vector includes:
          - Number of vehicles on the lane.
          - Total waiting time on the lane.
          - Emergency flag (1.0 if at least one emergency vehicle is present).
          - Number of halting vehicles.
          - The current phase of the first traffic light (if available).
          - Flag indicating whether the lane is controlled by a traffic light that is red.
          - A one-hot encoding for the intersection type.
        This version uses getAllProgramLogics for extra robustness.
        """
        incoming_lanes = self.int_conf.get("incoming_lanes", {})
        lane_order = []
        for group in sorted(incoming_lanes.keys()):
            lane_order.extend(incoming_lanes[group])
        num_lanes = len(lane_order)
        lane_feature_dim = 9
        lane_features = np.zeros((num_lanes, lane_feature_dim), dtype=np.float32)

        intersection_encoding = {
            "cross": [1.0, 0.0, 0.0],
            "roundabout": [0.0, 1.0, 0.0],
            "T_intersection": [0.0, 0.0, 1.0]
        }
        type_vector = intersection_encoding.get(self.intersection_type, [0.0, 0.0, 0.0])

        for i, lane_id in enumerate(lane_order):
            try:
                lane_features[i, 0] = traci.lane.getLastStepVehicleNumber(lane_id)
                lane_features[i, 1] = traci.lane.getWaitingTime(lane_id)
            except Exception as e:
                print(f"Error retrieving basic features for lane {lane_id}: {e}")
            flag = 0
            try:
                for car_id in traci.lane.getLastStepVehicleIDs(lane_id):
                    if traci.vehicle.getTypeID(car_id) == "emergency":
                        flag = 1
                        break
            except Exception as e:
                print(f"Error retrieving vehicle IDs for lane {lane_id}: {e}")
            lane_features[i, 2] = float(flag)
            try:
                lane_features[i, 3] = traci.lane.getLastStepHaltingNumber(lane_id)
            except Exception as e:
                print(f"Error retrieving halting count for lane {lane_id}: {e}")
            tl_ids = self.int_conf.get("traffic_light_ids", [])
            phase_val = 0.0
            if tl_ids:
                try:
                    logics = traci.trafficlight.getAllProgramLogics(tl_ids[0])
                    if logics:
                        current_phase = traci.trafficlight.getPhase(tl_ids[0])
                        phase_val = float(current_phase)
                except Exception as e:
                    print(f"Error retrieving phase for TL {tl_ids[0]}: {e}")
            lane_features[i, 4] = phase_val
            controlled_by_faulty_signal = 0.0
            if tl_ids:
                try:
                    logics = traci.trafficlight.getAllProgramLogics(tl_ids[0])
                    if logics:
                        current_phase = traci.trafficlight.getPhase(tl_ids[0])
                        phase_state = logics[0].phases[current_phase].state
                        # Check if this lane is controlled by a red signal.
                        try:
                            connections = traci.trafficlight.getControlledLanes(tl_ids[0])
                            if lane_id in connections:
                                idx = connections.index(lane_id)
                                if phase_state[idx] == 'r':
                                    controlled_by_faulty_signal = 1.0
                        except Exception as e:
                            print(f"Error retrieving controlled lanes for TL {tl_ids[0]}: {e}")
                except Exception as e:
                    print(f"Error retrieving traffic light state for TL {tl_ids[0]}: {e}")
            lane_features[i, 5] = controlled_by_faulty_signal
            # Append the one-hot encoding of the intersection type.
            lane_features[i, 6:9] = type_vector

        return lane_features

    def _pad_states(self, state_list):
        """
        Pads a list of state arrays so that all have the same number of lanes.
        Useful for batching states when the number of lanes may vary.
        """
        lane_feature_dim = state_list[0].shape[1]
        max_lanes = max(state.shape[0] for state in state_list)
        padded = []
        for state in state_list:
            pad_size = max_lanes - state.shape[0]
            if pad_size > 0:
                pad_width = ((0, pad_size), (0, 0))
                padded_state = np.pad(state, pad_width=pad_width, mode='constant')
            else:
                padded_state = state
            padded.append(padded_state)
        return np.array(padded, dtype=np.float32)

    def _save_episode_stats(self):
        self._reward_store.append(self._sum_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    def analyze_results(self):
        """
        Plots and displays various training statistics.
        """
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self._reward_store)
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.subplot(1, 3, 2)
        plt.plot(self._avg_queue_length_store)
        plt.title("Average Queue Length per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Queue Length")
        plt.subplot(1, 3, 3)
        plt.plot(self._cumulative_wait_store)
        plt.title("Cumulative Waiting Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Waiting Time")
        plt.show()
        print("Final Faulty Lights:", self.faulty_lights)
        if self._green_durations_log:
            plt.figure(figsize=(10, 4))
            plt.plot(self._green_durations_log)
            plt.title("Adaptive Green Duration Over Time")
            plt.xlabel("Step")
            plt.ylabel("Green Duration")
            plt.grid(True)
            plt.show()

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store


