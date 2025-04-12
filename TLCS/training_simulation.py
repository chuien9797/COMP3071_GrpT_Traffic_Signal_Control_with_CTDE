"""
Adaptive Traffic Light Simulation with Multi-Agent Support.
This Simulation class has been modified to operate with two agents.
It auto-scans the intersection configuration to determine which edges and lanes
are assigned to each agent. The class includes methods for state extraction, reward
computation, action selection (via REST calls), phase control, and experience replay.
"""

import traci
import numpy as np
import random
import timeit
import os
import matplotlib.pyplot as plt
from datetime import datetime
import requests

from emergency_handler import check_emergency
import intersection_config as int_config

# Global constants
RECOVERY_DELAY = 15               # Steps to recover faulty signals
FAULT_REWARD_SCALE = 0.5          # Scale reward if a fault is injected
EPISODE_FAULT_START = 25          # Start injecting faults from episode 25

# Phase codes (as defined in your environment)
PHASE_NS_GREEN = 0    # action 0
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2   # action 1
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4    # action 2
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6   # action 3
PHASE_EWL_YELLOW = 7

class Simulation:
    def __init__(
        self,
        Model,
        TargetModel,
        Memory,
        TrafficGen,
        sumo_cmd,
        gamma,
        max_steps,
        green_duration,
        yellow_duration,
        num_states,
        training_epochs,
        intersection_type="cross",
        signal_fault_prob=0.1,
    ):
        self._Model = Model
        self._TargetModel = TargetModel
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._training_epochs = training_epochs

        # Statistics for each episode (combined and per agent)
        self._reward_store = []              # Combined rewards (agent1 + agent2)
        self._reward_store_a1 = []           # Agent one rewards
        self._reward_store_a2 = []           # Agent two rewards
        self._cumulative_wait_store = []     # Combined cumulative waiting time
        self._cumulative_wait_store_a1 = []
        self._cumulative_wait_store_a2 = []
        self._avg_queue_length_store = []    # Combined average queue length
        self._avg_queue_length_store_a1 = []
        self._avg_queue_length_store_a2 = []
        self._q_loss_log = []                # Training loss log

        # Accumulators for multi-agent metrics
        self._sum_neg_reward_one = 0
        self._sum_neg_reward_two = 0
        self._sum_queue_length = 0
        self._sum_queue_length_a1 = 0
        self._sum_queue_length_a2 = 0
        self._sum_waiting_time = 0
        self._cumulative_waiting_time_agent_one = 0
        self._cumulative_waiting_time_agent_two = 0

        # Other internal trackers
        self._waiting_times = {}
        self.faulty_lights = set()
        self.fault_injected_this_episode = False
        self.skip_fault_this_episode = False
        self.manual_override = False
        self.recovery_queue = {}
        self._green_durations_log = []
        self.fault_details = []
        self._teleport_count = 0
        self._already_in = []  # For flow tracking

        # For state extraction: number of cells per agent.
        # Adjust this value based on your occupancy_grid settings.
        self._num_cells = 40  # Example: 40 cells per intersection

        self.intersection_type = intersection_type
        if self.intersection_type not in int_config.INTERSECTION_CONFIGS:
            raise ValueError(f"Intersection type '{self.intersection_type}' not found in config.")
        self.int_conf = int_config.INTERSECTION_CONFIGS[self.intersection_type]
        self._num_actions = len(self.int_conf["phase_mapping"])
        self._action_counts = np.zeros(self._num_actions, dtype=int)

    def run(self, episode, epsilon):
        os.makedirs("logs19", exist_ok=True)
        log_file = open(f"logs19/episode_{episode}.log", "w")

        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print(f"Simulating Episode {episode} on {self.intersection_type} (Multi-Agent)...")

        # Reset simulation variables
        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward_one = 0
        self._sum_neg_reward_two = 0
        self._sum_queue_length = 0
        self._sum_queue_length_a1 = 0
        self._sum_queue_length_a2 = 0
        self._sum_waiting_time = 0
        old_total_wait_one = 0
        old_total_wait_two = 0
        old_state_one = None
        old_state_two = None
        old_action_one = None
        old_action_two = None

        self.faulty_lights = set()
        self.fault_injected_this_episode = False
        self.skip_fault_this_episode = (episode < EPISODE_FAULT_START) or (random.random() < 0.5)

        start_time = timeit.default_timer()

        while self._step < self._max_steps:
            if check_emergency(self):
                continue

            # Get current state vectors for both agents.
            current_state_one, current_state_two = self._get_states_with_advanced_perception()

            # Optionally, append the other agent’s previous action if _num_states equals 321.
            if self._num_states == 321 and (old_action_one is not None and old_action_two is not None):
                current_state_one = np.append(current_state_one, old_action_two)
                current_state_two = np.append(current_state_two, old_action_one)

            # Compute rewards for each agent.
            current_total_wait_one = 0.2 * self._collect_waiting_times_first_intersection() + self._get_queue_length_intersection_one()
            reward_one = old_total_wait_one - current_total_wait_one
            current_total_wait_two = 0.2 * self._collect_waiting_times_second_intersection() + self._get_queue_length_intersection_two()
            reward_two = old_total_wait_two - current_total_wait_two

            # Mutual influence.
            reward_one += 0.5 * reward_two
            reward_two += 0.5 * reward_one

            # Accumulate waiting times.
            self._cumulative_waiting_time_agent_one += current_total_wait_one
            self._cumulative_waiting_time_agent_two += current_total_wait_two

            # If not the first step, add experience samples via REST.
            if self._step != 0:
                requests.post(
                    'http://127.0.0.1:5000/add_samples',
                    json={
                        'old_state_one': old_state_one.tolist(),
                        'old_state_two': old_state_two.tolist(),
                        'old_action_one': int(old_action_one),
                        'old_action_two': int(old_action_two),
                        'reward_one': reward_one,
                        'reward_two': reward_two,
                        'current_state_one': current_state_one.tolist(),
                        'current_state_two': current_state_two.tolist()
                    }
                )

            # Choose actions for both agents.
            action_one = self._choose_action(current_state_one, epsilon, 1)
            action_two = self._choose_action(current_state_two, epsilon, 2)

            # Manage yellow-phase transitions if needed.
            if self._step != 0 and old_action_one is not None and old_action_two is not None:
                if old_action_one != action_one and old_action_two != action_two:
                    self._set_yellow_phase(old_action_one)
                    self._set_yellow_phase_two(old_action_two)
                    self._simulate(self._yellow_duration)
                elif old_action_one != action_one:
                    self._set_yellow_phase(old_action_one)
                    self._simulate(self._yellow_duration)
                elif old_action_two != action_two:
                    self._set_yellow_phase_two(old_action_two)
                    self._simulate(self._yellow_duration)

            # Set green phases for both intersections.
            self._set_green_phase(action_one)
            self._set_green_phase_two(action_two)
            self._simulate(self._green_duration)

            # Update old state and action.
            old_state_one = current_state_one
            old_state_two = current_state_two
            old_action_one = action_one
            old_action_two = action_two
            old_total_wait_one = current_total_wait_one
            old_total_wait_two = current_total_wait_two

            if reward_one < 0:
                self._sum_neg_reward_one += reward_one
            if reward_two < 0:
                self._sum_neg_reward_two += reward_two

        # End of episode: record statistics and close simulation.
        self._save_episode_stats()
        total_episode_reward = self._sum_neg_reward_one + self._sum_neg_reward_two
        print("Total reward (combined):", total_episode_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        self._write_summary_log(episode, epsilon, simulation_time)
        print("Training phase skipped in Simulation.run (handled externally)")
        return simulation_time, 0

    def _write_summary_log(self, episode, epsilon, sim_time):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(f"logs19/episode_{episode}_summary.log", "w", encoding="utf-8") as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Intersection: {self.intersection_type}\n")
                total_reward = self._sum_neg_reward_one + self._sum_neg_reward_two
                f.write(f"Total reward (combined): {total_reward:.2f}\n")
                f.write(f"Agent 1 reward: {self._sum_neg_reward_one:.2f}\n")
                f.write(f"Agent 2 reward: {self._sum_neg_reward_two:.2f}\n")
                f.write(f"Epsilon: {round(epsilon, 2)}\n")
                f.write(f"Simulation duration: {sim_time}s\n")
                avg_queue = self._sum_queue_length / self._max_steps if self._max_steps else 0
                f.write(f"Average queue length: {avg_queue:.2f}\n")
                f.write(f"Cumulative wait time: {self._sum_waiting_time:.2f}\n")
                f.write(f"Fault injected: {'Yes' if self.fault_injected_this_episode else 'No'}\n")
                if self.fault_details:
                    f.write("\nFault Details:\n")
                    for detail in self.fault_details:
                        # Each detail should be a tuple: (step, tlid, original, modified)
                        f.write(f"Step {detail[0]} | TLID: {detail[1]} | Orig: {detail[2]} -> Mod: {detail[3]}\n")
                f.write("\nAction Distribution:\n")
                for i, count in enumerate(self._action_counts):
                    f.write(f"Action {i}: {count} times\n")
        except Exception as e:
            print(f"Error writing summary log: {e}")

    def _choose_action(self, state, epsilon, num):
        """
        Epsilon-greedy action selection.
        If exploring, returns a random action.
        Otherwise, performs a REST call for agent 'num' to get predicted Q-values and selects the best action.
        """
        if random.random() < epsilon:
            random_action = random.randint(0, self._num_actions - 1)
            print(f"[DEBUG] Random action chosen for agent {num}: {random_action}")
            return random_action
        else:
            response = requests.post(
                'http://127.0.0.1:5000/predict',
                json={'state': state.tolist(), 'num': int(num)}
            )
            pred = np.array(response.json()['prediction'])
            best_action = int(np.argmax(pred))
            print(f"[DEBUG] Best action chosen for agent {num}: {best_action}")
            return best_action

    def _set_yellow_phase(self, action_number):
        """
        Set yellow phase for the first intersection (agent one).
        """
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        yellow_phase = phase_map[action_number]["yellow"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tlid)
            except traci.exceptions.TraCIException:
                continue
            if not logics:
                continue
            num_phases = len(logics[0].phases)
            if yellow_phase < num_phases:
                traci.trafficlight.setProgram(tlid, "0")
                traci.trafficlight.setPhase(tlid, yellow_phase)
            else:
                print(f"⚠️ Skipping invalid yellow phase {yellow_phase} for {tlid} (only {num_phases} phases)")

    def _set_yellow_phase_two(self, action_number):
        """
        Set yellow phase for the second intersection (agent two).
        """
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        yellow_phase = phase_map[action_number]["yellow"]
        # Try to obtain a second traffic light ID.
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        second_tl = None
        if len(tl_ids) >= 2:
            second_tl = tl_ids[1]
        else:
            if tl_ids:
                candidate = "2_" + tl_ids[0]
                if candidate in traci.trafficlight.getIDList():
                    second_tl = candidate
        if second_tl is not None:
            try:
                traci.trafficlight.setProgram(second_tl, "0")
                traci.trafficlight.setPhase(second_tl, yellow_phase)
            except traci.exceptions.TraCIException as e:
                print(f"Error setting yellow phase for second intersection on {second_tl}: {e}")
        else:
            print("Warning: No second traffic light found for agent two; skipping yellow phase setting.")

    def _set_green_phase(self, action_number):
        """
        Set green phase for the first intersection (agent one).
        Auto-detects traffic light IDs from configuration.
        """
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        green_phase = phase_map[action_number]["green"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tlid)
            except traci.exceptions.TraCIException:
                continue
            if not logics:
                continue
            num_phases = len(logics[0].phases)
            if green_phase < num_phases:
                traci.trafficlight.setProgram(tlid, "0")
                traci.trafficlight.setPhase(tlid, green_phase)
            else:
                print(f"⚠️ Skipping invalid green phase {green_phase} for {tlid} (only {num_phases} phases)")

    def _set_green_phase_two(self, action_number):
        """
        Set green phase for the second intersection (agent two).
        Attempts to auto-detect or construct the second traffic light's ID.
        """
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        green_phase = phase_map[action_number]["green"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        second_tl = None
        if len(tl_ids) >= 2:
            second_tl = tl_ids[1]
        else:
            if tl_ids:
                candidate = "2_" + tl_ids[0]
                if candidate in traci.trafficlight.getIDList():
                    second_tl = candidate
        if second_tl is not None:
            try:
                traci.trafficlight.setProgram(second_tl, "0")
                traci.trafficlight.setPhase(second_tl, green_phase)
            except traci.exceptions.TraCIException as e:
                print(f"Error setting green phase for second intersection on {second_tl}: {e}")
        else:
            print("Warning: No second traffic light found for agent two; skipping green phase setting.")

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            self._inject_signal_faults()
            self._recover_faults_if_due()
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            q1 = self._get_queue_length_intersection_one()
            q2 = self._get_queue_length_intersection_two()
            self._sum_queue_length += (q1 + q2)
            self._sum_waiting_time += (q1 + q2)

    def _inject_signal_faults(self):
        self.manual_override = False
        if self.skip_fault_this_episode or self.fault_injected_this_episode:
            return
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            try:
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tlid)[0]
            except traci.exceptions.TraCIException:
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
                traci.trafficlight.setRedYellowGreenState(tlid, new_state_str)
                self.manual_override = True
                self.fault_injected_this_episode = True
                print(f"[Step {self._step}] ❌ Signal fault injected at TL={tlid}, phase={current_phase}")
                print(f"    Original state: {current_state}")
                print(f"    Modified state: {new_state_str}")
                print(f"    Flipped indices: {flipped_indices}")
                return

    def _recover_faults_if_due(self):
        """
        Check for signals that need to be recovered (reset) based on the recovery queue.
        """
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            key = (tlid, self._step)
            if key in self.recovery_queue:
                original_state = self.recovery_queue[key]
                try:
                    traci.trafficlight.setRedYellowGreenState(tlid, original_state)
                    print(f"[Step {self._step}] ✅ Signal recovered at TL={tlid}")
                    del self.recovery_queue[key]
                    self.manual_override = False
                except traci.exceptions.TraCIException:
                    pass

    def _collect_waiting_times_first_intersection(self):
        """
        Sum waiting times for agent one using the first half of monitor_edges.
        """
        # Get edges from config.
        edges = self.int_conf.get("monitor_edges", [])
        if edges and len(edges) >= 2:
            selected_edges = edges[:len(edges) // 2]
        else:
            # Option: you could also choose to signal an error or a warning.
            # For now, we'll return 0 if no monitor_edges are defined.
            selected_edges = []
            print("Warning: No monitor_edges defined in configuration for agent one.")
        local_wait = {}
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in selected_edges:
                local_wait[car_id] = wait_time
        return sum(local_wait.values())

    def _collect_waiting_times_second_intersection(self):
        """
        Sum waiting times for agent two using the second half of monitor_edges.
        """
        edges = self.int_conf.get("monitor_edges", [])
        if edges and len(edges) >= 2:
            selected_edges = edges[len(edges) // 2:]
        else:
            selected_edges = []
            print("Warning: No monitor_edges defined in configuration for agent two.")
        local_wait = {}
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in selected_edges:
                local_wait[car_id] = wait_time
        return sum(local_wait.values())

    def _get_queue_length_intersection_one(self):
        """
        Retrieve the total number of halted vehicles for agent one.
        Uses the first half of the edges defined in "monitor_edges" in the configuration.
        """
        edges = self.int_conf.get("monitor_edges", [])
        if edges and len(edges) >= 2:
            selected_edges = edges[:len(edges) // 2]
        elif edges:
            selected_edges = edges  # Use all if only one provided.
        else:
            selected_edges = []
            print("Warning: 'monitor_edges' not defined in configuration for agent one.")
        total = 0
        for edge in selected_edges:
            try:
                total += traci.edge.getLastStepHaltingNumber(edge)
            except traci.exceptions.TraCIException as e:
                print(f"Error: Edge '{edge}' is not known: {e}")
        return total

    def _get_queue_length_intersection_two(self):
        """
        Retrieve the total number of halted vehicles for agent two.
        Uses the second half of the edges defined in "monitor_edges" in the configuration.
        """
        edges = self.int_conf.get("monitor_edges", [])
        if edges and len(edges) >= 2:
            selected_edges = edges[len(edges) // 2:]
        elif edges:
            selected_edges = edges
        else:
            selected_edges = []
            print("Warning: 'monitor_edges' not defined in configuration for agent two.")
        total = 0
        for edge in selected_edges:
            try:
                total += traci.edge.getLastStepHaltingNumber(edge)
            except traci.exceptions.TraCIException as e:
                print(f"Error: Edge '{edge}' is not known: {e}")
        return total

    def _get_density(self):
        """
        Retrieve the density (vehicles per km) using edges from configuration.
        It first checks for a key "density_edges" in the config; if not found, it uses "monitor_edges".
        For each edge, density is computed from the number of vehicles and the length of a lane.
        """
        # Try to get a list of edges specifically for density;
        # otherwise use monitor_edges.
        edges = self.int_conf.get("density_edges", self.int_conf.get("monitor_edges", []))
        total_density = 0
        count = 0
        for edge in edges:
            try:
                num_vehicles = traci.edge.getLastStepVehicleNumber(edge)
                # Get the list of lane IDs for this edge and use the first for length.
                lane_ids = traci.edge.getLaneIDs(edge)
                if lane_ids:
                    lane_length = traci.lane.getLength(lane_ids[0])
                    density = num_vehicles / (lane_length / 1000.0)  # vehicles per km
                    total_density += density
                    count += 1
            except traci.exceptions.TraCIException as e:
                print(f"Error retrieving density for edge '{edge}': {e}")
        if count > 0:
            return total_density
        else:
            print("Warning: No valid edges found for computing density.")
            return 0

    def _get_flow(self):
        """
        Retrieve flow (vehicles per hour) using a list of edges from configuration.
        It first tries to use a key "flow_edges"; if not defined, it uses "monitor_edges".
        """
        edges = self.int_conf.get("flow_edges", self.int_conf.get("monitor_edges", []))
        counter_entered = 0
        already_in = []  # local list to track vehicles
        for edge in edges:
            try:
                vehicle_ids = traci.edge.getLastStepVehicleIDs(edge)
                for car_id in vehicle_ids:
                    if car_id not in already_in:
                        counter_entered += 1
                        already_in.append(car_id)
            except traci.exceptions.TraCIException as e:
                print(f"Error retrieving flow for edge '{edge}': {e}")
        # Scale based on simulation duration (you might use self._max_steps instead of a constant)
        return (counter_entered / self._max_steps) * 3600

    def _get_occupancy(self):
        """
        Retrieve the occupancy averaged over edges.
        It first attempts to use a configuration key "occupancy_edges",
        and if not available uses "monitor_edges".
        """
        edges = self.int_conf.get("occupancy_edges", self.int_conf.get("monitor_edges", []))
        total_occ = 0
        count = 0
        for edge in edges:
            try:
                occ = traci.edge.getLastStepOccupancy(edge)
                total_occ += occ
                count += 1
            except traci.exceptions.TraCIException as e:
                print(f"Error retrieving occupancy for edge '{edge}': {e}")
        if count > 0:
            return total_occ / count
        else:
            print("Warning: No valid edges found for computing occupancy.")
            return 0

    def _get_states_with_advanced_perception(self):
        """
        Construct state vectors for agent one and agent two.
        Each state is formed by concatenating four sets of features over cells:
          - Number of vehicles
          - Average speed
          - Cumulative waiting time
          - Number of queued vehicles
        The total vector is of size 2 * self._num_cells, then split equally.
        """
        nb_cars = np.zeros(self._num_cells * 2)
        avg_speed = np.zeros(self._num_cells * 2)
        cumulated_waiting_time = np.zeros(self._num_cells * 2)
        nb_queued_cars = np.zeros(self._num_cells * 2)

        incoming_lanes = self.int_conf.get("incoming_lanes", {})
        if "main" in incoming_lanes and "side" in incoming_lanes:
            lanes_agent1 = incoming_lanes["main"]
            lanes_agent2 = incoming_lanes["side"]
        else:
            all_lanes = []
            for key in sorted(incoming_lanes.keys()):
                all_lanes.extend(incoming_lanes[key])
            half = len(all_lanes) // 2
            lanes_agent1 = all_lanes[:half]
            lanes_agent2 = all_lanes[half:]

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            car_speed = traci.vehicle.getSpeed(car_id)
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_pos = 750 - lane_pos  # Inversion: closer means lower cell index

            if lane_pos < 7:
                cell = 0
            elif lane_pos < 14:
                cell = 1
            elif lane_pos < 21:
                cell = 2
            elif lane_pos < 28:
                cell = 3
            elif lane_pos < 40:
                cell = 4
            elif lane_pos < 60:
                cell = 5
            elif lane_pos < 100:
                cell = 6
            elif lane_pos < 160:
                cell = 7
            elif lane_pos < 400:
                cell = 8
            else:
                cell = 9

            if lane_id in lanes_agent1:
                idx = cell
            elif lane_id in lanes_agent2:
                idx = self._num_cells + cell
            else:
                idx = cell  # default to agent one

            nb_cars[idx] += 1
            avg_speed[idx] += car_speed
            if car_speed < 0.1:
                nb_queued_cars[idx] += 1
            cumulated_waiting_time[idx] += wait_time

        for i in range(len(avg_speed)):
            if nb_cars[i] > 0:
                avg_speed[i] /= nb_cars[i]

        state_one = np.concatenate((
            nb_cars[:self._num_cells],
            avg_speed[:self._num_cells],
            cumulated_waiting_time[:self._num_cells],
            nb_queued_cars[:self._num_cells]
        ))
        state_two = np.concatenate((
            nb_cars[self._num_cells:],
            avg_speed[self._num_cells:],
            cumulated_waiting_time[self._num_cells:],
            nb_queued_cars[self._num_cells:]
        ))
        return state_one, state_two

    def _pad_states(self, state_list):
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

    def _replay(self):
        batch = self._Memory.get_samples(self._Model.batch_size)
        if len(batch) == 0:
            return

        state_list = [sample[0] for sample in batch]
        next_state_list = [sample[3] for sample in batch]
        actions = np.array([sample[1] for sample in batch], dtype=np.int32)
        rewards = np.array([sample[2] for sample in batch], dtype=np.float32)

        states = self._pad_states(state_list)
        next_states = self._pad_states(next_state_list)

        q_s_a = self._Model.predict_batch(states)
        best_next_actions = np.argmax(self._Model.predict_batch(next_states), axis=1)
        target_q_next = self._TargetModel.predict_batch(next_states)
        target_q_vals = target_q_next[np.arange(len(batch)), best_next_actions]

        y = np.copy(q_s_a)
        y[np.arange(len(batch)), actions] = rewards + self._gamma * target_q_vals

        loss = np.mean(np.square(y - q_s_a))
        self._q_loss_log.append(loss)
        self._Model.train_batch(states, y)

    def _save_episode_stats(self):
        self._reward_store.append(self._sum_neg_reward_one + self._sum_neg_reward_two)
        self._reward_store_a1.append(self._sum_neg_reward_one)
        self._reward_store_a2.append(self._sum_neg_reward_two)
        self._cumulative_wait_store.append(self._cumulative_waiting_time_agent_one + self._cumulative_waiting_time_agent_two)
        self._cumulative_wait_store_a1.append(self._cumulative_waiting_time_agent_one)
        self._cumulative_wait_store_a2.append(self._cumulative_waiting_time_agent_two)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)
        self._avg_queue_length_store_a1.append(self._sum_queue_length_a1 / self._max_steps)
        self._avg_queue_length_store_a2.append(self._sum_queue_length_a2 / self._max_steps)

    def analyze_results(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self.reward_store)
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.subplot(1, 3, 2)
        plt.plot(self.avg_queue_length_store)
        plt.title("Average Queue Length per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Avg Queue Length")
        plt.subplot(1, 3, 3)
        plt.plot(self.cumulative_wait_store)
        plt.title("Cumulative Waiting Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Waiting Time")
        plt.show()
        print(f"Final Faulty Lights: {self.faulty_lights}")
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

    @property
    def q_loss_log(self):
        return self._q_loss_log

    def stop(self):
        return (
            self.reward_store[0],
            self._reward_store_a1[0],
            self._reward_store_a2[0],
            self.cumulative_wait_store[0],
            self._cumulative_wait_store_a1[0],
            self._cumulative_wait_store_a2[0],
            self.avg_queue_length_store[0],
            self._avg_queue_length_store_a1[0],
            self._avg_queue_length_store_a2[0],
            self._q_loss_log
        )
