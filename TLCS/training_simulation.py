import traci
import numpy as np
import random
import timeit
import os
import matplotlib.pyplot as plt
from datetime import datetime

from emergency_handler import check_emergency
import intersection_config as int_config

# Global constants
RECOVERY_DELAY = 15  # Steps to recover faulty signals
FAULT_REWARD_SCALE = 0.5  # Scale reward if fault occurs
EPISODE_FAULT_START = 25  # Start injecting faults after a given episode


class Simulation:
    def __init__(
            self,
            agents,  # List of controller agents, one per intersection
            Memory,  # A shared replay Memory instance
            TrafficGen,
            sumo_cmd,
            gamma,
            max_steps,
            green_duration,
            yellow_duration,
            num_states,  # Not used by aggregator but preserved for interface
            training_epochs,
            intersection_type="cross",
            signal_fault_prob=0.1,
    ):
        # Instead of a single model, we now store a list of controllers.
        self._agents = agents
        # Shared Memory instance
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

        # Global statistics for the entire simulation (across intersections)
        self._reward_store = []  # Combined reward per episode
        self._cumulative_wait_store = []  # Combined cumulative waiting time per episode
        self._avg_queue_length_store = []  # Combined average queue length per episode

        # Logging and fault-related arrays
        self._q_loss_log = []
        self._green_durations_log = []
        self.fault_details = []
        self.faulty_lights = set()
        self._emergency_crossed = 0
        self._emergency_total_delay = 0.0
        self._teleport_count = 0

        # Fault injection parameters
        self.signal_fault_prob = signal_fault_prob
        self.manual_override = False
        self.recovery_queue = {}

        self.int_conf = int_config.INTERSECTION_CONFIGS.get(intersection_type)
        if self.int_conf is None:
            raise ValueError(f"Intersection type '{intersection_type}' not found in config.")
        self.intersection_type = intersection_type

        # Determine number of intersections from configuration.
        # It is assumed that int_conf["traffic_light_ids"] is a list.
        self._num_intersections = 1
        if "traffic_light_ids" in self.int_conf:
            if isinstance(self.int_conf["traffic_light_ids"], list):
                self._num_intersections = len(self.int_conf["traffic_light_ids"])
            else:
                self._num_intersections = 1

        # Create per-intersection action counters.
        self._action_counts = [np.zeros(self._agents[0]._num_actions, dtype=int)
                               for _ in range(self._num_intersections)]

    def run(self, episode, epsilon):
        os.makedirs("logs19", exist_ok=True)
        log_file = open(f"logs19/episode_{episode}.log", "w")

        # Generate route file and start SUMO
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print(
            f"Simulating Episode {episode} on environment '{self.intersection_type}' with {self._num_intersections} intersection(s)...")

        # Reset simulation and per-intersection trackers.
        self._step = 0
        old_states = [None] * self._num_intersections
        old_actions = [None] * self._num_intersections
        old_total_waits = [0.0] * self._num_intersections

        self._sum_neg_reward = 0.0
        self._sum_queue_length = 0.0
        self._sum_waiting_time = 0.0
        self.faulty_lights = set()
        self._q_loss_log = []

        self.fault_injected_this_episode = False
        self.skip_fault_this_episode = (episode < EPISODE_FAULT_START) or (random.random() < 0.5)

        start_time = timeit.default_timer()

        while self._step < self._max_steps:
            if check_emergency(self):
                continue

            # Get overall state and split among intersections.
            overall_state = self._get_state()  # Shape: (total_lanes, lane_feature_dim)
            states = self._split_state(overall_state, self._num_intersections)

            current_total_waits = []
            for i in range(self._num_intersections):
                # For demonstration, use sum of waiting times (assume column 1 holds waiting times)
                wait_time = np.sum(states[i][:, 1])
                current_total_waits.append(wait_time)

            # Update memory with experiences (if not the very first step)
            if self._step != 0:
                for i in range(self._num_intersections):
                    reward = old_total_waits[i] - current_total_waits[i]
                    if self.fault_injected_this_episode:
                        reward *= FAULT_REWARD_SCALE
                    # Add sample: (old_state, old_action, reward, current_state)
                    self._Memory.add_sample((old_states[i], old_actions[i], reward, states[i]))
                    # print(
                    #     f"[Step {self._step}] Added sample for intersection {i + 1}. Memory size now: {self._Memory._size_now()} samples")
                    if reward < 0:
                        self._sum_neg_reward += reward

            # Each agent chooses an action for its intersection.
            actions = []
            for i in range(self._num_intersections):
                action = self._choose_action(states[i], epsilon, i)
                actions.append(action)
                self._action_counts[i][action] += 1
                log_file.write(f"[Step {self._step}] Intersection {i + 1} Action: {action}\n")

            # Handle yellow phase transitions for intersections with action change.
            for i in range(self._num_intersections):
                if self._step != 0 and old_actions[i] is not None and old_actions[i] != actions[i]:
                    self._set_yellow_phase(i, old_actions[i])
                    self._simulate(self._yellow_duration)

            # Set green phase for each intersection.
            for i in range(self._num_intersections):
                self._set_green_phase(i, actions[i])

            # Compute adaptive green duration based on overall state.
            adaptive_green = self._compute_adaptive_green_duration(overall_state)
            self._green_durations_log.append(adaptive_green)
            log_file.write(f"[Step {self._step}] Adaptive green duration: {adaptive_green}\n")
            self._simulate(adaptive_green)

            # Save current states, actions, and waiting times for the next step.
            for i in range(self._num_intersections):
                old_states[i] = states[i]
                old_actions[i] = actions[i]
                old_total_waits[i] = current_total_waits[i]

        # End of simulation: save stats and summary log.
        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        self._write_summary_log(episode, epsilon, simulation_time)

        print("Training phase starting...")
        start_train = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()  # Replay uses the shared Memory and the common agent model (assumed shared)
        training_time = round(timeit.default_timer() - start_train, 1)
        return simulation_time, training_time

    def _split_state(self, overall_state, num_parts):
        total_lanes = overall_state.shape[0]
        part = total_lanes // num_parts
        states = []
        for i in range(num_parts):
            start = i * part
            if i == num_parts - 1:
                s = overall_state[start:, :]
            else:
                s = overall_state[start:start + part, :]
            states.append(s)
        return states

    def _get_state(self):
        """Retrieve the overall state from SUMO.
        Returns an array of shape (total_lanes, lane_feature_dim) using the lanes specified.
        """
        incoming_lanes = self.int_conf["incoming_lanes"]
        lane_order = []
        for group in sorted(incoming_lanes.keys()):
            lane_order.extend(incoming_lanes[group])
        num_lanes = len(lane_order)
        lane_feature_dim = 9  # For example: vehicle count, waiting time, emergency flag, halting, current phase, faulty flag, type vector.
        lane_features = np.zeros((num_lanes, lane_feature_dim), dtype=np.float32)

        intersection_encoding = {
            "cross": [1.0, 0.0, 0.0],
            "roundabout": [0.0, 1.0, 0.0],
            "t_intersection": [0.0, 0.0, 1.0],
            "y_intersection": [0.33, 0.33, 0.34]
        }
        type_vector = intersection_encoding.get(self.intersection_type.lower(), [0.0, 0.0, 0.0])

        for i, lane_id in enumerate(lane_order):
            lane_features[i, 0] = traci.lane.getLastStepVehicleNumber(lane_id)
            lane_features[i, 1] = traci.lane.getWaitingTime(lane_id)
            flag = 0
            for car_id in traci.lane.getLastStepVehicleIDs(lane_id):
                if traci.vehicle.getTypeID(car_id) == "emergency":
                    flag = 1
                    break
            lane_features[i, 2] = float(flag)
            lane_features[i, 3] = traci.lane.getLastStepHaltingNumber(lane_id)
            tl_ids = self.int_conf.get("traffic_light_ids", [])
            phase_val = 0.0
            if tl_ids:
                phase_val = float(traci.trafficlight.getPhase(tl_ids[0]))
            lane_features[i, 4] = phase_val
            controlled_by_faulty = 0.0
            if tl_ids:
                try:
                    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_ids[0])[0]
                    current_phase = traci.trafficlight.getPhase(tl_ids[0])
                    phase_state = logic.phases[current_phase].state
                    connections = traci.trafficlight.getControlledLanes(tl_ids[0])
                    if lane_id in connections:
                        idx = connections.index(lane_id)
                        if phase_state[idx] == 'r':
                            controlled_by_faulty = 1.0
                except Exception as e:
                    pass
            lane_features[i, 5] = controlled_by_faulty
            lane_features[i, 6:9] = np.array(type_vector)
        return lane_features

    def _choose_action(self, state, epsilon, agent_index):
        """Epsilon–greedy action selection for the intersection corresponding to agent_index."""
        valid_actions = list(self.int_conf["phase_mapping"].keys())
        if random.random() < epsilon:
            action = random.choice(valid_actions)
            # Uncomment the next line to log random action selections.
            # print(f"[DEBUG] Intersection {agent_index+1} random action: {action}")
            return action
        else:
            q_vals = self._agents[agent_index].predict_one(state)[0]
            best_action = valid_actions[int(np.argmax(q_vals))]
            # Uncomment the next line to log greedy selections.
            # print(f"[DEBUG] Intersection {agent_index+1} best action: {best_action}")
            return best_action

    def _set_yellow_phase(self, agent_index, action_number):
        """Set the yellow phase for the intersection corresponding to agent_index."""
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        yellow_phase = phase_map[action_number]["yellow"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        if tl_ids and agent_index < len(tl_ids):
            tlid = tl_ids[agent_index]
            try:
                logics = traci.trafficlight.getAllProgramLogics(tlid)
                if logics:
                    num_phases = len(logics[0].phases)
                    if yellow_phase < num_phases:
                        traci.trafficlight.setProgram(tlid, "0")
                        traci.trafficlight.setPhase(tlid, yellow_phase)
                    else:
                        print(f"⚠️ Invalid yellow phase {yellow_phase} for TL {tlid}")
            except Exception as e:
                print(f"Error setting yellow phase for {tlid}: {e}")

    def _set_green_phase(self, agent_index, action_number):
        """Set the green phase for the intersection corresponding to agent_index."""
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        green_phase = phase_map[action_number]["green"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        if tl_ids and agent_index < len(tl_ids):
            tlid = tl_ids[agent_index]
            try:
                logics = traci.trafficlight.getAllProgramLogics(tlid)
                if logics:
                    num_phases = len(logics[0].phases)
                    if green_phase < num_phases:
                        traci.trafficlight.setProgram(tlid, "0")
                        traci.trafficlight.setPhase(tlid, green_phase)
                    else:
                        print(f"⚠️ Skipping invalid green phase {green_phase} for {tlid} (only {num_phases} phases)")
            except Exception as e:
                print(f"Error setting green phase for {tlid}: {e}")

    def _compute_adaptive_green_duration(self, overall_state):
        """Compute adaptive green duration based on overall state.
        Uses average waiting time (column 1) and total queue length (column 3).
        """
        avg_wait = np.mean(overall_state[:, 1])
        total_queue = np.sum(overall_state[:, 3])
        emergency_flag = np.any(overall_state[:, 2] > 0)
        base = self._green_duration
        wait_factor = int(avg_wait // 2)
        queue_factor = int(total_queue // 5)
        emergency_bonus = 3 if emergency_flag else 0
        extension = min(wait_factor + queue_factor + emergency_bonus, 10)
        return base + extension

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            self._inject_signal_faults()
            self._recover_faults_if_due()
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            q_len = self._get_queue_length()
            self._sum_queue_length += q_len
            self._sum_waiting_time += q_len
            self._teleport_count += traci.simulation.getStartingTeleportNumber()

            for veh_id in traci.vehicle.getIDList():
                if traci.vehicle.getTypeID(veh_id) == "emergency":
                    delay = traci.vehicle.getAccumulatedWaitingTime(veh_id)
                    self._emergency_total_delay += delay
                    if traci.vehicle.getRoadID(veh_id) == "":
                        self._emergency_crossed += 1

    def _inject_signal_faults(self):
        self.manual_override = False
        if self.skip_fault_this_episode or self.fault_injected_this_episode:
            return

        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            try:
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tlid)[0]
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
                    print(f"[Step {self._step}] Signal fault injected at TL={tlid}: {current_state} -> {new_state_str}")
                    return
            except Exception as e:
                continue

    def _recover_faults_if_due(self):
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            key = (tlid, self._step)
            if key in self.recovery_queue:
                original_state = self.recovery_queue[key]
                try:
                    traci.trafficlight.setRedYellowGreenState(tlid, original_state)
                    print(f"[Step {self._step}] Signal recovered at TL={tlid}")
                    del self.recovery_queue[key]
                    self.manual_override = False
                except Exception as e:
                    pass

    def _get_queue_length(self):
        """Compute total queue length across all incoming lanes."""
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lane_ids)

    def _replay(self):
        # Use the batch size of the first agent; assumes a shared model.
        batch = self._Memory.get_samples(self._agents[0].batch_size)
        # print(f"[Replay] Memory size: {self._Memory._size_now()} samples")
        if len(batch) == 0:
            print("[Replay] Not enough samples. Skipping training update.")
            return

        state_list = []
        next_state_list = []
        actions = []
        rewards = []
        for sample in batch:
            st, act, rew, nst = sample
            state_list.append(st)
            next_state_list.append(nst)
            actions.append(act)
            rewards.append(rew)

        states = self._pad_states(state_list)
        next_states = self._pad_states(next_state_list)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)

        # Replace self._Model with self._agents[0] (assuming a shared model).
        q_s_a = self._agents[0].predict_batch(states)
        best_next_actions = np.argmax(self._agents[0].predict_batch(next_states), axis=1)
        target_q_next = self._agents[0].predict_batch(next_states)
        target_q_vals = target_q_next[np.arange(len(batch)), best_next_actions]

        y = np.copy(q_s_a)
        y[np.arange(len(batch)), actions] = rewards + self._gamma * target_q_vals

        loss = np.mean(np.square(y - q_s_a))
        self._q_loss_log.append(loss)
        # print(f"[Replay] Training loss: {loss:.4f}")

        self._agents[0].train_batch(states, y)

    def _pad_states(self, state_list):
        lane_feature_dim = state_list[0].shape[1]
        max_lanes = max(state.shape[0] for state in state_list)
        padded = []
        for state in state_list:
            pad_size = max_lanes - state.shape[0]
            if pad_size > 0:
                padded_state = np.pad(state, ((0, pad_size), (0, 0)), mode='constant')
            else:
                padded_state = state
            padded.append(padded_state)
        return np.array(padded, dtype=np.float32)

    def _save_episode_stats(self):
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    def _write_summary_log(self, episode, epsilon, sim_time):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(f"logs19/episode_{episode}_summary.log", "w", encoding="utf-8") as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Intersection Type: {self.intersection_type}\n")
                f.write(f"Total reward: {self._sum_neg_reward:.2f}\n")
                f.write(f"Epsilon: {round(epsilon, 2)}\n")
                f.write(f"Simulation duration: {sim_time}s\n")
                f.write(f"Avg queue length: {self._sum_queue_length / self._max_steps:.2f}\n")
                f.write(f"Cumulative wait time: {self._sum_waiting_time:.2f}\n")
                f.write(f"Fault injected: {'Yes' if self.fault_injected_this_episode else 'No'}\n")
                if self.fault_details:
                    f.write("\nFault Details:\n")
                    for step, tlid, orig, mod in self.fault_details:
                        f.write(f"Step {step} | TLID: {tlid} | Orig: {orig} -> Mod: {mod}\n")
                f.write(f"Total Emergency Delay: {self._emergency_total_delay:.2f}\n")
                f.write("\nAction Distribution:\n")
                for i, counts in enumerate(self._action_counts):
                    f.write(f"Intersection {i + 1}:\n")
                    for a, count in enumerate(counts):
                        f.write(f" Action {a}: {count} times\n")
        except Exception as e:
            print(f"Error writing summary log: {e}")

    def analyze_results(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self._reward_store)
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.subplot(1, 3, 2)
        plt.plot(self._avg_queue_length_store)
        plt.title("Avg Queue Length per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Avg Queue Length")
        plt.subplot(1, 3, 3)
        plt.plot(self._cumulative_wait_store)
        plt.title("Cumulative Wait Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Wait Time")
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

    @property
    def q_loss_log(self):
        return self._q_loss_log
