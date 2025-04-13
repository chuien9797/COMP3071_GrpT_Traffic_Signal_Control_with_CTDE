import os
import random
import timeit
import datetime
import numpy as np
import matplotlib.pyplot as plt
import traci

from emergency_handler import check_emergency, handle_emergency_vehicle
import intersection_config as int_config

# Global constants for faults, etc.
RECOVERY_DELAY = 15
FAULT_REWARD_SCALE = 0.5
EPISODE_FAULT_START = 25


class Simulation:
    def __init__(self,
                 Models,  # List of agent models
                 TargetModels,  # List of target models (can be same as Models for shared parameters)
                 Memory,
                 TrafficGen,
                 sumo_cmd,
                 gamma,
                 max_steps,
                 green_duration,
                 yellow_duration,
                 num_states,  # Not used by aggregator but preserved for compatibility.
                 training_epochs,
                 intersection_type="cross",
                 signal_fault_prob=0.1):
        # Store multi-agent models as lists.
        self._Models = Models
        self._TargetModels = TargetModels
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

        # Global statistics and logs.
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._q_loss_log = []
        self._green_durations_log = []
        self.fault_details = []
        self.faulty_lights = set()
        self._emergency_crossed = 0
        self._emergency_total_delay = 0.0
        self._teleport_count = 0

        # Fault injection parameters.
        self.signal_fault_prob = signal_fault_prob
        self.manual_override = False
        self.recovery_queue = {}

        self.intersection_type = intersection_type
        if self.intersection_type not in int_config.INTERSECTION_CONFIGS:
            raise ValueError(f"Intersection type '{intersection_type}' not found in config.")
        self.int_conf = int_config.INTERSECTION_CONFIGS[self.intersection_type]
        self._num_actions = len(self.int_conf["phase_mapping"])
        # Set number of agents based on the provided traffic light IDs.
        self._traffic_light_ids = self.int_conf.get("traffic_light_ids", [])
        self.num_agents = len(self._traffic_light_ids) if isinstance(self._traffic_light_ids, list) else 1

    def run(self, episode, epsilon):
        os.makedirs("logs22", exist_ok=True)
        log_file = open(f"logs22/episode_{episode}.log", "w")

        # Generate routes with the current episode seed.
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print(f"Simulating Episode {episode} on {self.intersection_type}...")

        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        # For multi-agent, we initialize per-agent old state and action.
        old_states = [None] * self.num_agents
        old_actions = [None] * self.num_agents

        self.faulty_lights = set()
        self.fault_injected_this_episode = False
        self.skip_fault_this_episode = (episode < EPISODE_FAULT_START) or (random.random() < 0.5)
        self.handled_emergency_ids = set()
        start_time = timeit.default_timer()

        while self._step < self._max_steps:
            if check_emergency(self):
                continue

            # Get the overall state and partition it among agents.
            states = self._get_state()  # returns a list with one state per agent

            # For each agent, calculate reward, update memory and choose an action.
            actions = []
            for i in range(self.num_agents):
                current_state = states[i]
                # For reward, we use the difference of total wait from the previous step.
                # Here, we use a simplified global wait instead of per-agent wait; adjust as needed.
                current_total_wait = self._collect_waiting_times()
                reward = 0.0
                if self._step != 0 and old_states[i] is not None:
                    reward = float(old_total_wait - current_total_wait)
                    if self.fault_injected_this_episode:
                        reward *= FAULT_REWARD_SCALE
                    self._Memory.add_sample((old_states[i], old_actions[i], reward, current_state))
                # Choose action using the i-th model.
                action = self._choose_action(current_state, epsilon, model=self._Models[i])
                actions.append(action)
                # Log action for this agent.
                log_file.write(f"[Step {self._step}] Agent {i} Action: {action}\n")

            # If phase changed between steps, set yellow phases; here we assume old_actions from agent 0 as example.
            if self._step != 0:
                for i in range(self.num_agents):
                    if old_actions[i] is not None and old_actions[i] != actions[i]:
                        self._set_yellow_phase(i, old_actions[i], log_file)
                        self._simulate(self._yellow_duration, log_file)

            # Set green phases for each agent.
            for i, action in enumerate(actions):
                self._set_green_phase(i, action, log_file)

            # Adaptive green duration can be computed from global state; we use state from agent 0 as a proxy.
            adaptive_green = self._compute_adaptive_green_duration(states[0])
            self._green_durations_log.append(adaptive_green)
            log_file.write(f"[Step {self._step}] Adaptive green duration: {adaptive_green}\n")
            self._simulate(adaptive_green, log_file)

            # Store overall values for reward calculation.
            old_total_wait = self._collect_waiting_times()
            old_states = states
            old_actions = actions

            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        self._save_episode_stats()
        self._write_summary_log(episode, epsilon, simulation_time)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _collect_waiting_times(self):
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        total_wait = 0.0
        for veh in traci.vehicle.getIDList():
            lane = traci.vehicle.getLaneID(veh)
            if lane in incoming_lane_ids:
                total_wait += traci.vehicle.getAccumulatedWaitingTime(veh)
        return total_wait

    def _get_state(self):
        """
        Aggregates lane features into a 2D array, then partitions the lanes equally among the agents.
        Returns a list of state arrays (one per agent).
        """
        incoming_lanes = self.int_conf["incoming_lanes"]
        lane_order = []
        for group in sorted(incoming_lanes.keys()):
            lane_order.extend(incoming_lanes[group])
        num_lanes = len(lane_order)
        lane_feature_dim = 9
        lane_features = np.zeros((num_lanes, lane_feature_dim), dtype=np.float32)

        intersection_encoding = {
            "cross": [1.0, 0.0, 0.0],
            "roundabout": [0.0, 1.0, 0.0],
            "t_intersection": [0.0, 0.0, 1.0],
            "y_intersection": [0.33, 0.33, 0.34]
        }
        type_vector = intersection_encoding.get(self.intersection_type.lower(), [0.0, 0.0, 0.0])

        # Fill the lane features similar to before.
        sorted_lanes = sorted(sum([v for v in self.int_conf["incoming_lanes"].values()], []))
        for i, lane_id in enumerate(sorted_lanes):
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
                    logics = traci.trafficlight.getAllProgramLogics(tl_ids[0])
                    if logics:
                        current_phase = traci.trafficlight.getPhase(tl_ids[0])
                        phase_state = logics[0].phases[current_phase].state
                        connections = traci.trafficlight.getControlledLanes(tl_ids[0])
                        if lane_id in connections:
                            idx = connections.index(lane_id)
                            if phase_state[idx] == 'r':
                                controlled_by_faulty = 1.0
                except Exception as e:
                    pass
            lane_features[i, 5] = controlled_by_faulty
            lane_features[i, 6:9] = np.array(type_vector)

        # Partition lanes equally among agents.
        lanes_per_agent = num_lanes // self.num_agents if self.num_agents > 0 else num_lanes
        states = []
        for i in range(self.num_agents):
            start = i * lanes_per_agent
            # For the last agent, include any remaining lanes.
            if i == self.num_agents - 1:
                state_i = lane_features[start:]
            else:
                state_i = lane_features[start:start + lanes_per_agent]
            states.append(state_i)
        return states

    def _choose_action(self, state, epsilon, model):
        """
        Given a state (for a single agent) and a model, choose an action using epsilon-greedy.
        """
        valid_action_indices = list(self.int_conf["phase_mapping"].keys())
        if random.random() < epsilon:
            return random.choice(valid_action_indices)
        # Expand dimensions to add batch dimension if needed.
        state_expanded = np.expand_dims(state, axis=0)
        q_vals = model.predict_one(state)[0]  # shape: (num_actions,)
        # (Optionally, add any emergency adjustments here.)
        valid_q_vals = q_vals[valid_action_indices]
        best_valid_action = valid_action_indices[int(np.argmax(valid_q_vals))]
        return best_valid_action

    def _set_yellow_phase(self, agent_index, action_number, log_file=None):
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
                        if log_file:
                            log_file.write(
                                f"[SetYellow] TL {tlid}: Set yellow phase {yellow_phase} out of {num_phases} phases.\n")
                    else:
                        message = f"⚠️ Invalid yellow phase {yellow_phase} for TL {tlid} (only {num_phases} phases)"
                        print(message)
                        if log_file:
                            log_file.write(f"[SetYellow] {message}\n")
            except Exception as e:
                message = f"Error setting yellow phase for TL {tlid}: {e}"
                print(message)
                if log_file:
                    log_file.write(f"[SetYellow] {message}\n")

    def _set_green_phase(self, agent_index, action_number, log_file=None):
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            if log_file:
                log_file.write(f"[SetGreen] Action {action_number} not found in phase mapping.\n")
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
                        message = f"⚠️ Skipping invalid green phase {green_phase} for TL {tlid} (only {num_phases} phases)."
                        print(message)
                        if log_file:
                            log_file.write(message + "\n")
            except Exception as e:
                message = f"Error setting green phase for TL {tlid}: {e}"
                print(message)
                if log_file:
                    log_file.write(message + "\n")

    def _compute_adaptive_green_duration(self, state):
        avg_wait = np.mean(state[:, 1])
        queue_length = np.sum(state[:, 3])
        emergency_factor = np.any(state[:, 2] > 0)
        base = self._green_duration
        wait_factor = int(avg_wait // 2)
        queue_factor = int(queue_length // 5)
        emergency_bonus = 3 if emergency_factor else 0
        adaptive_extension = min(wait_factor + queue_factor + emergency_bonus, 10)
        return base + adaptive_extension

    def _simulate(self, steps_todo, log_file=None):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            self._inject_signal_faults(log_file)
            self._recover_faults_if_due(log_file)
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            q_len = self._get_queue_length()
            self._sum_queue_length += q_len
            current_wait = self._collect_waiting_times()
            self._sum_waiting_time += current_wait
            self._teleport_count += traci.simulation.getStartingTeleportNumber()
            if log_file:
                log_file.write(f"[Simulate] Step {self._step}: Queue length {q_len}, Waiting time {current_wait}\n")

    def _inject_signal_faults(self, log_file=None):
        self.manual_override = False
        if self.skip_fault_this_episode or self.fault_injected_this_episode:
            return
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            try:
                logics = traci.trafficlight.getAllProgramLogics(tlid)
                if not logics:
                    continue
                logic = logics[0]
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
                    message = (f"[InjectFault] TL {tlid}: Fault injected. "
                               f"Phase {current_phase} changed from {current_state} to {new_state_str}. "
                               f"Flipped indices: {flipped_indices}")
                    print(message)
                    if log_file:
                        log_file.write(message + "\n")
                    return
            except Exception as e:
                if log_file:
                    log_file.write(f"[InjectFault] Error on TL {tlid}: {e}\n")
                continue

    def _recover_faults_if_due(self, log_file=None):
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            key = (tlid, self._step)
            if key in self.recovery_queue:
                original_state = self.recovery_queue[key]
                try:
                    traci.trafficlight.setRedYellowGreenState(tlid, original_state)
                    message = f"[RecoverFault] TL {tlid}: Signal recovered at step {self._step}."
                    print(message)
                    if log_file:
                        log_file.write(message + "\n")
                    del self.recovery_queue[key]
                    self.manual_override = False
                except traci.exceptions.TraCIException as e:
                    if log_file:
                        log_file.write(f"[RecoverFault] TL {tlid}: Exception {e}\n")
                    pass

    def _get_queue_length(self):
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lane_ids)

    def _replay(self):
        batch = self._Memory.get_samples(self._Models[0].batch_size)
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
        q_s_a = self._Models[0].predict_batch(states)
        best_next_actions = np.argmax(self._Models[0].predict_batch(next_states), axis=1)
        target_q_next = self._TargetModels[0].predict_batch(next_states)
        target_q_vals = target_q_next[np.arange(len(batch)), best_next_actions]
        y = np.copy(q_s_a)
        y[np.arange(len(batch)), actions] = rewards + self._gamma * target_q_vals
        loss = np.mean(np.square(y - q_s_a))
        self._q_loss_log.append(loss)
        self._Models[0].train_batch(states, y)

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
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
                for i in range(self.num_agents):
                    f.write(f"Agent {i + 1}\n")
                # Additional logging as needed.
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
