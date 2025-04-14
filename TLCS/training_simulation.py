import os
import random
import timeit
import datetime
import numpy as np
import matplotlib.pyplot as plt
import traci

from emergency_handler import check_emergency, handle_emergency_vehicle
import intersection_config as int_config
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow as tf

# Global constants for faults, etc.
RECOVERY_DELAY = 15
FAULT_REWARD_SCALE = 0.5
EPISODE_FAULT_START = 25


###############################################################################
# Centralized Critic Network for CTDE
###############################################################################
class CentralizedCritic(tf.keras.Model):
    """
    A centralized critic network that maps the global state to a joint Q value.
    In this discrete-action setting, we assume a scalar value representing the joint
    evaluation of the current global state. This network is trained with an MSE loss.
    """
    def __init__(self, global_state_dim, hidden_units=64):
        super().__init__()
        self.dense1 = layers.Dense(hidden_units, activation='relu')
        self.dense2 = layers.Dense(hidden_units, activation='relu')
        self.out = layers.Dense(1, activation='linear')

    def call(self, global_state):
        x = self.dense1(global_state)
        x = self.dense2(x)
        q_joint = self.out(x)
        return q_joint  # shape: (batch_size, 1)


###############################################################################
# Memory Class with Encapsulated per-Agent Sample Retrieval
###############################################################################
class Memory:
    def __init__(self, size_max, size_min):
        """
        Initialize the replay memory.

        In a multi-agent scenario, each sample is stored as a tuple:
          (agent_id, state, action, reward, next_state, global_state)
        """
        self._samples = []
        self._size_max = size_max
        self._size_min = size_min

    def add_sample(self, sample):
        """
        Add a sample to the memory.

        Parameters:
            sample (tuple): Expected form is
              (agent_id, state, action, reward, next_state, global_state)
        """
        self._samples.append(sample)
        if self._size_now() > self._size_max:
            self._samples.pop(0)

    def get_samples(self, n):
        """
        Retrieve n random samples from memory if enough samples exist.
        """
        if self._size_now() < self._size_min:
            return []
        available_samples = self._size_now()
        if n > available_samples:
            return random.sample(self._samples, available_samples)
        else:
            return random.sample(self._samples, n)

    def get_samples_by_agent(self, agent_id, n):
        """
        Retrieve n random samples specifically for the given agent id.
        """
        agent_samples = [sample for sample in self._samples if sample[0] == agent_id]
        if len(agent_samples) < self._size_min:
            return []
        if n > len(agent_samples):
            return random.sample(agent_samples, len(agent_samples))
        else:
            return random.sample(agent_samples, n)

    def _size_now(self):
        return len(self._samples)


###############################################################################
# Simulation Class with Flexible Reward Sharing, Per-Agent Replay, and CTDE
###############################################################################
class Simulation:
    def __init__(self,
                 Models,         # List of agent models
                 TargetModels,   # List of target models (can be same as Models for shared params)
                 Memory,
                 TrafficGen,
                 sumo_cmd,
                 gamma,
                 max_steps,
                 green_duration,
                 yellow_duration,
                 num_states,     # Not used by aggregator but preserved for compatibility
                 training_epochs,
                 intersection_type="cross",
                 signal_fault_prob=0.1):
        # Store multi-agent models and other parameters.
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

        # Logging and statistics.
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

        self._sum_queue_length = 0
        self._sum_waiting_time = 0

        self.signal_fault_prob = signal_fault_prob
        self.manual_override = False
        self.recovery_queue = {}

        self.intersection_type = intersection_type
        if self.intersection_type not in int_config.INTERSECTION_CONFIGS:
            raise ValueError(f"Intersection type '{intersection_type}' not found in config.")
        self.int_conf = int_config.INTERSECTION_CONFIGS[self.intersection_type]
        self._num_actions = len(self.int_conf["phase_mapping"])
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        self.num_agents = len(tl_ids) if isinstance(tl_ids, list) else 1

        # Instantiate the centralized critic.
        # Compute global state dimension as (total number of lanes across all groups * lane_feature_dim).
        lane_feature_dim = 9
        total_lanes = sum(len(lanes) for lanes in self.int_conf["incoming_lanes"].values())
        global_state_dim = total_lanes * lane_feature_dim
        self.centralized_critic = CentralizedCritic(global_state_dim, hidden_units=64)
        self.centralized_critic.compile(loss='mse', optimizer=Adam(learning_rate=0.001))

    def _get_state(self):
        """
        Returns a list of state matrices—one per agent.
        Each state is a matrix of shape [num_lanes, lane_feature_dim].
        """
        groups = sorted(self.int_conf["incoming_lanes"].keys())
        states = []
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        lane_feature_dim = 9
        for agent_index, group in enumerate(groups):
            lanes = self.int_conf["incoming_lanes"][group]
            num_lanes = len(lanes)
            group_state = np.zeros((num_lanes, lane_feature_dim), dtype=np.float32)
            for i, lane_id in enumerate(lanes):
                group_state[i, 0] = traci.lane.getLastStepVehicleNumber(lane_id)
                group_state[i, 1] = traci.lane.getWaitingTime(lane_id)
                flag = 0
                for car_id in traci.lane.getLastStepVehicleIDs(lane_id):
                    if traci.vehicle.getTypeID(car_id) == "emergency":
                        flag = 1
                        break
                group_state[i, 2] = float(flag)
                group_state[i, 3] = traci.lane.getLastStepHaltingNumber(lane_id)
                if agent_index < len(tl_ids):
                    phase_val = float(traci.trafficlight.getPhase(tl_ids[agent_index]))
                else:
                    phase_val = 0.0
                group_state[i, 4] = phase_val
                controlled_by_faulty = 0.0
                if agent_index < len(tl_ids):
                    tlid = tl_ids[agent_index]
                    try:
                        logics = traci.trafficlight.getAllProgramLogics(tlid)
                        if logics:
                            current_phase = traci.trafficlight.getPhase(tlid)
                            phase_state = logics[0].phases[current_phase].state
                            connections = traci.trafficlight.getControlledLanes(tlid)
                            if lane_id in connections:
                                conn_idx = connections.index(lane_id)
                                if phase_state[conn_idx] == 'r':
                                    controlled_by_faulty = 1.0
                    except Exception as e:
                        pass
                group_state[i, 5] = controlled_by_faulty
                intersection_encoding = {
                    "cross": [1.0, 0.0, 0.0],
                    "roundabout": [0.0, 1.0, 0.0],
                    "t_intersection": [0.0, 0.0, 1.0],
                    "y_intersection": [0.33, 0.33, 0.34]
                }
                type_vector = intersection_encoding.get(self.intersection_type.lower(), [0.0, 0.0, 0.0])
                group_state[i, 6:9] = np.array(type_vector)
            states.append(group_state)
        return states

    def _get_global_state(self, states):
        """
        Combine local states into one global state vector.
        Here we flatten each agent’s state and concatenate them.
        """
        flat_list = [s.flatten() for s in states]
        global_state = np.concatenate(flat_list)
        return global_state

    def _collect_waiting_times_per_agent(self, agent_index):
        groups = sorted(self.int_conf["incoming_lanes"].keys())
        if agent_index < len(groups):
            lanes = self.int_conf["incoming_lanes"][groups[agent_index]]
        else:
            lanes = []
        total_wait = 0.0
        for lane in lanes:
            total_wait += traci.lane.getWaitingTime(lane)
        return total_wait

    def _get_queue_length_per_agent(self, agent_index):
        groups = sorted(self.int_conf["incoming_lanes"].keys())
        if agent_index < len(groups):
            lanes = self.int_conf["incoming_lanes"][groups[agent_index]]
        else:
            lanes = []
        total_halt = 0
        for lane in lanes:
            total_halt += traci.lane.getLastStepHaltingNumber(lane)
        return total_halt

    def run(self, episode, epsilon):
        """
        Main simulation loop:
          - Generates route files and steps through the simulation.
          - Computes per-agent local rewards.
          - Uses a flexible reward-sharing scheme: each agent’s final reward is a mix of
            its own local reward and the average of other agents' rewards.
          - Stores experiences in memory along with a computed global state.
          - After simulation, calls replay() and train_ctde() to update networks.
        """
        os.makedirs("logs22", exist_ok=True)
        log_file = open(f"logs22/episode_{episode}.log", "w")

        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print(f"Simulating Episode {episode} on {self.intersection_type}...")

        self._step = 0
        old_states = [None] * self.num_agents
        old_actions = [None] * self.num_agents
        old_waits = [0.0] * self.num_agents
        self._sum_neg_reward = 0
        self.faulty_lights = set()
        self.fault_injected_this_episode = False
        self.skip_fault_this_episode = (episode < EPISODE_FAULT_START) or (random.random() < 0.5)
        self.handled_emergency_ids = set()
        start_time = timeit.default_timer()

        # Reward scaling parameters.
        ALPHA_WAIT = 0.2
        BETA_QUEUE = 1.0
        PHASE_SWITCH_PENALTY = 10.0
        EMERGENCY_BONUS = 50.0
        REWARD_SCALE = 0.01

        while self._step < self._max_steps:
            emergency_present = check_emergency(self)
            states = self._get_state()
            current_waits = []
            for i in range(self.num_agents):
                wt = self._collect_waiting_times_per_agent(i)
                ql = self._get_queue_length_per_agent(i)
                current_wait = ALPHA_WAIT * wt + BETA_QUEUE * ql
                current_waits.append(current_wait)

            # Compute local rewards.
            local_rewards = [0.0] * self.num_agents
            for i in range(self.num_agents):
                if self._step != 0 and old_states[i] is not None:
                    local_reward = old_waits[i] - current_waits[i]
                else:
                    local_reward = 0.0

                if self._step != 0 and old_actions[i] is not None and \
                   old_actions[i] != self._choose_action(states[i], 0, self._Models[i]):
                    local_reward -= PHASE_SWITCH_PENALTY

                if emergency_present and np.any(states[i][:, 2] > 0):
                    local_reward += EMERGENCY_BONUS

                local_rewards[i] = local_reward * REWARD_SCALE

            # Flexible reward sharing across agents.
            lambda_coef = 0.5
            final_rewards = []
            for i in range(self.num_agents):
                if self.num_agents > 1:
                    other_sum = sum(local_rewards) - local_rewards[i]
                    global_component = other_sum / (self.num_agents - 1)
                else:
                    global_component = 0.0
                final_reward = (1 - lambda_coef) * local_rewards[i] + lambda_coef * global_component
                final_rewards.append(final_reward)

            # Compute global state (for CTDE).
            global_state = self._get_global_state(states)

            actions = []
            for i in range(self.num_agents):
                if self._step != 0 and old_states[i] is not None:
                    # Store experience: (agent_id, old_state, old_action, reward, new_state, global_state)
                    self._Memory.add_sample((i, old_states[i], old_actions[i], final_rewards[i], states[i], global_state))
                action = self._choose_action(states[i], epsilon, model=self._Models[i])
                actions.append(action)
                log_file.write(f"[Step {self._step}] Agent {i} Action: {action}, Reward: {final_rewards[i]:.4f}\n")

            # (Optional) Two-agent additional sharing.
            if self.num_agents == 2:
                mutual_weight = 0.5
                final_rewards[0] += mutual_weight * final_rewards[1]
                final_rewards[1] += mutual_weight * final_rewards[0]

            # Execute yellow phases as needed.
            if self._step != 0:
                for i in range(self.num_agents):
                    if old_actions[i] is not None and old_actions[i] != actions[i]:
                        self._set_yellow_phase(i, old_actions[i], log_file)
                        self._simulate(self._yellow_duration, log_file)

            # Set green phases.
            for i, action in enumerate(actions):
                self._set_green_phase(i, action, log_file)

            # Compute adaptive green duration using first agent’s state.
            adaptive_green = self._compute_adaptive_green_duration(states[0])
            self._green_durations_log.append(adaptive_green)
            log_file.write(f"[Step {self._step}] Adaptive green duration: {adaptive_green}\n")
            self._simulate(adaptive_green, log_file)

            old_states = states
            old_actions = actions
            old_waits = current_waits

            for r in final_rewards:
                if r < 0:
                    self._sum_neg_reward += r

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)
        self._save_episode_stats()
        self._write_summary_log(episode, epsilon, simulation_time)

        print("Training...")
        start_train_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
            # Perform centralized training with CTDE.
            self.train_ctde()
        training_time = round(timeit.default_timer() - start_train_time, 1)

        return simulation_time, training_time

    def _choose_action(self, state, epsilon, model):
        valid_action_indices = list(self.int_conf["phase_mapping"].keys())
        if random.random() < epsilon:
            return random.choice(valid_action_indices)
        q_vals = model.predict_one(state)[0]
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
        if agent_index < len(tl_ids):
            tlid = tl_ids[agent_index]
            try:
                logics = traci.trafficlight.getAllProgramLogics(tlid)
                if logics:
                    num_phases = len(logics[0].phases)
                    if yellow_phase < num_phases:
                        traci.trafficlight.setProgram(tlid, "0")
                        traci.trafficlight.setPhase(tlid, yellow_phase)
                        if log_file:
                            log_file.write(f"[SetYellow] TL {tlid}: Set yellow phase {yellow_phase} out of {num_phases} phases.\n")
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
        if agent_index < len(tl_ids):
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

    def _replay(self):
        """
        Update each agent's network by sampling its own experiences from memory.
        """
        for agent_index in range(self.num_agents):
            batch_size = self._Models[agent_index].batch_size
            batch = self._Memory.get_samples_by_agent(agent_index, batch_size)
            if len(batch) == 0:
                print(f"[Replay] Not enough samples for agent {agent_index}. Skipping training update.")
                continue

            state_list = []
            next_state_list = []
            actions = []
            rewards = []
            for sample in batch:
                # Each sample: (agent_id, state, action, reward, next_state, global_state)
                _, st, act, rew, nst, _ = sample
                state_list.append(st)
                next_state_list.append(nst)
                actions.append(act)
                rewards.append(rew)

            states = self._pad_states(state_list)
            next_states = self._pad_states(next_state_list)
            actions = np.array(actions, dtype=np.int32)
            rewards = np.array(rewards, dtype=np.float32)

            q_s_a = self._Models[agent_index].predict_batch(states)
            best_next_actions = np.argmax(self._Models[agent_index].predict_batch(next_states), axis=1)
            target_q_next = self._TargetModels[agent_index].predict_batch(next_states)
            target_q_vals = target_q_next[np.arange(len(batch)), best_next_actions]

            y = np.copy(q_s_a)
            y[np.arange(len(batch)), actions] = rewards + self._gamma * target_q_vals

            loss = np.mean(np.square(y - q_s_a))
            self._q_loss_log.append(loss)
            self._Models[agent_index].train_batch(states, y)

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
        except Exception as e:
            print(f"Error writing summary log: {e}")

    def train_ctde(self):
        """
        Centralized Training with Decentralized Execution (CTDE).

        In this implementation:
          1. A joint batch is sampled from the shared memory.
          2. The centralized critic is trained on (global_state, reward) pairs.
             (For simplicity, we use the immediate reward as the target.)
          3. For each agent, an auxiliary loss term is computed as the mean-squared error
             between the agent's Q value (for the taken action) and the centralized critic's
             prediction on the corresponding global state.
          4. In a full implementation, you would combine the DQN loss with this auxiliary term
             and update the agent's network. Here, we simply print the alignment loss.
        """
        # 1. Sample joint experiences (without filtering by agent).
        joint_batch = self._Memory.get_samples(self._Models[0].batch_size)
        if len(joint_batch) < self._Memory._size_min:
            print("[CTDE] Not enough samples for centralized training.")
            return

        global_states = []
        rewards = []
        for sample in joint_batch:
            # sample: (agent_id, state, action, reward, next_state, global_state)
            _, _, _, rew, _, g_st = sample
            global_states.append(g_st)
            rewards.append(rew)
        global_states = np.array(global_states)
        rewards = np.array(rewards).reshape(-1, 1)

        # 2. Train the centralized critic.
        target = rewards  # For simplicity, we set target = reward (no bootstrapping).
        critic_loss = self.centralized_critic.train_on_batch(global_states, target)
        # print(f"[CTDE] Centralized critic loss: {critic_loss}")

        # 3. For each agent, compute the auxiliary critic alignment loss.
        for agent_index in range(self.num_agents):
            batch_size = self._Models[agent_index].batch_size
            agent_batch = self._Memory.get_samples_by_agent(agent_index, batch_size)
            if not agent_batch:
                continue
            state_list = []
            actions = []
            global_states_agent = []
            for sample in agent_batch:
                # sample: (agent_id, state, action, reward, next_state, global_state)
                _, st, act, _, _, g_st = sample
                state_list.append(st)
                actions.append(act)
                global_states_agent.append(g_st)
            states = self._pad_states(state_list)
            actions = np.array(actions, dtype=np.int32)
            global_states_agent = np.array(global_states_agent)
            # Get Q-values from the agent's network.
            q_vals = self._Models[agent_index].predict_batch(states)
            # Extract Q value for each taken action.
            q_taken = q_vals[np.arange(len(actions)), actions].reshape(-1, 1)
            # Get centralized critic prediction.
            critic_preds = self.centralized_critic(global_states_agent)
            # Compute auxiliary loss.
            alignment_loss = np.mean((q_taken - critic_preds.numpy()) ** 2)
            # print(f"[CTDE] Agent {agent_index} critic alignment loss: {alignment_loss}")
            # In practice, you would combine this loss with the standard DQN loss
            # and perform a joint gradient update on the agent's network.

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