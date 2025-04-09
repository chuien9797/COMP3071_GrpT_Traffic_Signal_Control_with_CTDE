import traci
import numpy as np
import random
import timeit
import os
import matplotlib.pyplot as plt

from emergency_handler import check_emergency
import intersection_config as int_config


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
            num_actions,
            training_epochs,
            intersection_type="cross",
            signal_fault_prob=0.1,
            algorithm="DQN"  # "DQN" or "PPO"
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
        self._num_actions = num_actions
        self._training_epochs = training_epochs

        # For plotting statistics
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._emergency_q_logs = []
        self._waiting_times = {}
        self.signal_fault_prob = signal_fault_prob
        self.manual_override = False
        self.recovery_queue = {}
        self._emergency_wait_log = 0.0

        self.intersection_type = intersection_type
        if self.intersection_type not in int_config.INTERSECTION_CONFIGS:
            raise ValueError(f"Intersection type '{self.intersection_type}' not found in config.")
        self.int_conf = int_config.INTERSECTION_CONFIGS[self.intersection_type]

        # Save the algorithm ("DQN" or "PPO")
        self.algorithm = algorithm

    def run(self, episode, epsilon):
        os.makedirs("logs10", exist_ok=True)
        log_file = open(f"logs10/episode_{episode}.log", "w")
        start_time = timeit.default_timer()

        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print(f"Simulating Episode {episode} on {self.intersection_type}...")

        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_state = None
        old_action = None

        # For PPO, initialize waiting time measurement from initial state.
        if self.algorithm == "PPO":
            old_wait = self._collect_waiting_times()
        else:
            old_total_wait = 0

        self.faulty_lights = set()
        self.fault_injected_this_episode = False
        self.skip_fault_this_episode = random.random() < 0.5  # 50% episodes are clean

        # Prepare storage for PPO rollout:
        total_reward = 0.0
        trajectory = []  # Each element: (state, action, reward, log_prob, value)

        if self.algorithm == "DQN":
            while self._step < self._max_steps:
                if check_emergency(self):
                    continue
                current_state = self._get_state()
                current_total_wait = self._collect_waiting_times()
                reward = 0.0
                if self._step != 0:
                    reward = float(old_total_wait - current_total_wait)
                    self._Memory.add_sample((old_state, old_action, reward, current_state))
                action = self._choose_action(current_state, epsilon)
                log_file.write(f"[Step {self._step}] Action: {action}\n")
                if self._step != 0 and old_action is not None and old_action != action:
                    self._set_yellow_phase(old_action)
                    self._simulate(self._yellow_duration)
                self._set_green_phase(action)
                self._simulate(self._green_duration)
                old_state = current_state
                old_action = action
                old_total_wait = current_total_wait
                if reward < 0:
                    self._sum_neg_reward += reward
        elif self.algorithm == "PPO":
            # PPO rollout collection:
            old_wait = self._collect_waiting_times()
            while self._step < self._max_steps:
                if check_emergency(self):
                    continue
                state = self._get_state()
                action, log_prob, value = self._Model.act(state)
                log_file.write(f"[Step {self._step}] Action: {action}\n")
                if self._step != 0 and old_action is not None and old_action != action:
                    self._set_yellow_phase(old_action)
                    self._simulate(self._yellow_duration)
                self._set_green_phase(action)
                self._simulate(self._green_duration)
                new_wait = self._collect_waiting_times()
                queue_len = self._get_queue_length()
                switch = 1 if (old_action is not None and old_action != action) else 0
                # Sum the emergency flags (state[:,2] holds that information)
                emergency = int(np.sum(state[:, 2]))
                # Compute immediate reward using your reward function.
                reward = compute_reward(old_wait, new_wait, queue_len, switch, emergency)
                old_wait = new_wait

                trajectory.append((state, action, reward, log_prob, value))
                total_reward += reward
                old_state = state
                old_action = action

        self._save_episode_stats()
        if self.algorithm == "DQN":
            print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        elif self.algorithm == "PPO":
            print("Total reward:", total_reward)
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        # Training update.
        start_train = timeit.default_timer()
        if self.algorithm == "DQN":
            for _ in range(self._training_epochs):
                self._replay()
        elif self.algorithm == "PPO" and trajectory:
            # Unpack rollout: extract states, actions, rewards, log_probs, and values.
            states, actions, rewards, log_probs, values = zip(*trajectory)
            states = np.array(states)      # Shape: (N, num_lanes, lane_feature_dim)
            actions = np.array(actions, dtype=np.int32)
            rewards = np.array(rewards, dtype=np.float32)
            log_probs = np.array(log_probs, dtype=np.float32)
            values = np.array(values, dtype=np.float32)
            # Compute returns.
            returns = compute_discounted_returns(rewards, self._gamma)
            # Compute advantages as returns - value estimates.
            advantages = returns - values
            # Normalize advantages.
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
            self._Model.ppo_update(
                states=states,
                actions=actions,
                old_log_probs=log_probs,
                advantages=advantages,
                returns=returns
            )
        train_time = round(timeit.default_timer() - start_train, 1)

        return simulation_time, train_time

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step
        while steps_todo > 0:
            self._inject_signal_faults()
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            q_len = self._get_queue_length()
            self._sum_queue_length += q_len
            self._sum_waiting_time += q_len  # You may adjust if desired.

    def _inject_signal_faults(self):
        self.manual_override = False
        if self.skip_fault_this_episode or self.fault_injected_this_episode:
            return
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
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
                print(f"[Step {self._step}] ❌ Signal fault injected at TL={tlid}, phase={current_phase}")
                print(f"    Original state: {current_state}")
                print(f"    Modified state: {new_state_str}")
                print(f"    Flipped indices: {flipped_indices}")
                return

    def _collect_waiting_times(self):
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        car_list = traci.vehicle.getIDList()
        self._waiting_times = {}
        emergency_wait_time = 0.0
        normal_wait_time = 0.0
        for car_id in car_list:
            lane_id = traci.vehicle.getLaneID(car_id)
            if lane_id in incoming_lane_ids:
                wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
                self._waiting_times[car_id] = wait_time
                if traci.vehicle.getTypeID(car_id) == "emergency":
                    emergency_wait_time += wait_time
                else:
                    normal_wait_time += wait_time
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        self._emergency_wait_log = emergency_wait_time
        return normal_wait_time + (10.0 * emergency_wait_time)

    def _choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            q_vals = self._Model.predict_one(state)
            action = int(np.argmax(q_vals[0]))
        if np.any(state[:, 2] == 1.0):
            print(f"[Step {self._step}] 🚨 Emergency flag detected in state. Chosen action: {action}")
        return action

    def _set_green_phase(self, action_number):
        if hasattr(self, "manual_override") and self.manual_override:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        green_phase = phase_map[action_number]["green"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            if tlid not in self.faulty_lights:
                try:
                    traci.trafficlight.setProgram(tlid, "0")
                    traci.trafficlight.setPhase(tlid, green_phase)
                except traci.exceptions.TraCIException as e:
                    print(f"⚠️ Failed to set green phase {green_phase} for {tlid}: {e}")

    def _set_yellow_phase(self, action_number):
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        yellow_phase = phase_map[action_number]["yellow"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            logics = traci.trafficlight.getAllProgramLogics(tlid)
            if not logics:
                continue
            num_phases = len(logics[0].phases)
            if yellow_phase < num_phases:
                traci.trafficlight.setProgram(tlid, "0")
                traci.trafficlight.setPhase(tlid, yellow_phase)
            else:
                print(f"⚠️ Skipping invalid yellow phase {yellow_phase} for {tlid} (only {num_phases} phases)")

    def _get_queue_length(self):
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        return sum(traci.lane.getLastStepHaltingNumber(lane) for lane in incoming_lane_ids)

    def _get_state(self):
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
            "T_intersection": [0.0, 0.0, 1.0]
        }
        type_vector = intersection_encoding.get(self.intersection_type, [0.0, 0.0, 0.0])
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
            controlled_by_faulty_signal = 0.0
            if tl_ids:
                logics = traci.trafficlight.getAllProgramLogics(tl_ids[0])
                if logics:
                    logic = logics[0]
                    current_phase = traci.trafficlight.getPhase(tl_ids[0])
                    phase_state = logic.phases[current_phase].state
                else:
                    phase_state = ""
                if phase_state:
                    link_index = -1
                    try:
                        connections = traci.trafficlight.getControlledLanes(tl_ids[0])
                        if lane_id in connections:
                            link_index = connections.index(lane_id)
                    except:
                        pass
                    if link_index >= 0 and phase_state[link_index] == 'r':
                        controlled_by_faulty_signal = 1.0
            lane_features[i, 5] = controlled_by_faulty_signal
        return lane_features

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
        q_s_a = self._Model.predict_batch(states)
        best_next_actions = np.argmax(self._Model.predict_batch(next_states), axis=1)
        target_q_next = self._TargetModel.predict_batch(next_states)
        target_q_vals = target_q_next[np.arange(len(batch)), best_next_actions]
        y = np.copy(q_s_a)
        y[np.arange(len(batch)), actions] = rewards + self._gamma * target_q_vals
        self._Model.train_batch(states, y)

    def _save_episode_stats(self):
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)
        self._emergency_q_logs.append(self._emergency_wait_log)

    def analyze_results(self):
        plt.figure(figsize=(20, 5))
        plt.subplot(1, 4, 1)
        plt.plot(self.reward_store)
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.subplot(1, 4, 2)
        plt.plot(self.avg_queue_length_store)
        plt.title("Average Queue Length per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Queue Length")
        plt.subplot(1, 4, 3)
        plt.plot(self.cumulative_wait_store)
        plt.title("Cumulative Waiting Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Waiting Time")
        plt.subplot(1, 4, 4)
        plt.plot(self._emergency_q_logs)
        plt.title("Emergency Vehicle Waiting Time")
        plt.xlabel("Episode")
        plt.ylabel("Emergency Wait")
        plt.show()
        print(f"Final Faulty Lights: {self.faulty_lights}")

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store


# Helper function for PPO: compute discounted returns.
def compute_discounted_returns(rewards, gamma):
    discounted_returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + gamma * running_return
        discounted_returns[t] = running_return
    return discounted_returns


# NEW: Revised Helper function to compute immediate reward.
def compute_reward(old_wait, new_wait, queue_len, switch, emergency):
    """
    Compute a reward that reflects delay reduction, queue length, phase switching and emergencies.
    The waiting time difference is scaled to reduce variance and then penalized by queue and emergency factors.
    The output is clipped to a fixed range.
    """
    scaling_factor = 5000.0  # Scale down raw waiting time differences.
    diff = (old_wait - new_wait) / scaling_factor  # Positive if waiting time decreases.
    q_scaled = queue_len / scaling_factor

    raw_reward = diff + 0.4 * (diff - q_scaled) - 0.005 * q_scaled - 0.05 * switch - 2 * emergency
    reward = np.clip(raw_reward, -100.0, 100.0)
    return reward
