import traci
import numpy as np
import random
import timeit
import os

from emergency_handler import check_emergency
import intersection_config as int_config

class Simulation:
    def __init__(
        self,
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        gamma,
        max_steps,
        green_duration,
        yellow_duration,
        num_states,         # Not actually used here with aggregator, can be ignored
        num_actions,
        training_epochs,
        intersection_type="cross"
    ):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states      # not used in aggregator approach
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._emergency_q_logs = []
        self._waiting_times = {}

        # Load the intersection configuration
        self.intersection_type = intersection_type
        if self.intersection_type not in int_config.INTERSECTION_CONFIGS:
            raise ValueError(f"Intersection type '{self.intersection_type}' not found in config.")
        self.int_conf = int_config.INTERSECTION_CONFIGS[self.intersection_type]

    def run(self, episode, epsilon):
        """
        1) Generate routes and launch SUMO.
        2) At each step, build a lane-based state (shape: (num_lanes, lane_feature_dim)),
           choose an action, step the environment, and store the transition.
        3) After the episode, perform training.
        """
        os.makedirs("logs", exist_ok=True)
        log_file = open(f"logs/episode_{episode}.log", "w")

        start_time = timeit.default_timer()
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print(f"Simulating Episode {episode} on {self.intersection_type}...")

        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = None
        old_action = None

        while self._step < self._max_steps:
            if check_emergency(self):
                continue

            # Build adaptive lane-based state: shape = (num_lanes, lane_feature_dim)
            current_state = self._get_state()

            current_total_wait = self._collect_waiting_times()
            reward = 0.0
            if self._step != 0:
                reward = float(old_total_wait - current_total_wait)

            if self._step != 0:
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

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length

    def _collect_waiting_times(self):
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

    def _choose_action(self, state, epsilon):
        # state is shape (num_lanes, lane_feature_dim)
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            q_vals = self._Model.predict_one(state)  # shape: (1, num_actions)
            return int(np.argmax(q_vals[0]))

    def _set_green_phase(self, action_number):
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        green_phase = phase_map[action_number]["green"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            traci.trafficlight.setPhase(tlid, green_phase)

    def _set_yellow_phase(self, action_number):
        if "phase_mapping" not in self.int_conf:
            return
        phase_map = self.int_conf["phase_mapping"]
        if action_number not in phase_map:
            return
        yellow_phase = phase_map[action_number]["yellow"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            traci.trafficlight.setPhase(tlid, yellow_phase)

    def _get_queue_length(self):
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        total_queue = 0
        for lane in incoming_lane_ids:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
        return total_queue

    def _get_state(self):
        """
        Build lane-based features for each lane.
        Returns a 2D array with shape (num_lanes, lane_feature_dim),
        where lane_feature_dim = 5, containing:
          0) occupancy_count: number of vehicles on this lane.
          1) waiting_time: from traci.lane.getWaitingTime.
          2) emergency_flag: 1 if at least one emergency vehicle is present.
          3) halted_count: from traci.lane.getLastStepHaltingNumber.
          4) traffic_light_phase: replicate global phase (or 0 if none).
        """
        incoming_lanes = self.int_conf["incoming_lanes"]
        lane_order = []
        for group in sorted(incoming_lanes.keys()):
            lane_order.extend(incoming_lanes[group])
        num_lanes = len(lane_order)
        lane_feature_dim = 5
        lane_features = np.zeros((num_lanes, lane_feature_dim), dtype=np.float32)

        # 1) occupancy_count
        for i, lane_id in enumerate(lane_order):
            lane_features[i, 0] = traci.lane.getLastStepVehicleNumber(lane_id)
        # 2) waiting_time
        for i, lane_id in enumerate(lane_order):
            lane_features[i, 1] = traci.lane.getWaitingTime(lane_id)
        # 3) emergency_flag
        for i, lane_id in enumerate(lane_order):
            flag = 0
            for car_id in traci.lane.getLastStepVehicleIDs(lane_id):
                if traci.vehicle.getTypeID(car_id) == "emergency":
                    flag = 1
                    break
            lane_features[i, 2] = float(flag)
        # 4) halted_count
        for i, lane_id in enumerate(lane_order):
            lane_features[i, 3] = traci.lane.getLastStepHaltingNumber(lane_id)
        # 5) traffic_light_phase: replicate the phase of the first traffic light (if any)
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        phase_val = 0.0
        if tl_ids:
            phase_val = float(traci.trafficlight.getPhase(tl_ids[0]))
        for i in range(num_lanes):
            lane_features[i, 4] = phase_val

        return lane_features

    def _pad_states(self, state_list):
        """
        Pad a list of lane-state arrays (each shape: (num_lanes, lane_feature_dim))
        so that they all have the same number of lanes (equal to the max in the batch).
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

    def _replay(self):
        """
        Sample from replay memory and train.
        The aggregator model expects states shaped (batch_size, num_lanes, lane_feature_dim).
        Since different samples may have different numbers of lanes, we pad them per batch.
        """
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

        # Pad states so that all samples in the batch have the same number of lanes
        states = self._pad_states(state_list)       # shape: (batch_size, max_num_lanes, lane_feature_dim)
        next_states = self._pad_states(next_state_list)
        actions = np.array(actions, dtype=np.int32)
        rewards = np.array(rewards, dtype=np.float32)

        q_s_a = self._Model.predict_batch(states)       # shape: (batch_size, num_actions)
        q_s_a_next = self._Model.predict_batch(next_states)

        y = np.copy(q_s_a)
        for i in range(len(batch)):
            y[i, actions[i]] = rewards[i] + self._gamma * np.max(q_s_a_next[i])

        self._Model.train_batch(states, y)

    def _save_episode_stats(self):
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)

    @property
    def reward_store(self):
        return self._reward_store

    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
