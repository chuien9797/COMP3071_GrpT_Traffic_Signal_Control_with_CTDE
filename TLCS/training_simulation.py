import traci
import numpy as np
import random
import timeit
import os

from emergency_handler import check_emergency
import intersection_config as int_config


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps,
                 green_duration, yellow_duration, num_states, num_actions, training_epochs,
                 intersection_type="cross"):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states  # this value can be adjusted dynamically if needed
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._emergency_q_logs = []
        self._waiting_times = {}

        # Initialize attributes for simulation statistics
        self._sum_queue_length = 0
        self._sum_waiting_time = 0

        # Load the appropriate intersection configuration based on type
        self.intersection_type = intersection_type
        if self.intersection_type not in int_config.INTERSECTION_CONFIGS:
            raise ValueError("Intersection type '{}' not found in configuration.".format(self.intersection_type))
        self.int_conf = int_config.INTERSECTION_CONFIGS[self.intersection_type]

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session.
        Also logs Q-values when emergency flags are active.
        """
        start_time = timeit.default_timer()

        # Generate route file and start SUMO simulation
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = None
        old_action = None

        # Determine the number of emergency flags from the configuration (one per lane group)
        num_emergency_flags = len(sorted(self.int_conf["incoming_lanes"].keys()))

        while self._step < self._max_steps:
            if check_emergency(self):
                continue

            current_state = self._get_state()

            if np.sum(current_state[-num_emergency_flags:]) > 0:
                q_values = self._Model.predict_one(current_state)
                print("Emergency state detected. Q-values:", q_values)
                self._emergency_q_logs.append(q_values)

            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            action = self._choose_action(current_state, epsilon)

            if self._step != 0 and (old_action is not None and old_action != action):
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

        if len(self._emergency_q_logs) > 0:
            avg_emergency_q = np.mean(np.array(self._emergency_q_logs), axis=0)
            print("Average Q-values for emergency states:", avg_emergency_q)

        return simulation_time, training_time

    def _simulate(self, steps_todo):
        """
        Execute a number of simulation steps in SUMO while gathering statistics.
        """
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
        """
        Retrieve the accumulated waiting time for vehicles in the incoming lanes.
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
        return sum(self._waiting_times.values())

    def _choose_action(self, state, epsilon):
        """
        Choose an action using an epsilon-greedy policy.
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, action_number):
        valid_index = action_number % len(self.int_conf["phase_mapping"])
        phase_config = self.int_conf["phase_mapping"][valid_index]["yellow"]
        if self.intersection_type == "roundabout":
            for tl in ["TL1", "TL2", "TL3", "TL4"]:
                traci.trafficlight.setPhase(tl, phase_config)
        else:
            traci.trafficlight.setPhase("TL", phase_config)

    def _set_green_phase(self, action_number):
        valid_index = action_number % len(self.int_conf["phase_mapping"])
        phase_config = self.int_conf["phase_mapping"][valid_index]["green"]
        if self.intersection_type == "roundabout":
            for tl in ["TL1", "TL2", "TL3", "TL4"]:
                traci.trafficlight.setPhase(tl, phase_config)
        else:
            traci.trafficlight.setPhase("TL", phase_config)

    def _get_queue_length(self):
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        total_queue = 0
        available_lanes = traci.lane.getIDList()
        for lane in incoming_lane_ids:
            if lane in available_lanes:
                total_queue += traci.lane.getLastStepHaltingNumber(lane)
            else:
                print("Warning: lane {} not found in simulation".format(lane))
        return total_queue

    def _get_state(self):
        """
        Retrieve the current state of the intersection, composed of:
          1. An occupancy grid for standard vehicles.
          2. Emergency vehicle flags (one per lane group).
        """
        grid_conf = self.int_conf["occupancy_grid"]
        cells_per_lane = grid_conf["cells_per_lane"]
        max_distance = grid_conf["max_distance"]

        incoming_lanes = self.int_conf["incoming_lanes"]
        lane_order = []
        for group in sorted(incoming_lanes.keys()):
            lane_order.extend(incoming_lanes[group])
        total_lanes = len(lane_order)
        occupancy_length = total_lanes * cells_per_lane
        occupancy = np.zeros(occupancy_length)

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            if traci.vehicle.getTypeID(car_id) == "emergency":
                continue
            lane_id = traci.vehicle.getLaneID(car_id)
            if lane_id in lane_order:
                lane_index = lane_order.index(lane_id)
                lane_length = traci.lane.getLength(lane_id)
                lane_pos = traci.vehicle.getLanePosition(car_id)
                normalized_pos = lane_length - lane_pos
                cell = int((normalized_pos / max_distance) * cells_per_lane)
                cell = min(cell, cells_per_lane - 1)
                occupancy_index = lane_index * cells_per_lane + cell
                occupancy[occupancy_index] = 1

        emergency_flags = []
        for group in sorted(incoming_lanes.keys()):
            flag = 0
            for lane in incoming_lanes[group]:
                for car_id in car_list:
                    if traci.vehicle.getTypeID(car_id) == "emergency" and traci.vehicle.getLaneID(car_id) == lane:
                        flag = 1
                        break
                if flag == 1:
                    break
            emergency_flags.append(flag)

        emergency_flags = np.array(emergency_flags)
        state = np.concatenate((occupancy, emergency_flags))
        return state

    def _replay(self):
        """
        Sample a batch from memory, update the Q-values using the Q-learning equation, and train the model.
        """
        batch = self._Memory.get_samples(self._Model.batch_size)
        if len(batch) > 0:
            states = np.array([sample[0] for sample in batch])
            next_states = np.array([sample[3] for sample in batch])
            q_s_a = self._Model.predict_batch(states)
            q_s_a_next = self._Model.predict_batch(next_states)
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))
            for i, sample in enumerate(batch):
                state, action, reward, _ = sample
                current_q = q_s_a[i]
                current_q[action] = reward + self._gamma * np.amax(q_s_a_next[i])
                x[i] = state
                y[i] = current_q
            self._Model.train_batch(x, y)

    def _save_episode_stats(self):
        """
        Save episode statistics for later visualization.
        """
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
