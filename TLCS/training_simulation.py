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
        self._num_states = num_states
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
        Runs one simulation episode, then trains.
        Logs Q-values when emergency flags are active.
        """
        os.makedirs("logs", exist_ok=True)
        log_file = open(f"logs/episode_{episode}.log", "w")

        start_time = timeit.default_timer()
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

        # Number of lane groups for emergency flags
        num_emergency_flags = len(sorted(self.int_conf["incoming_lanes"].keys()))

        while self._step < self._max_steps:
            # Check for emergency vehicles (user-defined)
            if check_emergency(self):
                continue

            current_state = self._get_state()

            # If any emergency flags or waiting lanes triggered, log Q-values
            if np.sum(current_state[-(num_emergency_flags + 9):]) > 0:
                q_values = self._Model.predict_one(current_state)
                print("Emergency state detected. Q-values:", q_values)
                self._emergency_q_logs.append(q_values)

            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            action = self._choose_action(current_state, epsilon)

            # For logging
            # We'll assume the final entry in state is the "first" traffic light's phase
            current_phase = int(current_state[-1])
            log_file.write(f"[Step {self._step}] Action: {action}, Phase: {current_phase}\n")

            # If action changed, apply a yellow phase
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

        # Train after the episode
        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        # If we recorded Q-values in emergency states, log average
        if len(self._emergency_q_logs) > 0:
            avg_emergency_q = np.mean(np.array(self._emergency_q_logs), axis=0)
            print("Average Q-values for emergency states:", avg_emergency_q)

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
        """
        Retrieve accumulated waiting times for vehicles in the relevant inbound lanes.
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
        Epsilon-greedy action selection.
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            return np.argmax(self._Model.predict_one(state))

    def _set_green_phase(self, action_number):
        phase_config = self.int_conf["phase_mapping"][action_number]["green"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        # If no traffic lights, skip
        if not tl_ids:
            return
        for tlid in tl_ids:
            traci.trafficlight.setPhase(tlid, phase_config)

    def _set_yellow_phase(self, action_number):
        phase_config = self.int_conf["phase_mapping"][action_number]["yellow"]
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        if not tl_ids:
            return
        for tlid in tl_ids:
            traci.trafficlight.setPhase(tlid, phase_config)

    def _get_queue_length(self):
        """
        Sum of halting vehicles across all inbound lanes.
        """
        incoming_lane_ids = []
        for lanes in self.int_conf["incoming_lanes"].values():
            incoming_lane_ids.extend(lanes)
        total_queue = 0
        for lane in incoming_lane_ids:
            total_queue += traci.lane.getLastStepHaltingNumber(lane)
        return total_queue

    def _get_state(self):
        """
        Build state vector from:
         - occupancy grid (cars in each lane)
         - emergency flags
         - halted + waiting from monitor_edges / monitor_lanes
         - traffic light phase(s)
        """
        grid_conf = self.int_conf["occupancy_grid"]
        cells_per_lane = grid_conf["cells_per_lane"]
        max_distance   = grid_conf["max_distance"]

        # 1) Occupancy
        incoming_lanes = self.int_conf["incoming_lanes"]
        lane_order = []
        for group in sorted(incoming_lanes.keys()):
            lane_order.extend(incoming_lanes[group])
        total_lanes = len(lane_order)
        occupancy_length = total_lanes * cells_per_lane
        occupancy = np.zeros(occupancy_length)

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            # skip emergency in occupancy
            if traci.vehicle.getTypeID(car_id) == "emergency":
                continue
            lane_id = traci.vehicle.getLaneID(car_id)
            if lane_id in lane_order:
                lane_index  = lane_order.index(lane_id)
                lane_length = traci.lane.getLength(lane_id)
                lane_pos    = traci.vehicle.getLanePosition(car_id)
                dist_to_tls = lane_length - lane_pos
                cell = int((dist_to_tls / max_distance) * cells_per_lane)
                cell = min(cell, cells_per_lane - 1)
                occupancy_index = lane_index * cells_per_lane + cell
                occupancy[occupancy_index] = 1

        # 2) Emergency flags
        emergency_flags = []
        for group in sorted(incoming_lanes.keys()):
            flag = 0
            for lane in incoming_lanes[group]:
                for car_id in car_list:
                    if (traci.vehicle.getTypeID(car_id) == "emergency" and
                        traci.vehicle.getLaneID(car_id) == lane):
                        flag = 1
                        break
                if flag == 1:
                    break
            emergency_flags.append(flag)
        emergency_flags = np.array(emergency_flags)

        # 3) Halted / waiting
        monitor_edges = self.int_conf.get("monitor_edges", [])
        monitor_lanes = self.int_conf.get("monitor_lanes", [])
        halted  = [traci.edge.getLastStepHaltingNumber(e) for e in monitor_edges]
        waiting = [traci.lane.getWaitingTime(l) for l in monitor_lanes]

        # 4) Current traffic light phases (could be 0 or 4 lights, etc.)
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        phase_list = []
        for tlid in tl_ids:
            phase_list.append(traci.trafficlight.getPhase(tlid))
        if len(phase_list) == 0:
            # No signals => store just [0]
            current_phase = [0]
        elif len(phase_list) == 1:
            # Exactly one light
            current_phase = [phase_list[0]]
        else:
            # Multiple traffic lights
            # we can store them all if we want
            current_phase = phase_list

        # 5) Final concatenation
        state = np.concatenate((
            occupancy,         # occupancy grid
            emergency_flags,   # 1 flag per lane group
            halted,            # halting vehicles per monitored edge
            waiting,           # waiting times per monitored lane
            current_phase      # phase(s)
        ))
        return state

    def _replay(self):
        """
        Sample from replay memory and update the DQN.
        """
        batch = self._Memory.get_samples(self._Model.batch_size)
        if len(batch) == 0:
            return

        states      = np.array([sample[0] for sample in batch])
        next_states = np.array([sample[3] for sample in batch])
        q_s_a       = self._Model.predict_batch(states)
        q_s_a_next  = self._Model.predict_batch(next_states)

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
