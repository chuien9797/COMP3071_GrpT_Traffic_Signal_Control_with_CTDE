import traci
import numpy as np
import random
import timeit
import os

# Phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7


class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration,
                 num_states, num_actions, training_epochs):
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
        self._num_actions = num_actions  # e.g., 4
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs

    def run(self, episode, epsilon):
        """
        Runs one episode of simulation, then a training phase.
        This version uses a negative reward style for emergency vehicles.
        """
        start_time = timeit.default_timer()

        # Generate the route file for this episode and start SUMO
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        self._step = 0
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1

        while self._step < self._max_steps:
            # Get the current state of the intersection (occupancy grid)
            current_state = self._get_state()

            # Collect waiting times and compute an extra penalty for emergency vehicles waiting.
            current_total_wait, emergency_penalty = self._collect_waiting_times()

            # Reward is the reduction in waiting time minus the emergency penalty.
            reward = (old_total_wait - current_total_wait) - emergency_penalty

            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # Choose the action using an epsilon-greedy policy
            action = self._choose_action(current_state, epsilon)

            # If phase changes, set yellow before executing new phase.
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # Execute the chosen green phase.
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
        """
        Executes a number of simulation steps and accumulates statistics.
        """
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length  # each queued car contributes 1 unit per step

    def _collect_waiting_times(self):
        """
        Retrieves the total waiting time of vehicles on incoming roads,
        and computes an extra penalty for emergency vehicles waiting.
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        total_waiting_time = 0
        emergency_penalty = 0
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                total_waiting_time += wait_time
                # If this vehicle is an emergency vehicle, add a heavy penalty.
                if traci.vehicle.getTypeID(car_id) == "emergency":
                    emergency_penalty += 100  # fixed penalty per emergency vehicle per step; adjust as needed
        return total_waiting_time, emergency_penalty

    def _choose_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        """
        Sets the yellow phase based on the previous action.
        """
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Sets the green phase corresponding to the chosen action.
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)
        elif action_number == 3:
            traci.trafficlight.setPhase("TL", PHASE_EWL_GREEN)

    def _get_queue_length(self):
        """
        Retrieves the total number of cars with speed 0 in all incoming lanes.
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        return halt_N + halt_S + halt_E + halt_W

    def _get_state(self):
        """
        Retrieves the state of the intersection as an occupancy grid.
        (You may later extend this state with additional emergency-related features.)
        """
        state = np.zeros(self._num_states)
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # invert lane position (closer to 0 means near the intersection)
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 60:
                lane_cell = 5
            elif lane_pos < 100:
                lane_cell = 6
            elif lane_pos < 160:
                lane_cell = 7
            elif lane_pos < 400:
                lane_cell = 8
            elif lane_pos <= 750:
                lane_cell = 9

            if lane_id in ["W2TL_0", "W2TL_1", "W2TL_2"]:
                lane_group = 0
            elif lane_id == "W2TL_3":
                lane_group = 1
            elif lane_id in ["N2TL_0", "N2TL_1", "N2TL_2"]:
                lane_group = 2
            elif lane_id == "N2TL_3":
                lane_group = 3
            elif lane_id in ["E2TL_0", "E2TL_1", "E2TL_2"]:
                lane_group = 4
            elif lane_id == "E2TL_3":
                lane_group = 5
            elif lane_id in ["S2TL_0", "S2TL_1", "S2TL_2"]:
                lane_group = 6
            elif lane_id == "S2TL_3":
                lane_group = 7
            else:
                lane_group = -1

            if lane_group >= 1 and lane_group <= 7:
                car_position = int(str(lane_group) + str(lane_cell))
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell
                valid_car = True
            else:
                valid_car = False

            if valid_car:
                state[car_position] = 1

        return state

    def _replay(self):
        """
        Retrieves a batch of samples from memory, updates Q-values, and trains the DQN.
        """
        batch = self._Memory.get_samples(self._Model.batch_size)
        if len(batch) > 0:
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])
            q_s_a = self._Model.predict_batch(states)
            q_s_a_d = self._Model.predict_batch(next_states)
            x = np.zeros((len(batch), self._num_states))
            y = np.zeros((len(batch), self._num_actions))
            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]
                current_q = q_s_a[i]
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])
                x[i] = state
                y[i] = current_q
            self._Model.train_batch(x, y)

    def _save_episode_stats(self):
        """
        Saves statistics for the episode.
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
