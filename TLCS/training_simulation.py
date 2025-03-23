import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # action 3 code 11
PHASE_EWL_YELLOW = 7

from emergency_handler import check_emergency

class Simulation:
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states  # should be 84 (80 occupancy + 4 emergency flags)
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        # New: list to log Q-values when emergency flags are active
        self._emergency_q_logs = []

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session.
        Also logs Q-values when emergency flags are active.
        """
        start_time = timeit.default_timer()

        # route file for this simulation and set up sumo
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

            # Check for emergency vehicles using the refactored function
            if check_emergency(self):
                continue

            # get current state of the intersection (including emergency flags)
            current_state = self._get_state()

            # Log Q-values if emergency flag is active (if any of the last 4 entries is 1)
            if np.sum(current_state[-4:]) > 0:
                q_values = self._Model.predict_one(current_state)
                print("Emergency state detected. Q-values:", q_values)
                self._emergency_q_logs.append(q_values)

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
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

        # Log average Q-values for emergency states for analysis
        if len(self._emergency_q_logs) > 0:
            avg_emergency_q = np.mean(np.array(self._emergency_q_logs), axis=0)
            print("Average Q-values for emergency states:", avg_emergency_q)

        return simulation_time, training_time

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length  # 1 queued vehicle per step = 1 waited second

    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)
            if road_id in incoming_roads:
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time

    def _choose_action(self, state, epsilon):
        """
        Decide whether to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            return np.argmax(self._Model.predict_one(state))

    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase("TL", yellow_phase_code)

    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
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
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("N2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("S2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("E2TL")
        halt_W = traci.edge.getLastStepHaltingNumber("W2TL")
        queue_length = halt_N + halt_S + halt_E + halt_W
        return queue_length

    def _get_state(self):
        """
        Retrieve the state of the intersection from SUMO, including emergency vehicle presence.
        The state vector is composed of two parts:
          1. An occupancy grid (of length self._num_states - 4) for standard vehicles.
          2. Four binary flags indicating if an emergency vehicle is present on [N, S, E, W] incoming lanes.
        """
        # Occupancy grid for standard (non-emergency) vehicles
        occupancy_grid_length = self._num_states - 4  # expected to be 80
        occupancy = np.zeros(occupancy_grid_length)
        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            # Skip emergency vehicles in occupancy grid calculation
            if traci.vehicle.getTypeID(car_id) == "emergency":
                continue

            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # invert lane position: 0 means near the traffic light

            # Map lane position to cells
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

            # Determine lane group based on lane_id
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
                car_position = int(str(lane_group) + str(lane_cell))  # yields a number in [10, 79]
                valid_car = True
            elif lane_group == 0:
                car_position = lane_cell  # yields a number in [0,9]
                valid_car = True
            else:
                valid_car = False

            if valid_car and car_position < occupancy_grid_length:
                occupancy[car_position] = 1

        # Emergency vehicle flags (order: [N, S, E, W])
        emergency_flags = np.zeros(4)
        for car_id in car_list:
            if traci.vehicle.getTypeID(car_id) == "emergency":
                lane_id = traci.vehicle.getLaneID(car_id)
                if lane_id.startswith("N2TL"):
                    emergency_flags[0] = 1
                elif lane_id.startswith("S2TL"):
                    emergency_flags[1] = 1
                elif lane_id.startswith("E2TL"):
                    emergency_flags[2] = 1
                elif lane_id.startswith("W2TL"):
                    emergency_flags[3] = 1

        # Combine occupancy grid and emergency flags
        state = np.concatenate((occupancy, emergency_flags))
        return state

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each update the Q-learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:
            states = np.array([val[0] for val in batch])
            next_states = np.array([val[3] for val in batch])

            # Predict Q-values for current and next states
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
        Save episode statistics for later visualization
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
