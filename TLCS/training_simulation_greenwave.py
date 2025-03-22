import traci
import numpy as np
import random
import timeit
import os

# Phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # North-South Green
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # North-South Left Green
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # East-West Green
PHASE_EW_YELLOW = 5
PHASE_EWL_GREEN = 6  # East-West Left Green
PHASE_EWL_YELLOW = 7


class SimulationGreenWave:
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
        self._num_actions = num_actions
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs

        # ✅ Initialize Traffic Light List
        traci.start(self._sumo_cmd, label="init")
        self.tls_list = traci.trafficlight.getIDList()
        traci.close()

    def run(self, episode, epsilon):
        """
        Runs one episode of simulation, then a training phase.
        """
        start_time = timeit.default_timer()

        # Ensure Route File Exists
        route_file = "intersection/episode_routes.rou.xml"
        self._TrafficGen.generate_routefile(seed=episode)
        if not os.path.exists(route_file):
            print("ERROR: Route file does NOT exist! TrafficGenerator failed.")
            return None, None

        # Start SUMO
        traci.start(self._sumo_cmd)
        print("\n🔹 Simulating Green Wave Optimization...")
        print(f"Traffic Lights Found: {self.tls_list}")

        self._step = 0
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = None
        old_action = None

        while self._step < self._max_steps:
            current_state = self._get_state()
            current_total_wait, emergency_penalty = self._collect_waiting_times()

            # Green Wave Reward Adjustment
            green_wave_bonus = self._green_wave_bonus()
            queue_penalty = self._queue_penalty()

            reward = (old_total_wait - current_total_wait) - emergency_penalty + green_wave_bonus

            if self._step != 0 and old_state is not None:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # Choose action using epsilon-greedy policy
            action = self._choose_action(current_state, epsilon)

            # Debug: Print chosen action
            print(f"Chosen Action: {action}")

            # If phase changes, introduce yellow before switching
            if self._step != 0 and old_action is not None and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # Apply action
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print(f"Total Reward: {self._sum_neg_reward}, Epsilon: {round(epsilon, 2)}")
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training Model...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _queue_penalty(self):
        """
        Penalizes long queues to encourage more throughput.
        """
        queue_length = self._get_queue_length()
        return queue_length * 0.1  # Adjust weight if needed
    
    def _green_wave_bonus(self):
        """
        Rewards if traffic lights are synchronized and traffic moves smoothly.
        """
        moving_vehicles = sum(traci.edge.getLastStepVehicleNumber(edge) for edge in ["E2TL", "N2TL", "S2TL", "W2TL"])
        return moving_vehicles * 1.5  # Adjust weight if needed

    
    def _simulate(self, steps_todo):
        """
        Executes SUMO simulation steps.
        """
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()
            self._step += 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length
            steps_todo -= 1

    def _set_yellow_phase(self, old_action):
        """
        Sets the yellow phase before switching.
        """
        yellow_phase_map = {
            0: PHASE_NS_YELLOW,
            1: PHASE_NSL_YELLOW,
            2: PHASE_EW_YELLOW,
            3: PHASE_EWL_YELLOW
        }

        yellow_phase = yellow_phase_map.get(old_action, PHASE_NS_YELLOW)  # Default to NS Yellow if unknown

        for tls in self.tls_list:
            traci.trafficlight.setPhase(tls, yellow_phase)

    def _set_green_phase(self, action):
        """
        Sets the correct green phase for traffic lights.
        """
        green_phase_map = {
            0: PHASE_NS_GREEN,
            1: PHASE_NSL_GREEN,
            2: PHASE_EW_GREEN,
            3: PHASE_EWL_GREEN
        }

        green_phase = green_phase_map.get(action, PHASE_NS_GREEN)  # Default to NS Green if unknown

        for tls in self.tls_list:
            traci.trafficlight.setPhase(tls, green_phase)

    def _save_episode_stats(self):
        """
         FIXED: Added missing function to save episode statistics.
        """
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)
    
    def _replay(self):
        """
         FIXED: Added missing `_replay()` function.
        Updates Q-values using experience replay.
        """
        batch = self._Memory.get_samples(self._Model.batch_size)
        if batch:
            states = np.array([b[0] for b in batch])
            next_states = np.array([b[3] for b in batch])
            q_s_a = self._Model.predict_batch(states)
            q_s_a_d = self._Model.predict_batch(next_states)

            for i, (state, action, reward, _) in enumerate(batch):
                q_s_a[i, action] = reward + self._gamma * np.amax(q_s_a_d[i])

            self._Model.train_batch(states, q_s_a)
    
    def _get_queue_length(self):
        """
        Retrieves the number of stopped vehicles.
        """
        return sum(traci.edge.getLastStepHaltingNumber(edge) for edge in ["E2TL", "N2TL", "S2TL", "W2TL"])

    def _get_state(self):
        """
        Retrieves the state of the intersection.
        """
        state = np.zeros(self._num_states)

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 750 - lane_pos  # Invert position
            lane_cell = min(9, int(lane_pos // 75))  # 10 bins

            lane_groups = {"W2TL": 0, "N2TL": 1, "E2TL": 2, "S2TL": 3}
            lane_group = lane_groups.get(lane_id[:4], -1)

            if lane_group >= 0:
                index = lane_group * 10 + lane_cell
                if index < len(state):
                    state[index] = 1
                else:
                    print(f"⚠ WARNING: Index {index} is out of range.")

        return state

    def _green_wave_bonus(self):
        """
        Rewards if traffic lights are synchronized.
        """
        reward = 0
        for tls in self.tls_list:
            if traci.trafficlight.getPhase(tls) in [PHASE_NS_GREEN, PHASE_NSL_GREEN, PHASE_EW_GREEN, PHASE_EWL_GREEN]:
                reward += 5  # Adjust as needed
        return reward

    def _choose_action(self, state, epsilon):
        """
        Epsilon-greedy action selection.
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            return np.argmax(self._Model.predict_one(state))

    def _collect_waiting_times(self):
        """
        Retrieves total waiting time and emergency penalty.
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
                if traci.vehicle.getTypeID(car_id) == "emergency":
                    emergency_penalty += wait_time * 0.2
        return total_waiting_time, emergency_penalty
