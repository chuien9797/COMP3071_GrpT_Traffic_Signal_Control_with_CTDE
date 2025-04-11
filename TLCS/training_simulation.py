import traci
import numpy as np
import random
import timeit
import os
import matplotlib.pyplot as plt
from datetime import datetime

from emergency_handler import check_emergency
import intersection_config as int_config

RECOVERY_DELAY = 15               # Steps to recover faulty signals
FAULT_REWARD_SCALE = 0.5         # Scale reward if fault occurs
EPISODE_FAULT_START = 25         # Start injecting faults only from episode 10

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
        self._reward_store = []
        self._cumulative_wait_store = []
        self._avg_queue_length_store = []
        self._training_epochs = training_epochs
        self._emergency_q_logs = []
        self._waiting_times = {}
        self.signal_fault_prob = signal_fault_prob  # Probability a 'G' flips to 'r'
        self.manual_override = False 
        self.recovery_queue = {}  # { (tlid, phase): { 'step': recover_at, 'original': 'GGG...r' } }
        self._green_durations_log = [] 
        self.fault_details = []  # store tuples like (step, tlid, original, modified)
        self._q_loss_log = []
        self._action_counts = np.zeros(self._num_actions, dtype=int)
        self._emergency_crossed = 0
        self._emergency_total_delay = 0.0
        self._teleport_count = 0      

        self.intersection_type = intersection_type
        if self.intersection_type not in int_config.INTERSECTION_CONFIGS:
            raise ValueError(f"Intersection type '{self.intersection_type}' not found in config.")
        self.int_conf = int_config.INTERSECTION_CONFIGS[self.intersection_type]

    def run(self, episode, epsilon):
        os.makedirs("logs14", exist_ok=True)
        log_file = open(f"logs14/episode_{episode}.log", "w")

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
        self.faulty_lights = set()
        self.fault_injected_this_episode = False

        self.skip_fault_this_episode = (episode < EPISODE_FAULT_START) or (random.random() < 0.5)
        
        start_time = timeit.default_timer()

        while self._step < self._max_steps:
            if check_emergency(self):
                continue

            current_state = self._get_state()
            current_total_wait = self._collect_waiting_times()
            reward = 0.0
            if self._step != 0:
                reward = float(old_total_wait - current_total_wait)

                if self.fault_injected_this_episode:
                    reward *= FAULT_REWARD_SCALE

                self._Memory.add_sample((old_state, old_action, reward, current_state))

            action = self._choose_action(current_state, epsilon)
            self._action_counts[action] += 1
            log_file.write(f"[Step {self._step}] Action: {action}\n")

            if self._step != 0 and old_action is not None and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            self._set_green_phase(action)
            # Compute adaptive green duration
            adaptive_green = self._compute_adaptive_green_duration(current_state)
            self._green_durations_log.append(adaptive_green)
            print(f"[Step {self._step}] Adaptive green duration: {adaptive_green}")
            log_file.write(f"[Step {self._step}] Adaptive green duration: {adaptive_green}\n")
            self._simulate(adaptive_green)

            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

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
    
    def _write_summary_log(self, episode, epsilon, sim_time):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(f"logs14/episode_{episode}_summary.log", "w", encoding="utf-8") as f:
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Intersection: {self.intersection_type}\n")
            f.write(f"Total reward: {self._sum_neg_reward:.2f}\n")
            f.write(f"Epsilon: {round(epsilon, 2)}\n")
            f.write(f"Simulation duration: {sim_time}s\n")
            f.write(f"Avg queue length: {self._sum_queue_length / self._max_steps:.2f}\n")
            f.write(f"Cumulative wait time: {self._sum_waiting_time:.2f}\n")
            f.write(f"Fault injected: {'Yes' if self.fault_injected_this_episode else '❌ No'}\n")
            if self.fault_details:
                f.write("\nFault Details:\n")
                for step, tlid, original, modified in self.fault_details:
                    f.write(f"Step {step} | TLID: {tlid} | Orig: {original} -> Mod: {modified}\n")
            f.write(f"\nEmergency Vehicles Crossed: {self._emergency_crossed}\n")
            f.write(f"Total Emergency Delay: {self._emergency_total_delay:.2f}\n")
            f.write(f"Teleports This Episode: {self._teleport_count}\n")
            f.write("\nAction Distribution:\n")
            for i, count in enumerate(self._action_counts):
                f.write(f"Action {i}: {count} times\n")


    def _recover_faults_if_due(self):
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
            key = (tlid, self._step)
            if key in self.recovery_queue:
                original_state = self.recovery_queue[key]
                try:
                    traci.trafficlight.setRedYellowGreenState(tlid, original_state)
                    print(f"[Step {self._step}] ✅ Signal recovered at TL={tlid}")
                    del self.recovery_queue[key]
                    self.manual_override = False
                except traci.exceptions.TraCIException:
                    pass

    def _compute_adaptive_green_duration(self, state):
        avg_wait = np.mean(state[:, 1])
        queue_length = np.sum(state[:, 3])
        base = self._green_duration
        wait_factor = int(avg_wait // 2)
        queue_factor = int(queue_length // 5)
        adaptive_extension = min(wait_factor + queue_factor, 10)
        return base + adaptive_extension
    

    def _simulate(self, steps_todo):
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            self._inject_signal_faults()  # Apply signal-level faults before each step
            self._recover_faults_if_due()
            traci.simulationStep()
            self._step += 1
            steps_todo -= 1
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length
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
            return  # Skip if clean episode or already injected

        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for tlid in tl_ids:
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
        for car_id in car_list:
            lane_id = traci.vehicle.getLaneID(car_id)
            if lane_id in incoming_lane_ids:
                self._waiting_times[car_id] = traci.vehicle.getAccumulatedWaitingTime(car_id)
            else:
                if car_id in self._waiting_times:
                    del self._waiting_times[car_id]
        return float(sum(self._waiting_times.values()))

    def _choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1)
        else:
            q_vals = self._Model.predict_one(state)
            return int(np.argmax(q_vals[0]))

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
                # Reset the traffic light to its default logic before setting the phase
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
                traci.trafficlight.setProgram(tlid, "0")  # 👈 reset to default before setting phase
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
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_ids[0])[0]
                current_phase = traci.trafficlight.getPhase(tl_ids[0])
                phase_state = logic.phases[current_phase].state

                # Try to find which signal index controls this lane
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
            # print(f"Lane {i} faulty light status: {lane_features[i,5]}")

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
        
        # Double DQN logic
        best_next_actions = np.argmax(self._Model.predict_batch(next_states), axis=1)
        target_q_next = self._TargetModel.predict_batch(next_states)
        target_q_vals = target_q_next[np.arange(len(batch)), best_next_actions]

        y = np.copy(q_s_a)
        y[np.arange(len(batch)), actions] = rewards + self._gamma * target_q_vals

        # ✅ Add Q-loss tracking here
        loss = np.mean(np.square(y - q_s_a))
        self._q_loss_log.append(loss)

        self._Model.train_batch(states, y)

    def _save_episode_stats(self):
        self._reward_store.append(self._sum_neg_reward)
        self._cumulative_wait_store.append(self._sum_waiting_time)
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)


    def analyze_results(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(self.reward_store)
        plt.title("Reward per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        plt.subplot(1, 3, 2)
        plt.plot(self.avg_queue_length_store)
        plt.title("Average Queue Length per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Average Queue Length")

        plt.subplot(1, 3, 3)
        plt.plot(self.cumulative_wait_store)
        plt.title("Cumulative Waiting Time per Episode")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Waiting Time")

        plt.show()

        print(f"Final Faulty Lights: {self.faulty_lights}")

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
