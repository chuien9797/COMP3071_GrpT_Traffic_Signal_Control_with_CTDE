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


# --------------------------------------------------------------------------- #
# Global constants
# --------------------------------------------------------------------------- #
RECOVERY_DELAY      = 15
FAULT_REWARD_SCALE  = 0.5
EPISODE_FAULT_START = int(150 * 0.3)


# --------------------------------------------------------------------------- #
# Centralised Critic
# --------------------------------------------------------------------------- #
class CentralizedCritic(tf.keras.Model):
    """Simple two‑hidden‑layer value network for CTDE."""
    def __init__(self, global_state_dim: int, hidden_units: int = 64):
        super().__init__()
        self.d1  = layers.Dense(hidden_units, activation='relu')
        self.d2  = layers.Dense(hidden_units, activation='relu')
        self.out = layers.Dense(1, activation='linear')
        # Explicitly build so Keras knows all shapes, avoiding warnings.
        self.build((None, global_state_dim))

    # A no‑op build keeps Keras happy when we call self.build() above.
    def build(self, input_shape):
        super().build(input_shape)

    def call(self, s):
        x = self.d1(s)
        x = self.d2(x)
        return self.out(x)                              # (batch, 1)


# --------------------------------------------------------------------------- #
# Simulation
# --------------------------------------------------------------------------- #
class TestingSimulation:
    """
    One SUMO environment wrapper capable of training **multiple agents that all
    share the same policy network**.  A centralised critic is used for CTDE.
    """
    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def __init__(self,
                 Models,                  # list[TrainModelAggregator] – all refs to ONE model
                 TargetModels,            # target‑network refs (can be the same list)
                 Memory,
                 TrafficGen,
                 sumo_cmd,
                 gamma,
                 max_steps,
                 green_duration,
                 yellow_duration,
                 num_states,              # kept for compatibility (unused)
                 training_epochs,
                 intersection_type="cross",
                 signal_fault_prob=0.1,
                 centralized_critic=None  # optional – if None build locally
                 ):
        # ------------ generic parameters -------------------------------- #
        self._Models          = Models
        self._TargetModels    = TargetModels
        self._Memory          = Memory
        self._TrafficGen      = TrafficGen
        self._gamma           = gamma
        self._step            = 0
        self._sumo_cmd        = sumo_cmd
        self._max_steps       = max_steps
        self._green_duration  = green_duration
        self._yellow_duration = yellow_duration
        self._num_states      = num_states          # not used with set‑based model
        self._training_epochs = training_epochs

        # ------------ intersection‑specific setup ----------------------- #
        self.intersection_type = intersection_type
        if self.intersection_type not in int_config.INTERSECTION_CONFIGS:
            raise ValueError(f"Unknown intersection type '{intersection_type}'")
        self.int_conf     = int_config.INTERSECTION_CONFIGS[self.intersection_type]
        self._num_actions = len(self.int_conf["phase_mapping"])
        tl_ids            = self.int_conf.get("traffic_light_ids", [])
        self.num_agents   = len(tl_ids) if isinstance(tl_ids, list) else 1

        # ------------ stats tracking  ----------------------------------- #
        self._reward_store           = []
        self._cumulative_wait_store  = []
        self._avg_queue_length_store = []
        self._q_loss_log             = []
        self._green_durations_log    = []
        self.fault_details           = []
        self.faulty_lights           = set()
        self._emergency_crossed      = 0
        self._emergency_total_delay  = 0.0
        self._teleport_count         = 0
        self._sum_queue_length       = 0
        self._sum_waiting_time       = 0
        self.fault_injection_events = []  # list of (tlid, step)
        self.fault_recovery_times = []  # list of (recovery_step - inject_step)
        self._emergency_total_delay = 0.0
        self._emergency_crossed = 0

        # ------------ misc ---------------------------------------------- #
        self.signal_fault_prob = signal_fault_prob
        self.manual_override   = False
        self.recovery_queue    = {}

        # ------------ centralised critic -------------------------------- #
        lane_feature_dim = 9
        total_lanes      = sum(len(lanes)
                               for lanes in self.int_conf["incoming_lanes"].values())
        global_state_dim = total_lanes * lane_feature_dim
        self._critic_input_dim = global_state_dim

        if centralized_critic is None:
            self.centralized_critic = CentralizedCritic(global_state_dim, hidden_units=64)
            self.centralized_critic.compile(loss='mse',
                                            optimizer=Adam(learning_rate=1e-3))
        else:
            self.centralized_critic = centralized_critic


    def _get_state(self):
        """
        Returns a list of state matrices—one per agent.  Each state has
        shape [num_lanes, 9].
        """
        groups = sorted(self.int_conf["incoming_lanes"].keys())
        states = []
        tl_ids = self.int_conf.get("traffic_light_ids", [])
        for a_idx, group in enumerate(groups):
            lanes      = self.int_conf["incoming_lanes"][group]
            Ln         = len(lanes)
            st         = np.zeros((Ln, 9), dtype=np.float32)
            for i, lane_id in enumerate(lanes):
                st[i, 0] = traci.lane.getLastStepVehicleNumber(lane_id)
                st[i, 1] = traci.lane.getWaitingTime(lane_id)
                st[i, 2] = float(any(traci.vehicle.getTypeID(v) == "emergency"
                                     for v in traci.lane.getLastStepVehicleIDs(lane_id)))
                st[i, 3] = traci.lane.getLastStepHaltingNumber(lane_id)
                st[i, 4] = float(traci.trafficlight.getPhase(tl_ids[a_idx])) if a_idx < len(tl_ids) else 0.0
                # fault flag
                f = 0.0
                if a_idx < len(tl_ids):
                    tlid = tl_ids[a_idx]
                    try:
                        lg   = traci.trafficlight.getAllProgramLogics(tlid)
                        if lg:
                            cp  = traci.trafficlight.getPhase(tlid)
                            pst = lg[0].phases[cp].state
                            con = traci.trafficlight.getControlledLanes(tlid)
                            if lane_id in con and pst[con.index(lane_id)] == 'r':
                                f = 1.0
                    except Exception:
                        pass
                st[i, 5] = f
                st[i, 6:9] = np.array([0.33, 0.33, 0.34])  # simple one‑hot placeholder
            states.append(st)
        return states

    def _get_global_state(self, states):
        flat = np.concatenate([s.flatten() for s in states])
        tgt  = self._critic_input_dim
        if flat.size < tgt:
            flat = np.pad(flat, (0, tgt - flat.size))
        else:
            flat = flat[:tgt]
        return flat

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
        os.makedirs("logs25", exist_ok=True)
        log_file = open(f"logs25/episode_{episode}.log", "w")

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

            # --- Newly Inserted Emergency Handling Code ---
            # For each agent, if an emergency vehicle is detected and this agent has not yet been handled,
            # force the traffic light to the emergency green phase (phase 0) and mark it as handled.
            for i in range(self.num_agents):
                if i not in self.handled_emergency_ids and np.any(states[i][:, 2] > 0):
                    handle_emergency_vehicle(self, agent_index=i)
                    self.handled_emergency_ids.add(i)
            # -----------------------------------------------------

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

            # Compute the next global state using the newly observed states.
            next_global_state = self._get_global_state(states)

            actions = []
            for i in range(self.num_agents):
                if self._step != 0 and old_states[i] is not None:
                    # Store experience: (agent_id, old_state, old_action, reward, current state, next_global_state)
                    self._Memory.add_sample(
                        (i, old_states[i], old_actions[i], final_rewards[i], states[i], next_global_state))
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

        # Convert to integer type explicitly
        valid_action_indices = np.array(valid_action_indices, dtype=int)

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

            # get list of vehicles that just arrived at their destination
            arrived = traci.simulation.getArrivedIDList()
            for veh_id in arrived:
                if traci.vehicle.getTypeID(veh_id) == "emergency":
                    # count one more cleared emergency vehicle
                    self._emergency_crossed += 1

            steps_todo -= 1
            q_len = self._get_queue_length()
            self._sum_queue_length += q_len
            current_wait = self._collect_waiting_times()
            self._sum_waiting_time += current_wait

            emergency_delay = 0.0
            for veh in traci.vehicle.getIDList():
                if traci.vehicle.getTypeID(veh) == "emergency":
                    emergency_delay += traci.vehicle.getAccumulatedWaitingTime(veh)
            self._emergency_total_delay += emergency_delay

            if log_file:
                log_file.write(f"[Simulate] Step {self._step}: Queue length {q_len}, Waiting time {current_wait}, Emergency Delay: {emergency_delay}\n")

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
                    self.fault_injection_events.append((tlid, self._step))
                    # schedule a recovery after RECOVERY_DELAY steps
                    recovery_step = self._step + RECOVERY_DELAY
                    self.recovery_queue[(tlid, recovery_step)] = current_state
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
                inject_step = next(
                    s for (t, s) in self.fault_injection_events
                    if t == tlid
                )
                recovery_time = self._step - inject_step
                self.fault_recovery_times.append(recovery_time)
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
        Sample across *all* agents and perform **one** SGD update on the shared
        policy network.  (Redundant updates on the same model are avoided.)
        """
        batch_size = self._Models[0].batch_size
        merged     = []
        for aid in range(self.num_agents):
            merged.extend(self._Memory.get_samples_by_agent(aid, batch_size))
        if len(merged) < self._Memory._size_min:
            print("[Replay] Not enough samples for shared update.")
            return

        # Build tensors
        states      = self._pad_states([s[1] for s in merged])
        next_states = self._pad_states([s[4] for s in merged])
        actions     = np.array([s[2] for s in merged], dtype=np.int32)
        rewards     = np.array([s[3] for s in merged], dtype=np.float32)

        shared      = self._Models[0]                 # TrainModelAggregator
        q_s         = shared.predict_batch(states)
        q_next_main = shared.predict_batch(next_states)
        best_next   = np.argmax(q_next_main, axis=1)
        q_next_tgt  = self._TargetModels[0].predict_batch(next_states)
        target_q    = q_next_tgt[np.arange(len(merged)), best_next]

        y           = np.copy(q_s)
        y[np.arange(len(merged)), actions] = rewards + self._gamma * target_q

        loss        = np.mean((y - q_s) ** 2)
        self._q_loss_log.append(loss)
        shared.train_batch(states, y)
        self._TargetModels[0].soft_update_from(self._Models[0], tau=0.005)

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
            with open(f"logs25/episode_{episode}_summary.log", "w", encoding="utf-8") as f:
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

    # ------------------------------------------------------------------ #
    # Centralized Training with ONE shared‑policy update
    # ------------------------------------------------------------------ #
    def train_ctde(self, lambda_coef: float = 0.5):
        """
        Centralized Training ➜ Decentralized Execution (CTDE) for a *shared*
        policy network.
        """
        # ---------- 1) critic update --------------------------------------- #
        critic_batch = self._Memory.get_samples(self._Models[0].batch_size)
        if len(critic_batch) < self._Memory._size_min:
            print("[CTDE] Not enough samples for centralized critic.")
            return

        g_states = np.array([s[-1] for s in critic_batch])
        rewards  = np.array([s[3]  for s in critic_batch]).reshape(-1, 1)

        critic_next = self.centralized_critic(g_states)
        targets     = rewards + self._gamma * critic_next.numpy()
        self.centralized_critic.train_on_batch(g_states, targets)

        # ---------- 2) policy alignment update ----------------------------- #
        merged = []
        batch_size = self._Models[0].batch_size
        for aid in range(self.num_agents):
            merged.extend(self._Memory.get_samples_by_agent(aid, batch_size))
        if not merged:
            print("[CTDE] No samples available for shared‑policy alignment.")
            return

        states  = self._pad_states([s[1] for s in merged])
        actions = np.array([s[2] for s in merged], dtype=np.int32)
        next_g  = np.array([s[-1] for s in merged])

        shared_model = self._Models[0].model
        with tf.GradientTape() as tape:
            q_vals  = shared_model(states, training=True)
            q_taken = tf.reduce_sum(q_vals * tf.one_hot(actions, q_vals.shape[-1]), axis=1, keepdims=True)
            critic_p = self.centralized_critic(next_g, training=False)
            align_loss = tf.reduce_mean(tf.square(q_taken - critic_p))

        grads = tape.gradient(align_loss, shared_model.trainable_variables)
        shared_model.optimizer.apply_gradients(zip(grads, shared_model.trainable_variables))
        # print(f"[CTDE] Alignment loss: {align_loss.numpy():.4f}")

    def analyze_results(self):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(self._reward_store)
        plt.title("Reward per Episode")
        plt.xlabel("Episode"); plt.ylabel("Reward")
        plt.subplot(1, 3, 2)
        plt.plot(self._avg_queue_length_store)
        plt.title("Avg Queue Length per Episode")
        plt.xlabel("Episode"); plt.ylabel("Avg Queue Length")
        plt.subplot(1, 3, 3)
        plt.plot(self._cumulative_wait_store)
        plt.title("Cumulative Wait Time per Episode")
        plt.xlabel("Episode"); plt.ylabel("Cumulative Wait Time")
        plt.show()

        print("Final Faulty Lights:", self.faulty_lights)
        if self._green_durations_log:
            plt.figure(figsize=(10, 4))
            plt.plot(self._green_durations_log)
            plt.title("Adaptive Green Duration Over Time")
            plt.xlabel("Step"); plt.ylabel("Green Duration")
            plt.grid(True); plt.show()

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

