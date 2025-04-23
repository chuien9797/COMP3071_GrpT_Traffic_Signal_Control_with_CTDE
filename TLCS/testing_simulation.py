import traci
import numpy as np
import timeit
import os

from intersection_config import INTERSECTION_CONFIGS

class TestingSimulation:
    def __init__(self, Models, TrafficGen, sumo_cmd, max_steps, green_duration,
                 yellow_duration, num_states, intersection_type, inject_faults=False):
        self._Models = Models
        self._TrafficGen = TrafficGen
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._intersection_type = intersection_type
        self._num_agents = len(Models)
        self.inject_faults = inject_faults
        self.signal_fault_prob = 0.1
        self.enable_emergency_handling = True

        self._tls_ids = INTERSECTION_CONFIGS[intersection_type]["traffic_light_ids"]

        # Metric stores
        self.reward_store = []
        self.cumulative_wait_store = []
        self.avg_queue_length_store = []
        self.emergency_clearance_time = 0
        self._emergency_entry_exit = {}  # vehicle_id → (entry, exit)
        self._num_faults_injected = 0
        self._num_faults_recovered = 0
        self.avg_vehicle_delay = 0
        self.total_vehicles_set = set()

    def _get_state(self, tls_id):
        lanes = traci.trafficlight.getControlledLanes(tls_id)
        state = []

        for lane_id in lanes:
            num_veh = traci.lane.getLastStepVehicleNumber(lane_id)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane_id)
            occupancy = traci.lane.getLastStepOccupancy(lane_id)
            state.append([0, 0, 0, 0, 0, 0, num_veh, mean_speed, occupancy])

        while len(state) < 9:
            state.append([0] * 9)

        return np.array(state[:9], dtype=np.float32)

    def _set_phase(self, tls_id, action_idx):
        try:
            # First try integer key (preferred)
            phase_dict = INTERSECTION_CONFIGS[self._intersection_type]["phase_mapping"][action_idx]
        except KeyError:
            # Fall back to string key if needed
            phase_dict = INTERSECTION_CONFIGS[self._intersection_type]["phase_mapping"][str(action_idx)]
        green_phase = phase_dict["green"]
        traci.trafficlight.setPhase(tls_id, green_phase)

    def run(self, scenario_name="default", env_index=0):
        print(f"Starting SUMO for scenario '{scenario_name}'...")
        seed = 1000 + env_index * 10
        self._TrafficGen.generate_routefile(seed=seed)

        traci.start(self._sumo_cmd)

        step = 0
        queue_lengths = []
        total_wait_time = []

        action_duration = self._green_duration
        tls_ids = self._tls_ids

        start_time = timeit.default_timer()

        while step < self._max_steps:
            for i, tls_id in enumerate(tls_ids):
                state = self._get_state(tls_id)
                state = np.expand_dims(state, axis=0)

                # Debug: print state shape and content
                print(f"[{tls_id}] State shape: {state.shape}")
                print(f"[{tls_id}] State sample:\n{state[0][:3]}")  # first 3 lanes

                # Get the model for this traffic light agent
                model = self._Models[i]

                # Get action probabilities
                action_probs = model.predict_one(state).flatten()
                print(f"[{tls_id}] Q-values: {action_probs}")
                print(f"[{tls_id}] Chosen Action → {np.argmax(action_probs)}")


                # Pick the action
                action = int(np.argmax(action_probs))
                print(f"[{tls_id}] MODEL Action → {action}")


                if self.inject_faults and np.random.rand() < self.signal_fault_prob:
                    self._num_faults_injected += 1
                    print(f"[FAULT] Injected at {tls_id} on step {step}")
                    red_phase = INTERSECTION_CONFIGS[self._intersection_type].get("fault_phase", 0)
                    traci.trafficlight.setPhase(tls_id, red_phase)
                else:
                    self._set_phase(tls_id, action)

                if getattr(self, "enable_emergency_handling", False):
                    try:
                        self._handle_emergency_vehicle(tls_id)
                    except AttributeError:
                        pass

            for _ in range(action_duration):
                traci.simulationStep()
                step += 1

                total_wait = 0
                total_queued = 0

                for tls_id in tls_ids:
                    lanes = traci.trafficlight.getControlledLanes(tls_id)
                    for lane in lanes:
                        total_wait += traci.lane.getWaitingTime(lane)
                        total_queued += traci.lane.getLastStepVehicleNumber(lane)

                total_wait_time.append(total_wait)
                queue_lengths.append(total_queued)

                for vid in traci.vehicle.getIDList():
                    self.total_vehicles_set.add(vid)

                    if vid.startswith("emergency_"):
                        if vid not in self._emergency_entry_exit:
                            self._emergency_entry_exit[vid] = [step, None]
                        elif traci.vehicle.getRoadID(vid) == "":
                            self._emergency_entry_exit[vid][1] = step

        sim_time = round(timeit.default_timer() - start_time, 2)
        traci.close()

        total_vehicles = len(self.total_vehicles_set)
        self.avg_vehicle_delay = np.sum(total_wait_time) / total_vehicles if total_vehicles > 0 else 0

        for entry_exit in self._emergency_entry_exit.values():
            entry, exit = entry_exit
            if exit is not None:
                self.emergency_clearance_time += (exit - entry)

        self.reward_store.append(-np.mean(total_wait_time))
        self.cumulative_wait_store.append(np.sum(total_wait_time))
        self.avg_queue_length_store.append(np.mean(queue_lengths))

        print("\n--- TESTING METRICS ---")
        print(f"Total Sim Time: {sim_time:.2f}s")
        print(f"Avg Vehicle Delay: {self.avg_vehicle_delay:.2f}s")
        print(f"Avg Queue Length: {np.mean(queue_lengths):.2f}")
        print(f"Total Reward: {-np.mean(total_wait_time):.2f}")
        print(f"Emergency Clearance Time: {self.emergency_clearance_time:.2f}s")
        print(f"Faults Injected: {self._num_faults_injected}")
        print(f"Faults Recovered (tracked manually or TBD): {self._num_faults_recovered}")

        # ✅ LOG TO FILE
        os.makedirs("logs_test", exist_ok=True)
        with open("logs/testing_metrics.txt", "a") as f:
            f.write(f"\n--- Scenario: {scenario_name}, Env Index: {env_index} ---\n")
            f.write(f"Sim Time: {sim_time:.2f}s\n")
            f.write(f"Avg Vehicle Delay: {self.avg_vehicle_delay:.2f}s\n")
            f.write(f"Avg Queue Length: {np.mean(queue_lengths):.2f}\n")
            f.write(f"Total Reward: {-np.mean(total_wait_time):.2f}\n")
            f.write(f"Emergency Clearance Time: {self.emergency_clearance_time:.2f}s\n")
            f.write(f"Faults Injected: {self._num_faults_injected}\n")
            f.write(f"Faults Recovered: {self._num_faults_recovered}\n")
            f.write("=====================================\n")

        return sim_time