import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated):
        self._n_cars_generated = n_cars_generated
        self._max_steps = max_steps

        # Your provided route IDs
        self.route_ids = [
            "W_N1", "W_N2", "W_E", "W_S1", "W_S2",
            "N_W1", "N_W2", "N_E1", "N_E2", "N_S1", "N_S2",
            "E_W", "E_N1", "E_N2", "E_S1", "E_S2",
            "S_N1", "S_N2", "S_E1", "S_W1", "S_W2"
        ]

    def generate_routefile(self, seed):
        np.random.seed(seed)
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # Rescale timings to [0, max_steps]
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        for value in timings:
            scaled_value = ((self._max_steps) / (max_old - min_old)) * (value - max_old) + self._max_steps
            car_gen_steps.append(int(round(scaled_value)))

        vehicle_entries = []

        # Standard cars
        for car_counter, step in enumerate(car_gen_steps):
            route = np.random.choice(self.route_ids)
            entry = f'    <vehicle id="veh_{car_counter}" type="standard_car" route="{route}" depart="{step}" departLane="random" departSpeed="10" />'
            vehicle_entries.append((step, entry))

        # Emergency cars
        num_emergency = 3
        emergency_times = np.rint(np.sort(np.random.uniform(0, self._max_steps, num_emergency))).astype(int)
        for i, t in enumerate(emergency_times):
            route = np.random.choice(self.route_ids)
            entry = f'    <vehicle id="emergency_{i}" type="emergency" route="{route}" depart="{t}" departLane="random" departSpeed="10" />'
            vehicle_entries.append((t, entry))

        # Sort all vehicles by depart time
        vehicle_entries.sort(key=lambda x: x[0])

        # Write route file
        with open("intersection/episode2_routes.rou.xml", "w") as routes:
            print("""<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />""", file=routes)

            for rid in self.route_ids:
                edge_map = {
                    "W_N1": "E13 -E11", "W_N2": "E13 E14 -E16",
                    "W_E": "E13 E14 E15", "W_S1": "E13 E12", "W_S2": "E13 E14 E17",
                    "N_W1": "E11 -E13", "N_W2": "E16 -E14 -E13",
                    "N_E1": "E11 E14 E15", "N_E2": "E16 E15",
                    "N_S1": "-E12 -E11", "N_S2": "-E17 -E16",
                    "E_W": "-E15 -E14 -E13", "E_N1": "-E15 -E14 -E11", "E_N2": "-E15 -E16",
                    "E_S1": "-E15 E17", "E_S2": "-E15 -E14 E12",
                    "S_N1": "-E12 -E11", "S_N2": "-E17 -E16",
                    "S_E1": "-E17 E15", "S_W1": "-E12 -E13", "S_W2": "-E17 -E14 -E13"
                }
                print(f'    <route id="{rid}" edges="{edge_map[rid]}" />', file=routes)

            for _, entry in vehicle_entries:
                print(entry, file=routes)
            print("</routes>", file=routes)
