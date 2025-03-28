import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated, intersection_type="cross"):
        self._n_cars_generated = n_cars_generated  # number of standard cars per episode
        self._max_steps = max_steps
        self.intersection_type = intersection_type

    def generate_routefile(self, seed):
        """
        Generation of the route file for one episode.
        Standard vehicles follow a Weibull distribution for their depart times.
        Additionally, three emergency vehicles with random departure times are added.
        The final file is sorted by departure time.
        """
        np.random.seed(seed)  # for reproducibility

        # Generate departure timings for standard vehicles using a Weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # Rescale timings to fit in the interval [0, max_steps]
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        for value in timings:
            # Scale value to [0, max_steps]
            scaled_value = ((self._max_steps) / (max_old - min_old)) * (value - max_old) + self._max_steps
            car_gen_steps = np.append(car_gen_steps, scaled_value)
        car_gen_steps = np.rint(car_gen_steps).astype(int)  # round each value to int

        # List to store all vehicle entries as tuples: (depart_time, entry_string)
        vehicle_entries = []

        # Generate standard vehicle entries based on intersection type
        if self.intersection_type == "cross":
            for car_counter, step in enumerate(car_gen_steps):
                depart_time = step
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:  # 75% chance for going straight
                    route_straight = np.random.randint(1, 5)  # choose a random route among four options
                    if route_straight == 1:
                        entry = '    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_straight == 2:
                        entry = '    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_straight == 3:
                        entry = '    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    else:
                        entry = '    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                else:  # 25% chance for turning vehicles
                    route_turn = np.random.randint(1, 9)  # choose among eight possible turning routes
                    if route_turn == 1:
                        entry = '    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 2:
                        entry = '    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 3:
                        entry = '    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 4:
                        entry = '    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 5:
                        entry = '    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 6:
                        entry = '    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 7:
                        entry = '    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 8:
                        entry = '    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                vehicle_entries.append((depart_time, entry))
        elif self.intersection_type == "roundabout":
            # For roundabouts, assume different route definitions
            for car_counter, step in enumerate(car_gen_steps):
                depart_time = step
                # 80% chance: vehicles follow a straightforward roundabout route
                if np.random.uniform() < 0.8:
                    entry = '    <vehicle id="roundabout_%i" type="standard_car" route="roundabout_in_out" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                else:
                    # 20% chance: vehicles perform a turning maneuver inside the roundabout
                    entry = '    <vehicle id="roundabout_turn_%i" type="standard_car" route="roundabout_turn" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                vehicle_entries.append((depart_time, entry))
        else:
            # Default behavior: fallback to cross intersection routes
            for car_counter, step in enumerate(car_gen_steps):
                depart_time = step
                straight_or_turn = np.random.uniform()
                if straight_or_turn < 0.75:
                    route_straight = np.random.randint(1, 5)
                    if route_straight == 1:
                        entry = '    <vehicle id="W_E_%i" type="standard_car" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_straight == 2:
                        entry = '    <vehicle id="E_W_%i" type="standard_car" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_straight == 3:
                        entry = '    <vehicle id="N_S_%i" type="standard_car" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    else:
                        entry = '    <vehicle id="S_N_%i" type="standard_car" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                else:
                    route_turn = np.random.randint(1, 9)
                    if route_turn == 1:
                        entry = '    <vehicle id="W_N_%i" type="standard_car" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 2:
                        entry = '    <vehicle id="W_S_%i" type="standard_car" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 3:
                        entry = '    <vehicle id="N_W_%i" type="standard_car" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 4:
                        entry = '    <vehicle id="N_E_%i" type="standard_car" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 5:
                        entry = '    <vehicle id="E_N_%i" type="standard_car" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 6:
                        entry = '    <vehicle id="E_S_%i" type="standard_car" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 7:
                        entry = '    <vehicle id="S_W_%i" type="standard_car" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                    elif route_turn == 8:
                        entry = '    <vehicle id="S_E_%i" type="standard_car" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (car_counter, depart_time)
                vehicle_entries.append((depart_time, entry))

        # Generate emergency vehicle entries with random departure times
        num_emergency = 3
        emergency_departures = np.random.uniform(0, self._max_steps, num_emergency)
        emergency_departures = np.rint(np.sort(emergency_departures)).astype(int)
        # Define a list of routes for emergency vehicles (could be customized further per intersection type)
        routes_list = ["W_N", "W_E", "W_S", "N_W", "N_E", "N_S", "E_W", "E_N", "E_S", "S_W", "S_N", "S_E"]
        for i, depart_time in enumerate(emergency_departures):
            chosen_route = np.random.choice(routes_list)
            entry = '    <vehicle id="emergency_%i" type="emergency" route="%s" depart="%s" departLane="random" departSpeed="10" />' % (i, chosen_route, depart_time)
            vehicle_entries.append((depart_time, entry))

        # Sort all vehicle entries by departure time (first element of tuple)
        vehicle_entries.sort(key=lambda x: x[0])

        # Write the header and all sorted vehicle entries to the route file
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            if self.intersection_type == "cross":
                header = """<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

    <route id="W_N" edges="W2TL TL2N"/>
    <route id="W_E" edges="W2TL TL2E"/>
    <route id="W_S" edges="W2TL TL2S"/>
    <route id="N_W" edges="N2TL TL2W"/>
    <route id="N_E" edges="N2TL TL2E"/>
    <route id="N_S" edges="N2TL TL2S"/>
    <route id="E_W" edges="E2TL TL2W"/>
    <route id="E_N" edges="E2TL TL2N"/>
    <route id="E_S" edges="E2TL TL2S"/>
    <route id="S_W" edges="S2TL TL2W"/>
    <route id="S_N" edges="S2TL TL2N"/>
    <route id="S_E" edges="S2TL TL2E"/>"""
            elif self.intersection_type == "roundabout":
                header = """<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

    <route id="roundabout_in_out" edges="in_edge out_edge"/>
    <route id="roundabout_turn" edges="in_edge turn_edge out_edge"/>"""
            else:
                header = """<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

    <route id="W_N" edges="W2TL TL2N"/>
    <route id="W_E" edges="W2TL TL2E"/>
    <route id="W_S" edges="W2TL TL2S"/>
    <route id="N_W" edges="N2TL TL2W"/>
    <route id="N_E" edges="N2TL TL2E"/>
    <route id="N_S" edges="N2TL TL2S"/>
    <route id="E_W" edges="E2TL TL2W"/>
    <route id="E_N" edges="E2TL TL2N"/>
    <route id="E_S" edges="E2TL TL2S"/>
    <route id="S_W" edges="S2TL TL2W"/>
    <route id="S_N" edges="S2TL TL2N"/>
    <route id="S_E" edges="S2TL TL2E"/>"""
            print(header, file=routes)
            for _, entry in vehicle_entries:
                print(entry, file=routes)
            print("</routes>", file=routes)
