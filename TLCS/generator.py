import numpy as np
import math
import os
import intersection_config as int_config

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated, intersection_type="cross"):
        self._n_cars_generated = n_cars_generated  # number of standard cars per episode
        self._max_steps = max_steps
        self.intersection_type = intersection_type
        # Load the intersection configuration for this type.
        if self.intersection_type in int_config.INTERSECTION_CONFIGS:
            self.int_conf = int_config.INTERSECTION_CONFIGS[self.intersection_type]
        else:
            raise ValueError("Intersection type '{}' not found in configuration.".format(self.intersection_type))

    def generate_routefile(self, seed):
        """
        Generate the route file for one episode.
        Standard vehicles follow a Weibull distribution for their depart times.
        Additionally, three emergency vehicles with random departure times are added.
        The final file is sorted by departure time.
        """
        np.random.seed(seed)  # for reproducibility

        # Generate departure timings using a Weibull distribution.
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # Rescale timings to fit in the interval [0, max_steps].
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        for value in timings:
            scaled_value = ((self._max_steps) / (max_old - min_old)) * (value - max_old) + self._max_steps
            car_gen_steps = np.append(car_gen_steps, scaled_value)
        car_gen_steps = np.rint(car_gen_steps).astype(int)

        # Modularized route generation based on intersection type.
        if self.intersection_type == "cross":
            vehicle_entries = self._generate_cross_routes(car_gen_steps)
        elif self.intersection_type == "roundabout":
            vehicle_entries = self._generate_roundabout_routes(car_gen_steps)
        elif self.intersection_type == "2x2_grid":
            vehicle_entries = self._generate_2x2_grid_routes(car_gen_steps)
        elif self.intersection_type == "t_intersection":
            vehicle_entries = self._generate_t_intersection_routes(car_gen_steps)
        elif self.intersection_type == "ow":
            vehicle_entries = self._generate_ow_routes(car_gen_steps)
        else:
            vehicle_entries = self._generate_default_routes(car_gen_steps)

        # Generate emergency vehicle entries.
        emergency_entries = self._generate_emergency_routes()
        vehicle_entries.extend(emergency_entries)

        # Sort all vehicle entries by departure time.
        vehicle_entries.sort(key=lambda x: x[0])

        # Build the output file path based on the intersection type.
        output_folder = os.path.join("intersection", self.intersection_type)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Decide on filename based on intersection type
        if self.intersection_type == "2x2_grid":
            output_file = os.path.join(output_folder, "2x2_grid.rou.xml")
        elif self.intersection_type == "roundabout":
            output_file = os.path.join(output_folder, "roundabout.rou.xml")
        elif self.intersection_type == "cross_intersection":
            output_file = os.path.join(output_folder, "cross_routes.rou.xml")
        elif self.intersection_type == "t_intersection":
            output_file = os.path.join(output_folder, "t_intersection.rou.xml")
        elif self.intersection_type == "ow":
            output_file = os.path.join(output_folder, "ow.rou.xml")
        else:
            output_file = os.path.join(output_folder, "default_routes.rou.xml")


        # Get header from configuration (if provided) or use default.
        header = self._get_header()

        # Write header and all vehicle entries to the route file.
        with open(output_file, "w") as routes:
            print(header, file=routes)
            for _, entry in vehicle_entries:
                print(entry, file=routes)
            print("</routes>", file=routes)

    def _generate_cross_routes(self, car_gen_steps):
        """
        Generate routes for a cross intersection.
        Expects int_conf to have a 'route_config' key with 'straight' and 'turn' definitions.
        Example in intersection_config.py:
            "cross": {
                "route_config": {
                    "straight": {"routes": ["W_E", "E_W", "N_S", "S_N"], "probability": 0.75},
                    "turn": {"routes": ["W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E"], "probability": 0.25}
                },
                "header": "...custom header..."
            }
        """
        vehicle_entries = []
        route_conf = self.int_conf.get("route_config", {})
        straight_conf = route_conf.get("straight", {"routes": ["W_E", "E_W", "N_S", "S_N"], "probability": 0.75})
        turn_conf = route_conf.get("turn", {"routes": ["W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E"], "probability": 0.25})
        straight_prob = straight_conf.get("probability", 0.75)

        for car_counter, step in enumerate(car_gen_steps):
            depart_time = step
            if np.random.uniform() < straight_prob:
                chosen_route = np.random.choice(straight_conf.get("routes", []))
            else:
                chosen_route = np.random.choice(turn_conf.get("routes", []))
            entry = '    <vehicle id="{}" type="standard_car" route="{}" depart="{}" departLane="random" departSpeed="10" />'.format(
                chosen_route + "_" + str(car_counter), chosen_route, depart_time)
            vehicle_entries.append((depart_time, entry))
        return vehicle_entries

    def _generate_roundabout_routes(self, car_gen_steps):
        """
        Generate routes for a roundabout intersection.
        Expects a route_config with keys like 'in_out' and 'turn'.

        Example:
            "route_config": {
                "in_out": {"routes": ["route1","route2","route3","route4"], "probability": 0.8},
                "turn":   {"routes": ["route5","route6","route7","route8"], "probability": 0.2}
            }
        """
        vehicle_entries = []
        route_conf = self.int_conf.get("route_config", {})

        # Default placeholders, but your intersection_config.py should override them
        in_out_conf = route_conf.get("in_out", {"routes": ["roundabout_in_out"], "probability": 0.8})
        turn_conf   = route_conf.get("turn",   {"routes": ["roundabout_turn"],   "probability": 0.2})
        in_out_prob = in_out_conf.get("probability", 0.8)

        for car_counter, step in enumerate(car_gen_steps):
            depart_time = step
            if np.random.uniform() < in_out_prob:
                chosen_route = np.random.choice(in_out_conf.get("routes", []))
            else:
                chosen_route = np.random.choice(turn_conf.get("routes", []))
            entry = '    <vehicle id="roundabout_{}" type="standard_car" route="{}" depart="{}" departLane="random" departSpeed="10" />'.format(
                car_counter, chosen_route, depart_time)
            vehicle_entries.append((depart_time, entry))
        return vehicle_entries

    def _generate_t_intersection_routes(self, car_gen_steps):
        """
        Generate routes for a t_intersection.
        Expects route_config with keys like 'main' and 'turn'.
        """
        vehicle_entries = []
        route_conf = self.int_conf.get("route_config", {})
        main_conf = route_conf.get("main", {"routes": ["left_in_top_out", "right_in_top_out"], "probability": 0.7})
        turn_conf = route_conf.get("turn", {"routes": ["top_in_left_out", "top_in_right_out"], "probability": 0.3})
        main_prob = main_conf.get("probability", 0.7)

        for car_counter, step in enumerate(car_gen_steps):
            depart_time = step
            if np.random.uniform() < main_prob:
                chosen_route = np.random.choice(main_conf.get("routes", []))
            else:
                chosen_route = np.random.choice(turn_conf.get("routes", []))
            entry = '    <vehicle id="{}" type="standard_car" route="{}" depart="{}" departLane="random" departSpeed="10" />'.format(
                chosen_route + "_" + str(car_counter), chosen_route, depart_time)
            vehicle_entries.append((depart_time, entry))
        return vehicle_entries
    
    def _generate_2x2_grid_routes(self, car_gen_steps):
        """
        Generate routes for a 2x2_grid.
        Expects a route_config with keys such as 'main' and 'side'.
        """
        vehicle_entries = []
        route_conf = self.int_conf.get("route_config", {})
        main_conf = route_conf.get("main", {"routes": ["W_E", "E_W"], "probability": 0.7})
        side_conf = route_conf.get("side", {"routes": ["N_E", "N_W"], "probability": 0.3})
        main_prob = main_conf.get("probability", 0.7)

        for car_counter, step in enumerate(car_gen_steps):
            depart_time = step
            if np.random.uniform() < main_prob:
                chosen_route = np.random.choice(main_conf.get("routes", []))
            else:
                chosen_route = np.random.choice(side_conf.get("routes", []))
            entry = '    <vehicle id="{}" type="standard_car" route="{}" depart="{}" departLane="random" departSpeed="10" />'.format(
                chosen_route + "_" + str(car_counter), chosen_route, depart_time)
            vehicle_entries.append((depart_time, entry))
        return vehicle_entries
    
    def _generate_ow_routes(self, car_gen_steps):
        vehicle_entries = []
        route_conf = self.int_conf.get("route_config", {})
        main_conf = route_conf.get("main", {"routes": ["b1a", "b1b", "e6a", "e6b", "c2a"], "probability": 1.0})
        route_ids = main_conf.get("routes", [])
        prob = main_conf.get("probability", 1.0)

        for car_counter, step in enumerate(car_gen_steps):
            depart_time = step
            chosen_route = np.random.choice(route_ids)
            entry = '    <vehicle id="ow_{}" type="standard_car" route="{}" depart="{}" departLane="random" departSpeed="10" />'.format(
                car_counter, chosen_route, depart_time)
            vehicle_entries.append((depart_time, entry))
        return vehicle_entries

    def _generate_default_routes(self, car_gen_steps):
        """
        Fallback route generation if the intersection type is not recognized.
        Uses the cross intersection logic.
        """
        return self._generate_cross_routes(car_gen_steps)

    def _generate_emergency_routes(self):
        """
        Generate emergency vehicle entries.
        Reads an optional 'emergency_routes' key from the intersection configuration.
        If not defined, defaults are used.
        """
        vehicle_entries = []
        num_emergency = 3
        emergency_departures = np.random.uniform(0, self._max_steps, num_emergency)
        emergency_departures = np.rint(np.sort(emergency_departures)).astype(int)

        # Attempt to read from config first
        routes_list = self.int_conf.get("emergency_routes", None)

        # If no config for emergency routes is found, we pick default sets
        if not routes_list:
            if self.intersection_type == "2x2_grid":
                # 2x2_grid default
                routes_list = ["W_E", "E_W", "N_E", "N_W", "E_N", "W_N"]
            elif self.intersection_type == "roundabout":
                # Roundabout default → all route IDs
                routes_list = ["route1","route2","route3","route4","route5","route6","route7","route8"]
            else:
                # Cross or fallback
                routes_list = ["W_N", "W_E", "W_S", "N_W", "N_E", "N_S", "E_W", "E_N", "S_W", "S_N", "S_E"]

        for i, depart_time in enumerate(emergency_departures):
            chosen_route = np.random.choice(routes_list)
            entry = '    <vehicle id="emergency_{}" type="emergency" route="{}" depart="{}" departLane="random" departSpeed="10" />'.format(
                i, chosen_route, depart_time)
            vehicle_entries.append((depart_time, entry))
        return vehicle_entries

    def _get_header(self):
        """
        Retrieve a header from the intersection configuration if provided.
        Otherwise, return a default header for each intersection type.
        """
        header = self.int_conf.get("header", None)
        if header:
            return header
        else:
            if self.intersection_type == "cross":
                return """<routes>
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
                return """<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />

    <!-- Example of simple in/out routes, override in config or .rou.xml if needed -->
    <route id="roundabout_in_out" edges="in_edge out_edge"/>
    <route id="roundabout_turn" edges="in_edge turn_edge out_edge"/>"""
            elif self.intersection_type == "2x2_grid":
                return """<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true"/>
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5"/>

    <!-- Main road routes (west-east and east-west) -->
    <route id="W_E" edges="left_in right_out"/>
    <route id="E_W" edges="right_in left_out"/>

    <!-- Routes from the north approach joining the main road -->
    <route id="N_E" edges="top_in right_out"/>
    <route id="N_W" edges="top_in left_out"/>

    <!-- Optional: if vehicles on the main road can turn northward -->
    <route id="E_N" edges="right_in top_out"/>
    <route id="W_N" edges="left_in top_out"/>"""
            else:
                return """<routes>
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