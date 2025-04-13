"""
Intersection configuration for various intersection types (cross, roundabout,
T_intersection, Y_intersection).

We include:
  - sumocfg_file: path to the SUMO .sumocfg file for the network associated with that intersection.
  - traffic_light_ids: a list of traffic light IDs controlling the intersection.
       For multi-agent setups (e.g. when one intersection is split into two control groups),
       two IDs should be provided.
  - incoming_lanes: dictionary mapping logical lane groups to the lane IDs as defined in your net file.
  - phase_mapping: a mapping from agent actions (the RL “action”) to a dictionary with keys
       "green" and "yellow" for the corresponding phase indices.
  - occupancy_grid: parameters for the cell size and maximum distance for occupancy measurements.
  - route_config: definitions for route choices (including probabilities) for generating routes.
  - header: XML snippet for route definitions (used to generate the SUMO routes file).
  - monitor_edges: a list of edge IDs for measuring halted vehicles (queue length).
  - monitor_lanes: a list of lane IDs for measuring waiting times.
"""

INTERSECTION_CONFIGS = {
    "cross": {
        "sumocfg_file": "cross_intersection/cross_intersection.sumocfg",
        "traffic_light_ids": ["TL"],
        "incoming_lanes": {
            "N": ["N2TL_0", "N2TL_1", "N2TL_2", "N2TL_3"],
            "S": ["S2TL_0", "S2TL_1", "S2TL_2", "S2TL_3"],
            "E": ["E2TL_0", "E2TL_1", "E2TL_2", "E2TL_3"],
            "W": ["W2TL_0", "W2TL_1", "W2TL_2", "W2TL_3"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1},  # NS straight
            1: {"green": 2, "yellow": 3},  # NS left-turn
            2: {"green": 4, "yellow": 5},  # EW straight
            3: {"green": 6, "yellow": 7}   # EW left-turn
        },
        "occupancy_grid": {
            "cells_per_lane": 10,
            "max_distance": 750
        },
        "route_config": {
            "straight": {
                "routes": ["W_E", "E_W", "N_S", "S_N"],
                "probability": 0.75
            },
            "turn": {
                "routes": ["W_N", "W_S", "N_W", "N_E", "E_N", "E_S", "S_W", "S_E"],
                "probability": 0.25
            }
        },
        "communication_mode": True,
        "header": """<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0"
           maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0"
           minGap="2.5" maxSpeed="25" sigma="0.5" />

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
    <route id="S_E" edges="S2TL TL2E"/>""",
        # Monitor these edges for halting:
        "monitor_edges": ["N2TL", "S2TL", "E2TL", "W2TL"],
        # Monitor these lanes for waiting times:
        "monitor_lanes": ["N2TL_0", "S2TL_0", "E2TL_0", "W2TL_0"]
    },

    "roundabout": {
        "sumocfg_file": "roundabout/roundabout.sumocfg",
        "traffic_light_ids": ["TL1", "TL2", "TL3", "TL4"],
        "incoming_lanes": {
            "e1": ["e1_0", "e1_1"],
            "e2": ["e2_0", "e2_1"],
            "e3": ["e3_0", "e3_1"],
            "e4": ["e4_0", "e4_1"]
        },
        "phase_mapping": {
           0: {"green": 0, "yellow": 1},
           1: {"green": 1, "yellow": 2},
           2: {"green": 2, "yellow": 3},
           3: {"green": 3, "yellow": 0}# You must ensure that these indices are valid given the TL logic

        },
        "occupancy_grid": {
            "cells_per_lane": 8,
            "max_distance": 500
        },
        "route_config": {
            "in_out": {
                "routes": ["route1", "route2", "route3", "route4"],
                "probability": 0.8
            },
            "turn": {
                "routes": ["route5", "route6", "route7", "route8"],
                "probability": 0.2
            }
        },
        "communication_mode": True,
        "header": """<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0"
           maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0"
           minGap="2.5" maxSpeed="25" sigma="0.5" />

    <route id="route1" edges="e1 r1 e2_out"/>
    <route id="route2" edges="e2 r2 e3_out"/>
    <route id="route3" edges="e3 r3 e4_out"/>
    <route id="route4" edges="e4 r4 e1_out"/>
    <route id="route5" edges="e1 r1 r2 e3_out"/>
    <route id="route6" edges="e2 r2 r3 e4_out"/>
    <route id="route7" edges="e3 r3 r4 e1_out"/>
    <route id="route8" edges="e4 r4 r1 e2_out"/>""",
        # Monitor the inbound edges for halting:
        "monitor_edges": ["e1", "e2", "e3", "e4"],
        # Only track lane 0 for waiting times, or add e1_1,... if you prefer
        "monitor_lanes": ["e1_0", "e2_0", "e3_0", "e4_0"]
    },

    "T_intersection": {
        "sumocfg_file": "2x2_grid/2x2_grid.sumocfg",  # Adjust the file path to match your SUMO configuration for the grid network
        "traffic_light_ids": ["1", "2"],  # Explicitly supply two IDs from your net file (the tlLogic IDs "1" and "2")
        "incoming_lanes": {
            # These lane IDs are derived from the junction (e.g., junction id "1" in the 2x2 grid)
            # Here we group them into "vertical" and "horizontal" approaches.
            "vertical": ["-v11_0", "-v11_1", "v12_0", "v12_1"],
            "horizontal": ["h12_0", "h12_1", "-h11_0", "-h11_1"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1},
            1: {"green": 2, "yellow": 3},
            2: {"green": 0, "yellow": 1},
            3: {"green": 2, "yellow": 3}
        },
        "occupancy_grid": {
            "cells_per_lane": 10,
            "max_distance": 600
        },
        "route_config": {
            "main": {
                "routes": ["W_E", "E_W"],
                "probability": 0.7
            },
            "side": {
                "routes": ["N_E", "N_W"],
                "probability": 0.3
            }
        },
        "communication_mode": True,
        "header": """<routes>
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true" />
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
    <!-- Define routes based on the new lane naming -->
    <route id="main_W_E" edges="h12_0 h12_1"/>
    <route id="main_E_W" edges="-h11_0 -h11_1"/>
    <route id="side_N_E" edges="-v11_0 -v11_1"/>
    <route id="side_N_W" edges="v12_0 v12_1"/>
</routes>""",
        # For monitoring, we split the edges into two halves.
        "monitor_edges": ["-v11", "-h11", "v12", "h12"],
        "monitor_lanes": ["-v11_0", "-v11_1", "-h11_0", "-h11_1", "v12_0", "v12_1", "h12_0", "h12_1"]
    },

    "Y_intersection": {
        "sumocfg_file": "y_intersection/y_intersection.sumocfg",
        "traffic_light_ids": ["YTL1", "YTL2"],
        "incoming_lanes": {
            "branch1": ["Y_branch1_0", "Y_branch1_1"],
            "branch2": ["Y_branch2_0", "Y_branch2_1"],
            "branch3": ["Y_branch3_0", "Y_branch3_1"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1},
            1: {"green": 2, "yellow": 3},
            2: {"green": 4, "yellow": 5}
        },
        "occupancy_grid": {
            "cells_per_lane": 8,
            "max_distance": 600
        },
        "route_config": {
            "branch1": {
                "routes": ["Y_branch1"],
                "probability": 0.33
            },
            "branch2": {
                "routes": ["Y_branch2"],
                "probability": 0.33
            },
            "branch3": {
                "routes": ["Y_branch3"],
                "probability": 0.34
            }
        },
        "header": """<routes>
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true" />
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
    <route id="Y_branch1" edges="edge1 edge2"/>
    <route id="Y_branch2" edges="edge3 edge4"/>
    <route id="Y_branch3" edges="edge5 edge6"/>
</routes>""",
        "monitor_edges": ["Y_branch1", "Y_branch2", "Y_branch3"],
        "monitor_lanes": ["Y_branch1_0", "Y_branch2_0", "Y_branch3_0"]
    }
}

if __name__ == "__main__":
    print("Intersection configurations loaded.")
