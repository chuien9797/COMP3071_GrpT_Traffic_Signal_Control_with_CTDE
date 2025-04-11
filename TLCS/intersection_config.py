"""
Intersection configuration for various intersection types (cross, roundabout,
T_intersection, Y_intersection).

We include:
  - sumocfg_file: path to the SUMO .sumocfg for each intersection
  - incoming_lanes: dict of lane groups to their lane IDs
  - phase_mapping: how we map an RL "action" to (green, yellow) phases
  - occupancy_grid: (cells_per_lane, max_distance) for occupancy
  - route_config: how we define route probabilities for route generation
  - header: optional custom <routes> XML snippet
  - monitor_edges: which edges we want to call traci.edge.getLastStepHaltingNumber on
  - monitor_lanes: which lanes we want to call traci.lane.getWaitingTime on
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
        "sumocfg_file": "t_intersection/t_intersection.sumocfg",
        "traffic_light_ids": ["TL"],
        "incoming_lanes": {
            "main": ["left_in_0", "left_in_1", "right_in_0", "right_in_1"],
            "side": ["top_in_0", "top_in_1"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1},   # main road green
            1: {"green": 2, "yellow": 3},   # side road green
            2: {"green": 0, "yellow": 1},   # duplicate main road green
            3: {"green": 2, "yellow": 3}    # duplicate side road green
        },
        "occupancy_grid": {
            "cells_per_lane": 8,
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
        "header": """<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0"
           maxSpeed="20" sigma="0.5" emergency="true"/>
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0"
           minGap="2.5" maxSpeed="25" sigma="0.5"/>

    <!-- Main road routes (west-east and east-west) -->
    <route id="W_E" edges="left_in right_out"/>
    <route id="E_W" edges="right_in left_out"/>

    <!-- Routes from the north approach joining the main road -->
    <route id="N_E" edges="top_in right_out"/>
    <route id="N_W" edges="top_in left_out"/>

    <!-- Optional: if vehicles on the main road can turn northward -->
    <route id="E_N" edges="right_in top_out"/>
    <route id="W_N" edges="left_in top_out"/>""",
        # For T_intersection, let's monitor the three inbound edges
        # named left_in, right_in, top_in. If your net uses different names, adjust them.
        "monitor_edges": ["left_in", "right_in", "top_in"],
        # We'll track waiting time on lane 0 for each approach
        "monitor_lanes": ["left_in_0", "right_in_0", "top_in_0"]
    },

    "Y_intersection": {
        # If you have a Y intersection net + sumocfg:
        "sumocfg_file": "y_intersection/y_intersection.sumocfg",
        "incoming_lanes": {
            "branch1": ["Y_branch1_0", "Y_branch1_1"],
            "branch2": ["Y_branch2_0", "Y_branch2_1"],
            "branch3": ["Y_branch3_0", "Y_branch3_1"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1},  # branch1
            1: {"green": 2, "yellow": 3},  # branch2
            2: {"green": 4, "yellow": 5}   # branch3
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
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0"
           maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0"
           minGap="2.5" maxSpeed="25" sigma="0.5" />

    <route id="Y_branch1" edges="edge1 edge2"/>
    <route id="Y_branch2" edges="edge3 edge4"/>
    <route id="Y_branch3" edges="edge5 edge6"/>""",
        # For a Y, you might define 3 inbound edges: e.g. "Y_branch1","Y_branch2","Y_branch3"
        "monitor_edges": ["Y_branch1", "Y_branch2", "Y_branch3"],
        # Only lane 0 for each approach
        "monitor_lanes": ["Y_branch1_0", "Y_branch2_0", "Y_branch3_0"]
    }
}
