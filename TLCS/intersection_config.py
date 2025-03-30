"""
Intersection configuration for various intersection types.

This file defines different intersection configurations for your RL traffic agent.
For each intersection type, we specify:
  - incoming_lanes: a dictionary mapping lane groups (e.g., "N", "S", "E", "W" or other group names) to lists of lane IDs.
  - phase_mapping: a dictionary mapping each abstract action (0, 1, etc.) to a dictionary with keys "green" and "yellow",
                   which specify the corresponding SUMO phase numbers for the green and yellow phases.
  - occupancy_grid: parameters used to discretize lane positions into grid cells (e.g., number of cells per lane and maximum distance).
  - route_config: (new) a dictionary that defines available routes and turning probabilities.
  - header: (optional) a custom header for the routes file.
  - emergency_routes: (optional) a list of routes to be used for emergency vehicles.
"""

INTERSECTION_CONFIGS = {
    "cross": {
        "incoming_lanes": {
            "N": ["N2TL_0", "N2TL_1", "N2TL_2", "N2TL_3"],
            "S": ["S2TL_0", "S2TL_1", "S2TL_2", "S2TL_3"],
            "E": ["E2TL_0", "E2TL_1", "E2TL_2", "E2TL_3"],
            "W": ["W2TL_0", "W2TL_1", "W2TL_2", "W2TL_3"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1},  # NS green
            1: {"green": 2, "yellow": 3},  # NS left-turn green
            2: {"green": 4, "yellow": 5},  # EW green
            3: {"green": 6, "yellow": 7}   # EW left-turn green
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
        # "emergency_routes" can be added here if needed.
    },
    "roundabout": {
        "incoming_lanes": {
            "in": ["roundabout_in_0", "roundabout_in_1"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1}
        },
        "occupancy_grid": {
            "cells_per_lane": 8,
            "max_distance": 500
        },
        "route_config": {
            "in_out": {
                "routes": ["roundabout_in_out"],
                "probability": 0.8
            },
            "turn": {
                "routes": ["roundabout_turn"],
                "probability": 0.2
            }
        },
        "header": """<routes>
    <!-- Define emergency vehicle type -->
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
    
    <route id="roundabout_in_out" edges="in_edge out_edge"/>
    <route id="roundabout_turn" edges="in_edge turn_edge out_edge"/>"""
    },
    "T_intersection": {
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
    },
    "Y_intersection": {
        "incoming_lanes": {
            "branch1": ["Y_branch1_0", "Y_branch1_1"],
            "branch2": ["Y_branch2_0", "Y_branch2_1"],
            "branch3": ["Y_branch3_0", "Y_branch3_1"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1},  # Branch 1
            1: {"green": 2, "yellow": 3},  # Branch 2
            2: {"green": 4, "yellow": 5}   # Branch 3
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
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true" />
    <!-- Define standard vehicle type -->
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5" />
    
    <route id="Y_branch1" edges="edge1 edge2"/>
    <route id="Y_branch2" edges="edge3 edge4"/>
    <route id="Y_branch3" edges="edge5 edge6"/>"""
    }
}
