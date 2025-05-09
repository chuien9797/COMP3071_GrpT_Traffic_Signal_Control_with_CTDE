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

    "2x2_grid": {
    "sumocfg_file": "2x2_grid/2x2_grid.sumocfg",  # Adjust to your SUMO config file path
    "traffic_light_ids": ["1", "2", "5", "6"],  # Now four TL IDs for four intersections
    "incoming_lanes": {
        # For junction 1 (located, for example, at (300,600)) as in your net file:
        "junction1": ["-v11_0", "-v11_1", "h12_0", "h12_1", "-h11_0", "-h11_1"],
        # For junction 2 (located at (450,600)):
        "junction2": ["-v21_0", "-v21_1", "h13_0", "h13_1", "v22_0", "v22_1", "-h12_0", "-h12_1"],
        # For junction 3 (located at (300,450)):
        "junction3": ["-v12_0", "-v12_1", "h22_0", "h22_1", "v13_0", "v13_1", "-h21_0", "-h21_1"],
        # For junction 4 (located at (450,450)):
        "junction4": ["-v22_0", "-v22_1", "h23_0", "h23_1", "v23_0", "v23_1", "-h22_0", "-h22_1"]
    },
    "phase_mapping": {
        # Assuming each traffic light uses a similar 8-phase cycle,
        # you can use a uniform mapping if appropriate.
        0: {"green": 0, "yellow": 1},
        1: {"green": 2, "yellow": 3},
        2: {"green": 4, "yellow": 5},
        3: {"green": 6, "yellow": 7}
    },
    "occupancy_grid": {
        "cells_per_lane": 10,
        "max_distance": 600
    },
    "route_config": {
        # You may need to update or split your routes for a grid network.
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
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" 
           maxSpeed="20" sigma="0.5" emergency="true" />
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" 
           minGap="2.5" maxSpeed="25" sigma="0.5" />
    <!-- Define routes appropriate for your grid; update as needed -->
    <route id="W_E" edges="..."/>
    <route id="E_W" edges="..."/>
    <route id="N_E" edges="..."/>
    <route id="N_W" edges="..."/>
""",
    # You might want to monitor each intersection separately.
    "monitor_edges": ["-v11", "-h11", "h12", "-v21", "h13", "v22", "-v12", "-h21", "h22", "-v22", "h23", "v23"],
    "monitor_lanes": [
        "-v11_0", "-v11_1", "-h11_0", "-h11_1",
        "h12_0", "h12_1",
        "-v21_0", "-v21_1", "h13_0", "h13_1", "v22_0", "v22_1", "-h12_0", "-h12_1",
        "-v12_0", "-v12_1", "-h21_0", "-h21_1", "h22_0", "h22_1", "v13_0", "v13_1",
        "-v22_0", "-v22_1", "h23_0", "h23_1", "v23_0", "v23_1", "-h22_0", "-h22_1"
    ]
    },

    "1x2_grid": {
        "sumocfg_file": "1x2_grid/1x2_grid.sumocfg",
        "traffic_light_ids": ["TL", "2_TL"],
        "incoming_lanes": {
            "TL": [
                "N2TL_0", "N2TL_1", "N2TL_2", "N2TL_3",
                "S2TL_0", "S2TL_1", "S2TL_2", "S2TL_3",
                "W2TL_0", "W2TL_1", "W2TL_2", "W2TL_3",
                "2_TL2W_0", "2_TL2W_1", "2_TL2W_2", "2_TL2W_3"
            ],
            "2_TL": [
                "2_N2TL_0", "2_N2TL_1", "2_N2TL_2", "2_N2TL_3",
                "2_S2TL_0", "2_S2TL_1", "2_S2TL_2", "2_S2TL_3",
                "2_E2TL_0", "2_E2TL_1", "2_E2TL_2", "2_E2TL_3",
                "TL2E_0", "TL2E_1", "TL2E_2", "TL2E_3"
            ]
        },
        "phase_mapping": {
            "0": { "green": 0, "yellow": 1 },
            "1": { "green": 2, "yellow": 3 }
        },
        "occupancy_grid": {
            "cells_per_lane": 10,
            "max_distance": 750
        },
        "route_config": {
            "main": {
                "routes": ["E_W", "W_E", "N_S_1", "N_S_2", "S_N_1", "S_N_2"],
                "probability": 0.7
            },
            "side": {
                "routes": ["N_W", "S_E", "E_N", "W_S"],
                "probability": 0.3
            }
        },
        "communication_mode": True,
        "header": "<routes>\n\
            <vType id=\"emergency\" accel=\"3.0\" decel=\"6.0\" color=\"1,0,0\" maxSpeed=\"20\" sigma=\"0.5\" emergency=\"true\" />\n\
            <vType id=\"standard_car\" accel=\"1.0\" decel=\"4.5\" length=\"5.0\" minGap=\"2.5\" maxSpeed=\"25\" sigma=\"0.5\" />\n\
            <route id=\"E_W\" edges=\"2_E2TL 2_TL2W TL2W\" />\n\
            <route id=\"W_E\" edges=\"W2TL TL2E 2_TL2E\" />\n\
            <route id=\"N_S_1\" edges=\"2_N2TL 2_TL2S\" />\n\
            <route id=\"N_S_2\" edges=\"N2TL TL2S\" />\n\
            <route id=\"S_N_1\" edges=\"S2TL TL2N\" />\n\
            <route id=\"S_N_2\" edges=\"2_S2TL 2_TL2N\" />\n\
            <route id=\"N_W\" edges=\"2_N2TL 2_TL2W\" />\n\
            <route id=\"S_E\" edges=\"S2TL TL2E\" />\n\
            <route id=\"E_N\" edges=\"2_E2TL 2_TL2N\" />\n\
            <route id=\"W_S\" edges=\"W2TL TL2E 2_TL2S\" />",
        "monitor_edges": [
            "2_E2TL", "2_TL2E", "2_N2TL", "2_TL2N", "2_S2TL", "2_TL2S", "2_TL2W",
            "N2TL", "TL2N", "S2TL", "TL2S", "W2TL", "TL2W", "TL2E"
        ],
        "monitor_lanes": [
            "2_E2TL_0", "2_E2TL_1", "2_E2TL_2", "2_E2TL_3",
            "2_TL2E_0", "2_TL2E_1", "2_TL2E_2", "2_TL2E_3",
            "2_N2TL_0", "2_N2TL_1", "2_N2TL_2", "2_N2TL_3",
            "2_TL2N_0", "2_TL2N_1", "2_TL2N_2", "2_TL2N_3",
            "2_S2TL_0", "2_S2TL_1", "2_S2TL_2", "2_S2TL_3",
            "2_TL2S_0", "2_TL2S_1", "2_TL2S_2", "2_TL2S_3",
            "2_TL2W_0", "2_TL2W_1", "2_TL2W_2", "2_TL2W_3",
            "N2TL_0", "N2TL_1", "N2TL_2", "N2TL_3",
            "TL2N_0", "TL2N_1", "TL2N_2", "TL2N_3",
            "S2TL_0", "S2TL_1", "S2TL_2", "S2TL_3",
            "TL2S_0", "TL2S_1", "TL2S_2", "TL2S_3",
            "W2TL_0", "W2TL_1", "W2TL_2", "W2TL_3",
            "TL2W_0", "TL2W_1", "TL2W_2", "TL2W_3",
            "TL2E_0", "TL2E_1", "TL2E_2", "TL2E_3"
        ]
    },

"double_t": {
    "sumocfg_file": "double_t/double_t.sumocfg",
    "traffic_light_ids": ["N2", "N5"],

    "incoming_lanes": {
        "N2": [
            "left1_0", "left1_1",
            "top_0",
            "mid2_0", "mid2_1"
        ],
        "N5": [
            "mid1_0", "mid1_1",
            "right2_0", "right2_1",
            "bottom_0"
        ]
    },

    "phase_mapping": {
        "0": { "green": 0, "yellow": 1 },
        "1": { "green": 2, "yellow": 3 }
    },

    "occupancy_grid": {
        "cells_per_lane": 10,
        "max_distance": 750
    },

    "route_config": {
        "main": {
            "routes": ["left1_mid1", "right2_mid2"],
            "probability": 0.6
        },
        "branch": {
            "routes": ["top_mid1", "bottom_mid2", "left1_top_r", "right2_bottom_r"],
            "probability": 0.4
        }
    },

    "emergency_routes": [
        "left1_mid1", "right2_mid2",
        "top_mid1", "bottom_mid2",
        "left1_top_r", "right2_bottom_r"
    ],

    "header": """<routes>
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true"/>
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5"/>

    <route id="left1_mid1" edges="left1 mid1"/>
    <route id="right2_mid2" edges="right2 mid2"/>
    <route id="top_mid1" edges="top mid1"/>
    <route id="bottom_mid2" edges="bottom mid2"/>
    <route id="left1_top_r" edges="left1 top_r"/>
    <route id="right2_bottom_r" edges="right2 bottom_r"/>""",

    "monitor_edges": [
        "left1", "left2", "top", "top_r", "mid1", "mid2",
        "bottom", "bottom_r", "right1", "right2"
    ],

    "monitor_lanes": [
        "left1_0", "left1_1", "left2_0", "left2_1",
        "top_0", "top_r_0",
        "mid1_0", "mid1_1", "mid2_0", "mid2_1",
        "bottom_0", "bottom_r_0",
        "right1_0", "right1_1", "right2_0", "right2_1"
    ]
},

"t_with_u_turn": {
    "sumocfg_file": "t_with_u_turn/t_with_u_turn.sumocfg",
    "traffic_light_ids": ["center"],
    "incoming_lanes": {
        "center": [
            "left_to_center_0", "left_to_center_1",
            "right_to_center_0", "right_to_center_1",
            "down_to_center_0"
        ]
    },
    "phase_mapping": {
        "0": {"green": 0, "yellow": 1},
        "1": {"green": 2, "yellow": 3}
    },
    "occupancy_grid": {
        "cells_per_lane": 10,
        "max_distance": 150
    },
    "route_config": {
        "main": {
            "routes": [
                "left_to_right",
                "right_to_left",
                "down_to_left",
                "down_to_right"
            ],
            "probability": 0.7
        },
        "u_turn": {
            "routes": [
                "left_uturn",
                "right_uturn",
                "down_uturn"
            ],
            "probability": 0.3
        }
    },
    "emergency_routes": [
        "left_to_right", "right_to_left",
        "down_to_left", "down_to_right",
        "left_uturn", "right_uturn", "down_uturn"
    ],
    "header": """<routes>
    <vType id="emergency" accel="3.0" decel="6.0" color="1,0,0" maxSpeed="20" sigma="0.5" emergency="true"/>
    <vType id="standard_car" accel="1.0" decel="4.5" length="5.0" minGap="2.5" maxSpeed="25" sigma="0.5"/>
    
    <route id="left_to_right" edges="left_to_center center_to_right"/>
    <route id="right_to_left" edges="right_to_center center_to_left"/>
    <route id="down_to_left" edges="down_to_center center_to_left"/>
    <route id="down_to_right" edges="down_to_center center_to_right"/>
    
    <route id="left_uturn" edges="left_to_center center_to_left"/>
    <route id="right_uturn" edges="right_to_center center_to_right"/>
    <route id="down_uturn" edges="down_to_center center_to_down"/>"""
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