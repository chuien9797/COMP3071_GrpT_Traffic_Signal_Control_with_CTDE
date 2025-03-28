"""
Intersection configuration for various intersection types.

This file defines different intersection configurations for your RL traffic agent.
For each intersection type, we specify:
  - incoming_lanes: a dictionary mapping lane groups (e.g., "N", "S", "E", "W" or other group names) to lists of lane IDs.
  - phase_mapping: a dictionary mapping each abstract action (0, 1, etc.) to a dictionary with keys "green" and "yellow",
                   which specify the corresponding SUMO phase numbers for the green and yellow phases.
  - occupancy_grid: parameters used to discretize lane positions into grid cells (e.g., number of cells per lane and maximum distance).
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
            0: {"green": 0, "yellow": 1},  # Abstract action 0 activates NS green
            1: {"green": 2, "yellow": 3},  # Abstract action 1 activates NS left-turn green
            2: {"green": 4, "yellow": 5},  # Abstract action 2 activates EW green
            3: {"green": 6, "yellow": 7}   # Abstract action 3 activates EW left-turn green
        },
        "occupancy_grid": {
            "cells_per_lane": 10,  # Divide each lane into 10 cells
            "max_distance": 750    # Maximum distance from the traffic light to consider (in meters)
        }
    },
    "roundabout": {
        "incoming_lanes": {
            # Define the lane IDs as in your roundabout SUMO network.
            "in": ["roundabout_in_0", "roundabout_in_1"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1}  # Adjust these phase codes to your roundabout’s logic
            # Add additional actions if your roundabout design requires them.
        },
        "occupancy_grid": {
            "cells_per_lane": 8,
            "max_distance": 500
        }
    },
    "T_intersection": {
        "incoming_lanes": {
            "main": ["left_in_0", "left_in_1", "right_in_0", "right_in_1"],
            "side": ["top_in_0", "top_in_1"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1},   # Action 0: main road green
            1: {"green": 2, "yellow": 3},   # Action 1: side road green
            2: {"green": 0, "yellow": 1},   # Action 2: main road green (duplicate of action 0)
            3: {"green": 2, "yellow": 3}    # Action 3: side road green (duplicate of action 1)
        },
        "occupancy_grid": {
            "cells_per_lane": 8,
            "max_distance": 600
        }
    },
    "Y_intersection": {
        "incoming_lanes": {
            "branch1": ["Y_branch1_0", "Y_branch1_1"],
            "branch2": ["Y_branch2_0", "Y_branch2_1"],
            "branch3": ["Y_branch3_0", "Y_branch3_1"]
        },
        "phase_mapping": {
            0: {"green": 0, "yellow": 1},  # Action 0 for branch1
            1: {"green": 2, "yellow": 3},  # Action 1 for branch2
            2: {"green": 4, "yellow": 5}   # Action 2 for branch3
        },
        "occupancy_grid": {
            "cells_per_lane": 8,
            "max_distance": 600
        }
    }
}
