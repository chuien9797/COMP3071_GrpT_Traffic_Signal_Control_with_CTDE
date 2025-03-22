# File: env_utils.py
import numpy as np
import traci

def get_state(num_states):
    """
    Returns the state of the intersection as an occupancy grid.
    Modify the logic as needed for your environment.
    """
    state = np.zeros(num_states)
    car_list = traci.vehicle.getIDList()
    for car_id in car_list:
        lane_pos = traci.vehicle.getLanePosition(car_id)
        lane_id = traci.vehicle.getLaneID(car_id)
        # Invert lane position so that lower values mean closer to the intersection
        lane_pos = 750 - lane_pos
        # Determine lane cell based on thresholds
        if lane_pos < 7:
            lane_cell = 0
        elif lane_pos < 14:
            lane_cell = 1
        elif lane_pos < 21:
            lane_cell = 2
        elif lane_pos < 28:
            lane_cell = 3
        elif lane_pos < 40:
            lane_cell = 4
        elif lane_pos < 60:
            lane_cell = 5
        elif lane_pos < 100:
            lane_cell = 6
        elif lane_pos < 160:
            lane_cell = 7
        elif lane_pos < 400:
            lane_cell = 8
        else:
            lane_cell = 9

        # Map lane IDs to lane groups (update these mappings as needed)
        if lane_id in ["W2TL_0", "W2TL_1", "W2TL_2"]:
            lane_group = 0
        elif lane_id == "W2TL_3":
            lane_group = 1
        elif lane_id in ["N2TL_0", "N2TL_1", "N2TL_2"]:
            lane_group = 2
        elif lane_id == "N2TL_3":
            lane_group = 3
        elif lane_id in ["E2TL_0", "E2TL_1", "E2TL_2"]:
            lane_group = 4
        elif lane_id == "E2TL_3":
            lane_group = 5
        elif lane_id in ["S2TL_0", "S2TL_1", "S2TL_2"]:
            lane_group = 6
        elif lane_id == "S2TL_3":
            lane_group = 7
        else:
            lane_group = -1

        if lane_group >= 0:
            # Combine lane_group and lane_cell to compute an index; adjust as needed
            pos_index = lane_group * 10 + lane_cell
            if pos_index < num_states:
                state[pos_index] = 1
    return state

def set_green_phase(action):
    """
    Sets the green phase corresponding to the given action.
    Modify the phase codes if your environment uses different numbers.
    """
    if action == 0:
        traci.trafficlight.setPhase("TL", 0)  # Example: NS_GREEN
    elif action == 1:
        traci.trafficlight.setPhase("TL", 2)  # Example: NSL_GREEN
    elif action == 2:
        traci.trafficlight.setPhase("TL", 4)  # Example: EW_GREEN
    elif action == 3:
        traci.trafficlight.setPhase("TL", 6)  # Example: EWL_GREEN

def compute_reward(old_total_wait, current_total_wait, emergency_penalty=0):
    """
    Computes the reward as the reduction in waiting time minus any penalty.
    Adjust the computation if you have a different reward structure.
    """
    return (old_total_wait - current_total_wait) - emergency_penalty

def get_total_wait():
    """
    Sums the waiting times of vehicles on incoming roads.
    Modify the list of incoming roads as needed.
    """
    incoming_roads = ["E2TL", "N2TL", "W2TL", "S2TL"]
    total_wait = 0
    for veh in traci.vehicle.getIDList():
        road_id = traci.vehicle.getRoadID(veh)
        if road_id in incoming_roads:
            total_wait += traci.vehicle.getAccumulatedWaitingTime(veh)
    return total_wait
