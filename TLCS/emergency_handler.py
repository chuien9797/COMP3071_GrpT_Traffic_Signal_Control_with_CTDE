import traci

def check_emergency(simulation):
    """
    Check if any emergency vehicle is present in the incoming lanes.
    If found, call handle_emergency_vehicle and return True; otherwise return False.
    """
    for veh in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(veh) == "emergency":
            lane_id = traci.vehicle.getLaneID(veh)
            if lane_id in ["E2TL", "N2TL", "W2TL", "S2TL"]:
                print("Emergency vehicle detected:", veh)
                handle_emergency_vehicle(simulation, veh)
                return True
    return False

def handle_emergency_vehicle(simulation, veh_id):
    """
    Handle the emergency vehicle by setting the appropriate traffic light phase.
    simulation: an instance of the Simulation class (to access _set_green_phase and _simulate).
    """
    route_id = traci.vehicle.getRouteID(veh_id)
    straight_routes = ["N_S", "S_N", "E_W", "W_E"]
    if route_id in straight_routes:
        if route_id[0] in ['N', 'S']:
            emergency_action = 0  # NS_GREEN
        else:
            emergency_action = 2  # EW_GREEN
    else:
        if route_id[0] in ['N', 'S']:
            emergency_action = 1  # NSL_GREEN
        else:
            emergency_action = 3  # EWL_GREEN
    print("Setting emergency phase:", emergency_action, "for vehicle:", veh_id)
    simulation._set_green_phase(emergency_action)
    simulation._simulate(simulation._green_duration)
