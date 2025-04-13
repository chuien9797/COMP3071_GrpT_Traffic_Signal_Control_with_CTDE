import traci


def check_emergency(simulation):
    """
    Check for the presence of an emergency vehicle on the incoming lanes.
    The expected lane prefixes are determined by the simulation's intersection type.
    If an emergency vehicle is detected, the emergency handler is invoked
    for each intersection whose controlled lanes include that emergency vehicle.

    Returns:
         True if any emergency vehicle is processed; False otherwise.
    """
    # Determine lane prefixes based on intersection type.
    itype = simulation.intersection_type.lower()
    if itype == "cross":
        lane_prefixes = ["E2TL", "N2TL", "W2TL", "S2TL"]
    elif itype == "roundabout":
        lane_prefixes = ["e1", "e2", "e3", "e4"]
    elif itype == "t_intersection":
        lane_prefixes = ["left_in", "right_in", "top_in"]
    elif itype == "y_intersection":
        lane_prefixes = ["Y_branch1", "Y_branch2", "Y_branch3"]
    else:
        # Default: assume cross.
        lane_prefixes = ["E2TL", "N2TL", "W2TL", "S2TL"]

    for veh in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(veh) == "emergency":
            lane_id = traci.vehicle.getLaneID(veh)
            if any(lane_id.startswith(prefix) for prefix in lane_prefixes):
                # print("Emergency vehicle detected:", veh, "on lane:", lane_id)
                handle_emergency_vehicle(simulation, veh, lane_id)
                # Return True (at least one emergency was handled).
                return True
    return False


def handle_emergency_vehicle(simulation, veh_id, lane_id):
    """
    Handles an emergency vehicle by determining the appropriate emergency action
    based on the intersection type and the vehicle's route, and then applying
    the override only for those intersections (agents) whose controlled lanes include
    the emergency vehicle's lane.

    For example, for a cross intersection, if the vehicle's route is straight,
    the action may be chosen as 0 (NS green) or 2 (EW green); for turning, a different action.

    This function now iterates over the traffic lights defined in the intersection configuration.

    Parameters:
       simulation: The Simulation instance.
       veh_id (str): Emergency vehicle ID.
       lane_id (str): The lane in which the vehicle is found.
    """
    route_id = traci.vehicle.getRouteID(veh_id)
    itype = simulation.intersection_type.lower()
    emergency_action = None

    if itype == "cross":
        straight_routes = ["N_S", "S_N", "E_W", "W_E"]
        if route_id in straight_routes:
            if route_id[0] in ['N', 'S']:
                emergency_action = 0
            else:
                emergency_action = 2
        else:
            if route_id[0] in ['N', 'S']:
                emergency_action = 1
            else:
                emergency_action = 3

    elif itype == "roundabout":
        emergency_action = 0

    elif itype == "t_intersection":
        if any(main in route_id for main in ["W_E", "E_W"]):
            emergency_action = 0
        else:
            emergency_action = 1

    elif itype == "y_intersection":
        if "Y_branch1" in route_id:
            emergency_action = 0
        elif "Y_branch2" in route_id:
            emergency_action = 1
        elif "Y_branch3" in route_id:
            emergency_action = 2
        else:
            emergency_action = 0

    else:
        # Default cross logic
        straight_routes = ["N_S", "S_N", "E_W", "W_E"]
        if route_id in straight_routes:
            if route_id[0] in ['N', 'S']:
                emergency_action = 0
            else:
                emergency_action = 2
        else:
            if route_id[0] in ['N', 'S']:
                emergency_action = 1
            else:
                emergency_action = 3

    if emergency_action is not None:
        # For each intersection, check if its controlled lanes include lane_id.
        tl_ids = simulation.int_conf.get("traffic_light_ids", [])
        for index, tlid in enumerate(tl_ids):
            try:
                controlled_lanes = traci.trafficlight.getControlledLanes(tlid)
            except Exception as e:
                print(f"Error retrieving controlled lanes for {tlid}: {e}")
                continue
            if lane_id in controlled_lanes:
                # print(
                #     f"Setting emergency phase {emergency_action} at intersection {index + 1} (TL {tlid}) for vehicle {veh_id}")
                simulation._set_green_phase(index, emergency_action)
                # Advance simulation using the green duration for emergency override.
                simulation._simulate(simulation._green_duration)
    else:
        print("No emergency action determined for vehicle:", veh_id)

    # Log the emergency event.
    if not hasattr(simulation, "_emergency_q_logs"):
        simulation._emergency_q_logs = []
    simulation._emergency_q_logs.append((simulation._step, veh_id))
