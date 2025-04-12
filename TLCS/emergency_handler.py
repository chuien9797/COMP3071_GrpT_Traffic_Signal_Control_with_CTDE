import traci


def check_emergency(simulation):
    """
    Check for the presence of an emergency vehicle on the incoming lanes.
    The inbound lane prefixes are chosen according to the simulation's intersection type.
    If an emergency vehicle is detected, the emergency handler is invoked.
    """
    # Determine expected lane prefixes based on intersection type.
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
        # Default fallback: use cross road prefixes.
        lane_prefixes = ["E2TL", "N2TL", "W2TL", "S2TL"]

    for veh in traci.vehicle.getIDList():
        if traci.vehicle.getTypeID(veh) == "emergency":
            lane_id = traci.vehicle.getLaneID(veh)
            if any(lane_id.startswith(prefix) for prefix in lane_prefixes):
                print("Emergency vehicle detected:", veh, "on lane:", lane_id)
                handle_emergency_vehicle(simulation, veh)
                return True
    return False


def handle_emergency_vehicle(simulation, veh_id):
    """
    Handles the emergency vehicle by commanding the traffic lights to switch to an emergency phase.
    The chosen phase depends on the simulation's intersection type and the vehicle's route.

    Example logic:
      - For a cross intersection, if the route is in straight routes, use 0 (NS green) or 2 (EW green);
        otherwise, use 1 (NS left-turn green) or 3 (EW left-turn green).
      - For roundabout, we assume only a single emergency action is defined.
      - For T and Y intersections, the logic is chosen as an example; you may need to adjust it.
    """
    route_id = traci.vehicle.getRouteID(veh_id)
    itype = simulation.intersection_type.lower()
    emergency_action = None

    if itype == "cross":
        # Define which route IDs correspond to "straight" movements.
        straight_routes = ["N_S", "S_N", "E_W", "W_E"]
        if route_id in straight_routes:
            # For NS straight, pick action 0; for EW, action 2.
            if route_id[0] in ['N', 'S']:
                emergency_action = 0
            else:
                emergency_action = 2
        else:
            # For turning movements: for NS, use action 1; else action 3.
            if route_id[0] in ['N', 'S']:
                emergency_action = 1
            else:
                emergency_action = 3

    elif itype == "roundabout":
        # For roundabouts, often the control is simpler.
        # Here we assume the roundabout configuration defines one or more actions.
        # For simplicity, choose action 0 as the emergency override.
        emergency_action = 0

    elif itype == "t_intersection":
        # Example logic: if the route involves the main road (e.g., contains "W_E" or "E_W"), use phase 0;
        # otherwise, use phase 1.
        if any(main in route_id for main in ["W_E", "E_W"]):
            emergency_action = 0
        else:
            emergency_action = 1

    elif itype == "y_intersection":
        # For Y intersections, if the route identifier contains a branch number, map accordingly.
        if "Y_branch1" in route_id:
            emergency_action = 0
        elif "Y_branch2" in route_id:
            emergency_action = 1
        elif "Y_branch3" in route_id:
            emergency_action = 2
        else:
            emergency_action = 0  # default fallback

    else:
        # Default behavior: use cross logic
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

    # print("Setting emergency phase:", emergency_action, "for vehicle:", veh_id, "with route:", route_id)
    if emergency_action is not None:
        simulation._set_green_phase(emergency_action)
        simulation._simulate(simulation._green_duration)
    else:
        print("No emergency action determined for vehicle:", veh_id)

    # Log the emergency event.
    if not hasattr(simulation, "_emergency_q_logs"):
        simulation._emergency_q_logs = []
    simulation._emergency_q_logs.append((simulation._step, veh_id))