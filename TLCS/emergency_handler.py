import traci

def check_emergency(simulation):
    """
    Check for the presence of an emergency vehicle on the incoming lanes.
    Avoids handling the same emergency vehicle multiple times.
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
        lane_prefixes = ["E2TL", "N2TL", "W2TL", "S2TL"]

    # Initialize the handled set if not yet created
    if not hasattr(simulation, "handled_emergency_ids"):
        simulation.handled_emergency_ids = set()

    for veh in traci.vehicle.getIDList():
        if veh in simulation.handled_emergency_ids:
            continue

        if traci.vehicle.getTypeID(veh) == "emergency":
            lane_id = traci.vehicle.getLaneID(veh)
            if any(lane_id.startswith(prefix) for prefix in lane_prefixes):
                print("Emergency vehicle detected:", veh, "on lane:", lane_id)
                simulation.handled_emergency_ids.add(veh)
                handle_emergency_vehicle(simulation, veh)
                return True
    return False


def handle_emergency_vehicle(simulation, veh_id):
    """
    Handles the emergency vehicle by commanding the traffic lights to switch to an emergency phase.
    """
    route_id = traci.vehicle.getRouteID(veh_id)
    itype = simulation.intersection_type.lower()
    emergency_action = None

    if itype == "cross":
        straight_routes = ["N_S", "S_N", "E_W", "W_E"]
        if route_id in straight_routes:
            emergency_action = 0 if route_id[0] in ['N', 'S'] else 2
        else:
            emergency_action = 1 if route_id[0] in ['N', 'S'] else 3

    elif itype == "roundabout":
        emergency_action = 0  # You can design this further per TL

    elif itype == "t_intersection":
        emergency_action = 0 if any(main in route_id for main in ["W_E", "E_W"]) else 1

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
        emergency_action = 0

    # 🔁 Loop over all traffic lights (multi-agent)
    for i in range(len(simulation.int_conf.get("traffic_light_ids", []))):
        simulation._set_green_phase(i, emergency_action)

    # Simulate emergency duration
    simulation._simulate(simulation._green_duration)

    if not hasattr(simulation, "_emergency_q_logs"):
        simulation._emergency_q_logs = []
    simulation._emergency_q_logs.append((simulation._step, veh_id))
