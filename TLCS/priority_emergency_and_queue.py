def prioritize_emergency_and_queue(q_vals, state, phase_mapping, incoming_lanes, num_actions, tl_ids, traci):
    """
    Boost Q-values of actions that open:
    1. Lanes with emergency vehicles (highest priority)
    2. Lanes with high queue lengths (secondary priority)
    """
    lane_order = []
    for group in sorted(incoming_lanes.keys()):
        lane_order.extend(incoming_lanes[group])

    for a in range(num_actions):
        if a not in phase_mapping:
            continue

        green_phase = phase_mapping[a]["green"]

        for tlid in tl_ids:
            try:
                logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tlid)[0]
                if green_phase >= len(logic.phases):
                    continue
                state_str = logic.phases[green_phase].state
                controlled = traci.trafficlight.getControlledLanes(tlid)

                for i, lane_id in enumerate(lane_order):
                    if lane_id in controlled:
                        idx = controlled.index(lane_id)
                        if state_str[idx] == 'G':
                            # Boost for emergency
                            if state[i, 2] > 0:  # emergency vehicle flag
                                q_vals[a] += 2.0  # EMERGENCY BONUS (high priority)
                            # Boost for queue
                            q_vals[a] += state[i, 3] * 0.1  # QUEUE LENGTH BOOST (tuneable)
            except:
                continue

    return q_vals
