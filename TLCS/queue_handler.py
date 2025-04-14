import traci

def should_interrupt_for_queue(simulation):
    """
    Return True if any red-lane has queue length above threshold (e.g., > 5),
    and log the queue info.
    """
    state = simulation._get_state()
    tl_ids = simulation.int_conf["traffic_light_ids"]
    current_phase = traci.trafficlight.getPhase(tl_ids[0])
    logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_ids[0])[0]
    phase_state = logic.phases[current_phase].state
    controlled_lanes = traci.trafficlight.getControlledLanes(tl_ids[0])

    for i, lane_id in enumerate(simulation._get_lane_order()):
        if lane_id in controlled_lanes:
            idx = controlled_lanes.index(lane_id)
            if idx >= len(phase_state):
                continue
            if phase_state[idx] == 'r':  # lane has red signal
                queue_len = state[i, 3]  # column 3 is queue length
                if queue_len > 5:  # threshold for urgent queue
                    # ✅ Log the interrupt reason
                    simulation._queue_interrupt_log.append({
                        "step": simulation._step,
                        "lane": lane_id,
                        "queue_len": queue_len
                    })
                    return True
    return False


def interrupt_for_queue(simulation):
    """
    Interrupt current green phase and switch to better one based on queue.
    """
    current_state = simulation._get_state()
    current_action = simulation._get_current_action_from_phase()

    simulation._set_yellow_phase(current_action)
    simulation._simulate(simulation._yellow_duration)

    new_action = simulation._choose_action(current_state, epsilon=0.0)
    simulation._set_green_phase(new_action)
    adaptive_green = simulation._compute_adaptive_green_duration(current_state)
    simulation._green_durations_log.append(adaptive_green)
    print(f"[INTERRUPT] Switched early to action {new_action} with green duration {adaptive_green}")
    simulation._simulate(adaptive_green)


def get_current_action_from_phase(simulation):
    tlid = simulation.int_conf["traffic_light_ids"][0]
    current_phase = traci.trafficlight.getPhase(tlid)
    for a, mapping in simulation.int_conf["phase_mapping"].items():
        if mapping["green"] == current_phase:
            return a
    return 0  # fallback default
