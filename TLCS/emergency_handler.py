# emergency_handler.py

import traci
import numpy as np


def check_emergency(simulation):
    """
    Check whether any lane in any agent's state has an emergency vehicle.
    This function aggregates over each agent's state returned by simulation._get_state().
    It assumes that the emergency indicator is at index 2 in the lane feature vector.

    Parameters:
        simulation: The simulation instance.

    Returns:
        True if any lane across all agents has an emergency flag set; False otherwise.
    """
    # Get the state for each agent (each node)
    states = simulation._get_state()
    for state in states:
        # Check the emergency flag (feature index 2) in each lane state.
        if np.any(state[:, 2] > 0):
            return True
    return False


def handle_emergency_vehicle(simulation, agent_index=None):
    """
    Adjust traffic light phases to expedite clearance of emergency vehicles.

    If an agent_index is provided, only that agent's traffic light is forced to the emergency green.
    Otherwise, the function iterates through all agents: if an emergency flag is detected in that agent's lanes,
    its traffic light is forced to an emergency green phase.

    NOTE: In this example, we assume that phase 0 represents an emergency green phase.

    Parameters:
        simulation: The simulation instance.
        agent_index (optional): An integer index of the agent to process.
                            If None, process all agents.
    """
    tl_ids = simulation.int_conf.get("traffic_light_ids", [])

    # If a specific agent is requested...
    if agent_index is not None:
        if agent_index < len(tl_ids):
            tlid = tl_ids[agent_index]
            try:
                # Immediately force the traffic light to an emergency green phase (assumed phase index 0)
                traci.trafficlight.setProgram(tlid, "0")
                traci.trafficlight.setPhase(tlid, 0)
                print(f"[Emergency] Agent {agent_index} (TL {tlid}): Emergency green activated.")
            except Exception as e:
                print(f"[Emergency] Error handling emergency for agent {agent_index} (TL {tlid}): {e}")
    else:
        # Process all agents by checking their state for an emergency flag.
        states = simulation._get_state()
        for idx, state in enumerate(states):
            # If any lane in this agent's state has emergency flag...
            if np.any(state[:, 2] > 0):
                if idx < len(tl_ids):
                    tlid = tl_ids[idx]
                    try:
                        traci.trafficlight.setProgram(tlid, "0")
                        traci.trafficlight.setPhase(tlid, 0)
                        print(f"[Emergency] Agent {idx} (TL {tlid}): Emergency green activated.")
                    except Exception as e:
                        print(f"[Emergency] Error handling emergency for agent {idx} (TL {tlid}): {e}")
