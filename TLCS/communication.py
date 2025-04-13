"""
communication.py

A simple communication module for multiagent coordination in traffic simulation.

This module implements a basic message passing interface so that agents can
explicitly share information (e.g., their chosen actions, Q-values, or local observations)
with one another. It uses a global dictionary to store messages for each agent.

Notes:
  - In a multi-threaded or multi-process scenario, consider adding thread safety
    (e.g., via locks) or using a multiprocessing.Manager dictionary.
  - This module assumes that agent IDs are integers.
"""

# Global dictionary to hold messages for each agent.
agent_messages = {}

# Optionally, for thread safety you can import threading and use a Lock:
# import threading
# msg_lock = threading.Lock()

def send_message(agent_id, message):
    """
    Send a message to a specific agent.

    Parameters:
        agent_id (int): The identifier of the receiving agent.
        message (dict): The message (as a dictionary) to be sent.
    """
    # For thread safety in multithreading, uncomment the following:
    # with msg_lock:
    if agent_id not in agent_messages:
        agent_messages[agent_id] = []
    agent_messages[agent_id].append(message)
    # Optionally, log the message sending:
    # print(f"[Communication] Message sent to agent {agent_id}: {message}")


def get_messages(agent_id):
    """
    Retrieve and clear all messages for a given agent.

    Parameters:
        agent_id (int): The identifier for the agent.

    Returns:
        list: A list of messages that were stored for the agent.
    """
    # If thread safety is needed, wrap with lock:
    # with msg_lock:
    msgs = agent_messages.get(agent_id, [])
    # Clear messages after retrieval.
    agent_messages[agent_id] = []
    print(f"[Communication] Agent {agent_id} retrieved {len(msgs)} messages.")
    return msgs


def broadcast(message):
    """
    Broadcast a message to all agents that have messages waiting in the system.

    Parameters:
        message (dict): The message to broadcast.
    """
    # Loop over a snapshot of current keys.
    for agent_id in list(agent_messages.keys()):
        send_message(agent_id, message)
    # Optionally, log broadcasting:
    # print(f"[Communication] Broadcast message to all agents: {message}")
