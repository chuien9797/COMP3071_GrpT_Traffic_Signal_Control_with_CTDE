# communication.py

"""
A simple communication module for multiagent coordination in traffic simulation.

This module implements a basic message passing interface so that agents can explicitly share
information (e.g., their chosen actions, Q-values, or local observations) with one another.
It uses a global dictionary to store messages for each agent.
"""

# Global dictionary to hold messages for each agent.
agent_messages = {}

def send_message(agent_id, message):
    """
    Send a message to a specific agent.

    Parameters:
        agent_id (int): The identifier of the receiving agent.
        message (dict): The message (as a dictionary) to be sent.
    """
    if agent_id not in agent_messages:
        agent_messages[agent_id] = []
    agent_messages[agent_id].append(message)

def get_messages(agent_id):
    """
    Retrieve and clear all messages for a given agent.

    Parameters:
        agent_id (int): The identifier for the agent.

    Returns:
        list: A list of messages that were stored for the agent.
    """
    msgs = agent_messages.get(agent_id, [])
    agent_messages[agent_id] = []
    print(f"[Communication] Agent {agent_id} retrieved {len(msgs)} messages.")
    return msgs

def broadcast(message):
    """
    Broadcast a message to all agents that have messages waiting in the system.

    Parameters:
        message (dict): The message to broadcast.
    """
    for agent_id in list(agent_messages.keys()):
        send_message(agent_id, message)
