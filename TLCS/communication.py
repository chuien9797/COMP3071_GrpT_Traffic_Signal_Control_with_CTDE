"""
Simple message‑passing utilities for multi‑agent coordination.

Agents can explicitly share information (chosen actions, Q‑values, local
observations, etc.) by writing and reading dictionaries from an in‑memory
mailbox. Each agent has its own FIFO inbox keyed by its integer ID.
"""

from typing import Any, Dict, List

agent_messages: Dict[int, List[Dict[str, Any]]] = {}

def init_agents(agent_ids: List[int]) -> None:
    """
    (Optional) Pre‑register a list of agent IDs with empty inboxes.

    This is useful if you plan to call ``broadcast()`` before any agent
    has sent its first message.
    """
    for aid in agent_ids:
        agent_messages.setdefault(aid, [])


def send_message(agent_id: int, message: Dict[str, Any]) -> None:
    """
    Append *message* to *agent_id*'s inbox.
    """
    agent_messages.setdefault(agent_id, []).append(message)


def get_messages(agent_id: int) -> List[Dict[str, Any]]:
    """
    Retrieve **and clear** all pending messages for *agent_id*.
    """
    msgs = agent_messages.get(agent_id, [])
    agent_messages[agent_id] = []
    print(f"[Communication] Agent {agent_id} retrieved {len(msgs)} messages.")
    return msgs


def broadcast(message: Dict[str, Any]) -> None:
    """
    Send *message* to every agent currently known to the mailbox.
    Agents that haven’t been registered yet (no inbox) will be skipped,
    unless you call ``init_agents()`` first.
    """
    for aid in list(agent_messages.keys()):
        send_message(aid, message)
