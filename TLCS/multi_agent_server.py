from flask import Flask, request, jsonify
import numpy as np
import os
from waitress import serve
import logging

# Suppress extra logging messages from werkzeug
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Import your project modules
from memory import Memory
from model import TrainModelAggregator

app = Flask(__name__)

# --- Define Hyperparameters for the Aggregator Agents ---
lane_feature_dim = 5
embedding_dim = 32
final_hidden = 64
batch_size = 100
learning_rate = 0.001
num_actions = 4

memory_size_min = 600
memory_size_max = 50000

# --- Create two Aggregator Agent Instances ---
# Agent for the left intersection
agent_left = TrainModelAggregator(
    lane_feature_dim=lane_feature_dim,
    embedding_dim=embedding_dim,
    final_hidden=final_hidden,
    num_actions=num_actions,
    batch_size=batch_size,
    learning_rate=learning_rate
)
mem_left = Memory(memory_size_max, memory_size_min)

# Agent for the right intersection
agent_right = TrainModelAggregator(
    lane_feature_dim=lane_feature_dim,
    embedding_dim=embedding_dim,
    final_hidden=final_hidden,
    num_actions=num_actions,
    batch_size=batch_size,
    learning_rate=learning_rate
)
mem_right = Memory(memory_size_max, memory_size_min)


# --- REST API Endpoints ---

@app.route('/initialize_agents', methods=['POST'])
def initialize_agents():
    """
    Initialize or update agent hyperparameters based on JSON settings.
    Expected keys: lane_feature_dim, embedding_dim, final_hidden, batch_size, learning_rate,
                   memory_size_max, memory_size_min
    """
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify(error=str(e)), 400

    # Update left agent
    agent_left._lane_feature_dim = data.get('lane_feature_dim', lane_feature_dim)
    agent_left._embedding_dim = data.get('embedding_dim', embedding_dim)
    agent_left._final_hidden = data.get('final_hidden', final_hidden)
    agent_left._batch_size = data.get('batch_size', batch_size)
    agent_left._learning_rate = data.get('learning_rate', learning_rate)
    mem_left._size_max = data.get('memory_size_max', memory_size_max)
    mem_left._size_min = data.get('memory_size_min', memory_size_min)

    # Update right agent
    agent_right._lane_feature_dim = data.get('lane_feature_dim', lane_feature_dim)
    agent_right._embedding_dim = data.get('embedding_dim', embedding_dim)
    agent_right._final_hidden = data.get('final_hidden', final_hidden)
    agent_right._batch_size = data.get('batch_size', batch_size)
    agent_right._learning_rate = data.get('learning_rate', learning_rate)
    mem_right._size_max = data.get('memory_size_max', memory_size_max)
    mem_right._size_min = data.get('memory_size_min', memory_size_min)

    return "Agents Initialized", 200


@app.route('/add_samples', methods=['POST'])
def add_samples():
    """
    Add training samples for both agents.
    The JSON payload must include samples for each agent with keys:
      - old_state_one, old_action_one, reward_one, current_state_one
      - old_state_two, old_action_two, reward_two, current_state_two
    """
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify(error=str(e)), 400

    try:
        # Process left agent sample
        sample_left = (
            np.array(data['old_state_one']),
            data['old_action_one'],
            data['reward_one'],
            np.array(data['current_state_one'])
        )
        mem_left.add_sample(sample_left)

        # Process right agent sample
        sample_right = (
            np.array(data['old_state_two']),
            data['old_action_two'],
            data['reward_two'],
            np.array(data['current_state_two'])
        )
        mem_right.add_sample(sample_right)
    except KeyError as e:
        return jsonify(error=f"Missing field: {str(e)}"), 400

    return "Samples added", 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Given a state and an agent identifier, return the Q-value predictions.
    Expected JSON payload must include:
      - state: the flat state vector for prediction
      - agent_num: 1 (left) or 2 (right)
    The state is reshaped into a 3D tensor of shape (1, num_lanes, lane_feature_dim)
    before being passed to the model.
    """
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify(error=str(e)), 400

    agent_num = data.get('agent_num', 1)
    try:
        state = np.array(data['state'])
    except KeyError as e:
        return jsonify(error=f"Missing field: {str(e)}"), 400

    # If state is a flat vector (1D), reshape it to (1, num_lanes, lane_feature_dim)
    if state.ndim == 1:
        total_length = state.shape[0]
        if total_length % lane_feature_dim != 0:
            return jsonify(error="State length is not a multiple of lane_feature_dim"), 400
        num_lanes = total_length // lane_feature_dim
        state = np.reshape(state, (1, num_lanes, lane_feature_dim))

    if agent_num == 1:
        model = agent_left
    elif agent_num == 2:
        model = agent_right
    else:
        return jsonify(error="Invalid agent number"), 400

    try:
        prediction = model.predict_one(state)
    except Exception as e:
        return jsonify(error=str(e)), 500

    return jsonify(prediction=prediction.tolist()), 200


@app.route('/replay', methods=['POST'])
def replay():
    """
    Sample a batch from memory, perform the Q-learning update, and train the network.
    Expected JSON payload should include:
      - gamma: discount factor (optional, default=0.75)
      - agent_num: 1 (left) or 2 (right)
    """
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify(error=str(e)), 400

    gamma = data.get('gamma', 0.75)
    agent_num = data.get('agent_num', 1)

    if agent_num == 1:
        model = agent_left
        mem = mem_left
    elif agent_num == 2:
        model = agent_right
        mem = mem_right
    else:
        return jsonify(error="Invalid agent number"), 400

    batch = mem.get_samples(model.batch_size)
    if len(batch) > 0:
        # Assuming the aggregator expects a state shape: (batch_size, num_lanes, lane_feature_dim)
        states = np.array([sample[0] for sample in batch])
        next_states = np.array([sample[3] for sample in batch])
        q_vals = model.predict_batch(states)
        next_q_vals = model.predict_batch(next_states)

        x = np.zeros(states.shape)
        y = np.copy(q_vals)

        for i, sample in enumerate(batch):
            state, action, reward, _ = sample
            y[i, action] = reward + gamma * np.amax(next_q_vals[i])
            x[i] = state
        model.train_batch(x, y)
        loss = model._model.evaluate(x, y, verbose=0)
    else:
        loss = None
    return jsonify(loss=loss), 200


@app.route('/save_models', methods=['POST'])
def save_models():
    """
    Save both agents' models into specified subdirectories.
    The JSON payload must include:
      - path: the base directory where models will be saved.
    """
    try:
        data = request.get_json()
    except Exception as e:
        return jsonify(error=str(e)), 400

    base_path = data.get('path', './models')
    agent_left_path = os.path.join(base_path, 'agent_left')
    agent_right_path = os.path.join(base_path, 'agent_right')
    os.makedirs(agent_left_path, exist_ok=True)
    os.makedirs(agent_right_path, exist_ok=True)

    agent_left.save_model(agent_left_path)
    agent_right.save_model(agent_right_path)
    return "Models saved", 200


if __name__ == '__main__':
    # Run the server with waitress on port 5000
    serve(app, host='127.0.0.1', port=5000)
