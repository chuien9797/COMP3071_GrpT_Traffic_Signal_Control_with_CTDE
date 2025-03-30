from __future__ import absolute_import, print_function


def compute_environment_parameters(intersection_conf):
    """
    Compute dynamic parameters from the intersection configuration.

    Args:
        intersection_conf (dict): Intersection configuration dictionary.

    Returns:
        num_states (int): Calculated state dimension.
        num_actions (int): Calculated number of actions.
    """
    incoming_lanes = intersection_conf["incoming_lanes"]
    total_lanes = sum(len(lanes) for lanes in incoming_lanes.values())
    cells_per_lane = intersection_conf["occupancy_grid"]["cells_per_lane"]
    num_emergency_flags = len(incoming_lanes.keys())
    num_states = total_lanes * cells_per_lane + num_emergency_flags

    num_actions = len(intersection_conf["phase_mapping"])
    return num_states, num_actions


def build_dynamic_model(input_dim, output_dim, hidden_layers):
    """
    Build a neural network model dynamically using the computed dimensions.

    Args:
        input_dim (int): Dimension of the input layer.
        output_dim (int): Dimension of the output layer.
        hidden_layers (list): List of integers representing hidden layer sizes.

    Returns:
        model: A compiled neural network model.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, losses
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    for size in hidden_layers:
        model.add(layers.Dense(size, activation='relu'))
    model.add(layers.Dense(output_dim, activation='linear'))
    model.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError())
    return model