from __future__ import absolute_import, print_function

import sys
import os
import datetime
from shutil import copyfile
import configparser
import random
import timeit

# Append the parent directory so that 'TLCS' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import import_train_configuration, set_sumo, set_train_path
from environment_utils import compute_environment_parameters, build_dynamic_model

from TLCS.rl_models.ppo_model import PPOModel, PPOSimulation

from generator import TrafficGenerator
from visualization import Visualization

import intersection_config as int_config

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.optimizers import Adam

import traci  # SUMO interface


# ------------------ MAML Helper Functions ------------------

def dqn_loss(model, states, actions, targets):
    """
    Compute the DQN loss (MSE) between Q-values for selected actions and target Q-values.
    """
    q_values = model(states)
    actions_one_hot = tf.one_hot(actions, depth=q_values.shape[-1])
    q_selected = tf.reduce_sum(q_values * actions_one_hot, axis=1)
    loss_val = tf.reduce_mean(tf.square(targets - q_selected))
    return loss_val

def set_model_weights(model, new_weights):
    """
    Assign new_weights (a list of tensors) to model.trainable_variables.
    """
    for var, new_w in zip(model.trainable_variables, new_weights):
        var.assign(new_w)

def collect_experience(simulation, batch_size=32, steps=100):
    """
    Run the simulation for a fixed number of steps to collect a mini-batch of experience.
    This version runs a mini-run of the simulation loop using random actions and a dummy reward.
    It resets the simulation's step counter and clears memory before collecting experience.
    (In your final implementation, replace the random action and dummy reward with real simulation data.)
    """
    # Ensure connection to SUMO
    try:
        traci.simulationStep()
    except traci.exceptions.FatalTraCIError:
        traci.start(simulation._sumo_cmd)
    try:
        # Clear memory and reset step counter
        simulation._Memory._samples = []
        simulation._step = 0
        old_state = simulation._get_state()
        for _ in range(steps):
            # Choose a random action for simulation progression
            action = random.randint(0, simulation._num_actions - 1)
            simulation._set_green_phase(action)
            simulation._simulate(1)
            new_state = simulation._get_state()
            # Compute a dummy reward (for instance, 0 or based on waiting time difference)
            reward = 0
            # Only add sample if not the first step
            if simulation._step != 0:
                simulation._Memory.add_sample((old_state, action, reward, new_state))
            old_state = new_state
        batch = simulation._Memory.get_samples(batch_size)
        return batch
    finally:
        traci.close()

def compute_targets(model, batch, gamma):
    """
    Given a batch of experience tuples (state, action, reward, next_state),
    compute the target Q-values as: target = reward + gamma * max(Q(next_state)).
    Returns tensors for states, actions, and computed targets.
    """
    states, actions, rewards, next_states = zip(*batch)
    states = tf.convert_to_tensor(np.array(states), dtype=tf.float32)
    actions = tf.convert_to_tensor(np.array(actions), dtype=tf.int32)
    rewards = tf.convert_to_tensor(np.array(rewards), dtype=tf.float32)
    next_states = tf.convert_to_tensor(np.array(next_states), dtype=tf.float32)
    q_next = model(next_states)
    max_q_next = tf.reduce_max(q_next, axis=1)
    targets = rewards + gamma * max_q_next
    return states, actions, targets

# ------------------ End MAML Helper Functions ------------------


# ------------------ Main Code (Configuration & Setup) ------------------

if __name__ == "__main__":
    # Load configuration and set up simulation paths
    config = import_train_configuration("training_settings.ini")

    # Check for command line override of intersection type
    if len(sys.argv) > 1:
        print("Overriding intersection type with command line argument:", sys.argv[1])
        config['intersection_type'] = sys.argv[1]

    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    # Load intersection configuration based on intersection type
    intersection_type = config.get('intersection_type', 'cross')
    if intersection_type not in int_config.INTERSECTION_CONFIGS:
        raise ValueError("Intersection type '{}' not found in configuration.".format(intersection_type))
    int_conf = int_config.INTERSECTION_CONFIGS[intersection_type]

    # Compute environment parameters
    num_states, num_actions = compute_environment_parameters(int_conf)
    config['num_states'] = num_states
    config['num_actions'] = num_actions

    print("Computed num_states:", num_states)
    print("Computed num_actions:", num_actions)

    algorithm = config.get('algorithm', 'DQN')

    if algorithm == 'PPO':
        model = PPOModel(
            input_dim=config['num_states'],
            output_dim=config['num_actions'],
            hidden_size=config['ppo_hidden_size'],
            learning_rate=config['ppo_learning_rate'],
            clip_ratio=config['ppo_clip_ratio'],
            update_epochs=config['ppo_update_epochs']
        )
        traffic_gen = TrafficGenerator(config['max_steps'], config['n_cars_generated'],
                                       intersection_type=intersection_type)
        Simulation_obj = PPOSimulation(
            model=model,
            traffic_gen=traffic_gen,
            sumo_cmd=sumo_cmd,
            gamma=config['gamma'],
            max_steps=config['max_steps'],
            green_duration=config['green_duration'],
            yellow_duration=config['yellow_duration'],
            num_states=config['num_states'],
            num_actions=config['num_actions'],
            training_epochs=config['ppo_training_epochs'],
            intersection_type=intersection_type
        )
        print("PPO branch not set up for MAML meta-training. Exiting.")
        sys.exit(1)
    elif algorithm == 'DQN':
        from model import TrainModel
        from training_simulation import Simulation
        from memory import Memory

        hidden_layers = config.get('dqn_hidden_layers', [64, 64])
        dynamic_model = build_dynamic_model(num_states, num_actions, hidden_layers)

        Model_obj = TrainModel(
            int(config['num_layers']),
            int(config['width_layers']),
            int(config['batch_size']),
            float(config['learning_rate']),
            input_dim=int(config['num_states']),
            output_dim=int(config['num_actions']),
            model=dynamic_model
        )
        MemoryInstance = Memory(
            int(config['memory_size_max']),
            int(config['memory_size_min'])
        )
        TrafficGen = TrafficGenerator(int(config['max_steps']), int(config['n_cars_generated']),
                                      intersection_type=intersection_type)
        Simulation_obj = Simulation(
            Model_obj,
            MemoryInstance,
            TrafficGen,
            sumo_cmd,
            float(config['gamma']),
            int(config['max_steps']),
            int(config['green_duration']),
            int(config['yellow_duration']),
            int(config['num_states']),
            int(config['num_actions']),
            int(config['training_epochs']),
            intersection_type=intersection_type
        )
    else:
        raise ValueError("Unsupported algorithm: {}. Please choose either 'PPO' or 'DQN'.".format(algorithm))

    # ------------------ MAML Meta-Training Loop ------------------
    # Define tasks (each task is an intersection type)
    tasks = ["cross", "roundabout", "T_intersection"]

    meta_steps = 1000      # Number of meta-training iterations
    inner_steps = 1        # Number of inner-loop updates per task
    inner_lr = 0.01        # Inner-loop learning rate
    meta_lr = 0.001        # Meta (outer-loop) learning rate
    gamma = float(config['gamma'])

    optimizer = Adam(meta_lr)

    # Use the computed dimensions
    input_dim = int(config['num_states'])
    output_dim = int(config['num_actions'])

    print("\nStarting MAML meta-training for DQN...\n")
    for meta_iter in range(meta_steps):
        # Initialize meta-gradient accumulator
        meta_gradient_sum = [tf.zeros_like(var) for var in Model_obj._model.trainable_variables]

        # Loop over each task
        for task in tasks:
            # Reconfigure the simulation for the given task
            Simulation_obj.intersection_type = task
            if task not in int_config.INTERSECTION_CONFIGS:
                raise ValueError("Intersection type '{}' not found.".format(task))
            Simulation_obj.int_conf = int_config.INTERSECTION_CONFIGS[task]

            # Save the current meta-parameters (original weights)
            original_weights = [w.numpy() for w in Model_obj._model.trainable_variables]

            # -------- Inner Loop: Adaptation on task --------
            batch_inner = collect_experience(Simulation_obj, batch_size=32, steps=100)
            # Check that batch_inner is not empty (if empty, skip this task)
            if len(batch_inner) < 4:
                continue
            states_inner, actions_inner, targets_inner = compute_targets(Model_obj._model, batch_inner, gamma)
            adapted_weights = Model_obj._model.trainable_variables
            for _ in range(inner_steps):
                with tf.GradientTape() as tape:
                    set_model_weights(Model_obj._model, adapted_weights)
                    loss_inner = dqn_loss(Model_obj._model, states_inner, actions_inner, targets_inner)
                grads = tape.gradient(loss_inner, Model_obj._model.trainable_variables)
                adapted_weights = [w - inner_lr * g for w, g in zip(Model_obj._model.trainable_variables, grads)]
            # -------- End Inner Loop --------

            # -------- Outer Loop: Meta-loss computation --------
            batch_meta = collect_experience(Simulation_obj, batch_size=32, steps=100)
            if len(batch_meta) < 4:
                continue
            states_meta, actions_meta, targets_meta = compute_targets(Model_obj._model, batch_meta, gamma)
            set_model_weights(Model_obj._model, adapted_weights)
            with tf.GradientTape() as meta_tape:
                meta_loss = dqn_loss(Model_obj._model, states_meta, actions_meta, targets_meta)
            meta_grads = meta_tape.gradient(meta_loss, Model_obj._model.trainable_variables)
            meta_gradient_sum = [mg + g for mg, g in zip(meta_gradient_sum, meta_grads)]
            # Restore the original meta-parameters for the next task
            set_model_weights(Model_obj._model, original_weights)
        # End task loop

        # Average meta-gradients over tasks and apply the update
        meta_gradient_avg = [mg / len(tasks) for mg in meta_gradient_sum]
        optimizer.apply_gradients(zip(meta_gradient_avg, Model_obj._model.trainable_variables))

        if meta_iter % 10 == 0:
            print("Meta iteration", meta_iter, "completed with meta loss:", meta_loss.numpy())
    # ------------------ End Meta-Training Loop ------------------

    meta_training_end_time = datetime.datetime.now()
    print("\nMeta-training started at:", meta_training_end_time - datetime.timedelta(seconds=meta_steps))
    print("Meta-training ended at:", meta_training_end_time)

    # Save the meta-trained model and configuration for future reference
    if algorithm == 'DQN':
        Model_obj.save_model(path)
    copyfile(src="training_settings.ini", dst=os.path.join(path, "training_settings.ini"))

    # Visualization (using your existing stats, if desired)
    viz = Visualization(path, dpi=96)
    viz.save_data_and_plot(data=Simulation_obj.reward_store, filename="reward",
                           xlabel="Episode", ylabel="Cumulative negative reward")
    viz.save_data_and_plot(data=Simulation_obj.cumulative_wait_store, filename="delay",
                           xlabel="Episode", ylabel="Cumulative delay (s)")
    viz.save_data_and_plot(data=Simulation_obj.avg_queue_length_store, filename="queue",
                           xlabel="Episode", ylabel="Average queue length (vehicles)")
