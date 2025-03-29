from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import datetime
from shutil import copyfile

from TLCS.rl_models.ppo_model import PPOModel, PPOSimulation

# Append the parent directory so that 'TLCS' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import import_train_configuration, set_sumo, set_train_path
from environment_utils import compute_environment_parameters, build_dynamic_model

# Import PPO classes from our new merged file

# If DQN is used, keep the old simulation (and memory) imports:
# from model import TrainModel
# from training_simulation import Simulation
# from memory import Memory

from generator import TrafficGenerator
from visualization import Visualization

import intersection_config as int_config

if __name__ == "__main__":
    # Load configuration and set up simulation paths
    config = import_train_configuration("training_settings.ini")
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    import tensorflow as tf

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    # Load intersection configuration based on intersection type
    intersection_type = config.get('intersection_type', 'cross')
    if intersection_type not in int_config.INTERSECTION_CONFIGS:
        raise ValueError("Intersection type '{}' not found in configuration.".format(intersection_type))
    int_conf = int_config.INTERSECTION_CONFIGS[intersection_type]

    # Automatically compute state and action dimensions from intersection configuration
    num_states, num_actions = compute_environment_parameters(int_conf)
    config['num_states'] = num_states
    config['num_actions'] = num_actions

    print("Computed num_states:", num_states)
    print("Computed num_actions:", num_actions)

    # Choose the algorithm based on the configuration
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
        Simulation = PPOSimulation(
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

    elif algorithm == 'DQN':
        # Instantiate DQN model and simulation as before.
        from model import TrainModel
        from training_simulation import Simulation
        from memory import Memory

        # Build dynamic neural network model for DQN
        hidden_layers = config.get('dqn_hidden_layers', [64, 64])
        dynamic_model = build_dynamic_model(num_states, num_actions, hidden_layers)

        Model = TrainModel(
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
        Simulation = Simulation(
            Model,
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
            intersection_type=intersection_type  # NEW parameter
        )
    else:
        raise ValueError("Unsupported algorithm: {}. Please choose either 'PPO' or 'DQN'.".format(algorithm))

    total_episodes = int(config['total_episodes'])
    start_time = datetime.datetime.now()

    if algorithm == 'PPO':
        for ep in range(total_episodes):
            print(f"----- Episode {ep + 1} of {total_episodes}")
            states, actions, rewards, reward_sum, sim_time = Simulation.run_episode(ep)
            train_time = Simulation.update(states, actions, rewards)
            print(f"Reward: {reward_sum} | Sim time: {sim_time}s | Train time: {train_time}s")

    elif algorithm == 'DQN':
        episode = 0
        while episode < total_episodes:
            print("----- Episode", episode + 1, "of", total_episodes)
            epsilon = 1.0 - (episode / total_episodes)
            simulation_time, training_time = Simulation.run(episode, epsilon)
            print('Simulation time:', simulation_time, 's - Training time:', training_time, 's')
            episode += 1

    end_time = datetime.datetime.now()
    print("\n----- Start time:", start_time)
    print("----- End time:", end_time)
    print("----- Session info saved at:", path)

    # Save the trained model and configuration for future reference
    # For DQN, the model is stored in Model variable, for PPO in model variable
    if algorithm == 'PPO':
        model.save_model(path)
    elif algorithm == 'DQN':
        Model.save_model(path)
    copyfile(src="training_settings.ini", dst=os.path.join(path, "training_settings.ini"))

    # Visualization: For PPO, visualize episode rewards; for DQN, use existing stats.
    viz = Visualization(path, dpi=96)
    if algorithm == 'PPO':
        viz.save_data_and_plot(data=Simulation.episode_rewards, filename="ppo_reward",
                               xlabel="Episode", ylabel="Reward")
    else:
        viz.save_data_and_plot(data=Simulation.reward_store, filename="reward",
                               xlabel="Episode", ylabel="Cumulative negative reward")
        viz.save_data_and_plot(data=Simulation.cumulative_wait_store, filename="delay",
                               xlabel="Episode", ylabel="Cumulative delay (s)")
        viz.save_data_and_plot(data=Simulation.avg_queue_length_store, filename="queue",
                               xlabel="Episode", ylabel="Average queue length (vehicles)")
