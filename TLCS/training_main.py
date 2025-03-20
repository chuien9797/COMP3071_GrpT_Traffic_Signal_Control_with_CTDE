from __future__ import absolute_import
from __future__ import print_function

import sys
import os

# Append the parent directory so that 'TLCS' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import datetime
from shutil import copyfile

from rl_models.ppo_model import PPOModel
from training_simulation import Simulation
from generator import TrafficGenerator
from memory import Memory
from visualization import Visualization
from utils import import_train_configuration, set_sumo, set_train_path

if __name__ == "__main__":

    # Import configuration and set up simulation paths
    config = import_train_configuration(config_file='training_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    import tensorflow as tf

    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    # Choose the algorithm based on the configuration
    algorithm = config.get('algorithm', 'PPO')

    if algorithm == 'PPO':
        # Try to get the PPO-specific configuration; if missing, use defaults.
        ppo_config = config.get('ppo', {})
        Model = PPOModel(
            input_dim=int(config['num_states']),
            output_dim=int(config['num_actions']),
            hidden_size=int(ppo_config.get('hidden_size', 64)),
            learning_rate=float(ppo_config.get('learning_rate', 0.001)),
            clip_ratio=float(ppo_config.get('clip_ratio', 0.2)),
            update_epochs=int(ppo_config.get('update_epochs', 10))
        )
    elif algorithm == 'DQN':
        from model import TrainModel

        Model = TrainModel(
            int(config['num_layers']),
            int(config['width_layers']),
            int(config['batch_size']),
            float(config['learning_rate']),
            input_dim=int(config['num_states']),
            output_dim=int(config['num_actions'])
        )
    else:
        raise ValueError("Unsupported algorithm: {}. Please choose either 'PPO' or 'DQN'.".format(algorithm))

    # Initialize memory, traffic generator, visualization, and simulation components
    Memory = Memory(
        int(config['memory_size_max']),
        int(config['memory_size_min'])
    )

    TrafficGen = TrafficGenerator(
        int(config['max_steps']),
        int(config['n_cars_generated'])
    )

    Visualization = Visualization(
        path,
        dpi=96
    )

    Simulation = Simulation(
        Model,
        Memory,
        TrafficGen,
        sumo_cmd,
        float(config['gamma']),
        int(config['max_steps']),
        int(config['green_duration']),
        int(config['yellow_duration']),
        int(config['num_states']),
        int(config['num_actions']),
        int(config['training_epochs'])
    )

    episode = 0
    timestamp_start = datetime.datetime.now()

    # Main training loop
    while episode < int(config['total_episodes']):
        print('\n----- Episode', str(episode + 1), 'of', str(config['total_episodes']))
        if algorithm == 'PPO':
            # For PPO, epsilon-greedy is not typically used
            epsilon = 0.0
        elif algorithm == 'DQN':
            # For DQN, use an epsilon-greedy approach that decays over episodes
            epsilon = 1.0 - (episode / int(config['total_episodes']))

        simulation_time, training_time = Simulation.run(episode, epsilon)
        print('Simulation time:', simulation_time, 's - Training time:', training_time, 's - Total:',
              round(simulation_time + training_time, 1), 's')
        episode += 1

    print("\n----- Start time:", timestamp_start)
    print("----- End time:", datetime.datetime.now())
    print("----- Session info saved at:", path)

    # Save the trained model and configuration for future reference
    Model.save_model(path)
    copyfile(src='training_settings.ini', dst=os.path.join(path, 'training_settings.ini'))

    # Generate and save performance plots
    Visualization.save_data_and_plot(data=Simulation.reward_store, filename='reward', xlabel='Episode',
                                     ylabel='Cumulative negative reward')
    Visualization.save_data_and_plot(data=Simulation.cumulative_wait_store, filename='delay', xlabel='Episode',
                                     ylabel='Cumulative delay (s)')
    Visualization.save_data_and_plot(data=Simulation.avg_queue_length_store, filename='queue', xlabel='Episode',
                                     ylabel='Average queue length (vehicles)')
