from __future__ import absolute_import
from __future__ import print_function

import sys
import os
import datetime
from shutil import copyfile

# Append the parent directory so that 'TLCS' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import import_train_configuration, set_sumo, set_train_path

# Import PPO classes from our new merged file
from ppo_training_loop import PPOModel, PPOSimulation
# If DQN is used, keep the old simulation (and memory) imports:
# from model import TrainModel
# from training_simulation import Simulation
# from memory import Memory

from generator import TrafficGenerator
from visualization import Visualization

if __name__ == "__main__":
    # Load configuration and set up simulation paths
    config = import_train_configuration("training_settings.ini")
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    path = set_train_path(config['models_path_name'])

    import tensorflow as tf
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    # Choose the algorithm based on the configuration
    algorithm = config.get('algorithm', 'PPO')

    if algorithm == 'PPO':
        model = PPOModel(
            input_dim=config['num_states'],
            output_dim=config['num_actions'],
            hidden_size=config['ppo_hidden_size'],
            learning_rate=config['ppo_learning_rate'],
            clip_ratio=config['ppo_clip_ratio'],
            update_epochs=config['ppo_update_epochs']
        )
        traffic_gen = TrafficGenerator(config['max_steps'], config['n_cars_generated'])
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
            training_epochs=config['ppo_training_epochs']
        )

    elif algorithm == 'DQN':
        # Instantiate DQN model and simulation as before.
        from model import TrainModel
        Model = TrainModel(
            int(config['num_layers']),
            int(config['width_layers']),
            int(config['batch_size']),
            float(config['learning_rate']),
            input_dim=int(config['num_states']),
            output_dim=int(config['num_actions'])
        )
        from training_simulation import Simulation
        from memory import Memory
        MemoryInstance = Memory(
            int(config['memory_size_max']),
            int(config['memory_size_min'])
        )
        TrafficGen = TrafficGenerator(int(config['max_steps']), int(config['n_cars_generated']))
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
            int(config['training_epochs'])
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
    model.save_model(path)
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
