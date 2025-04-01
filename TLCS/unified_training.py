##############################################################################
# Filename: unified_training.py (or training_main_multi_env.py)
# Purpose:  Interleave multiple intersection types in a single DQN training run
#           using *one* model, each with its own SUMO .sumocfg file.
##############################################################################

import os
import sys
import random
import datetime
import numpy as np
import tensorflow as tf

# Your existing imports
from utils import import_train_configuration, set_sumo, set_train_path
from environment_utils import compute_environment_parameters, build_dynamic_model
from intersection_config import INTERSECTION_CONFIGS  # We read sumocfg_file from here
from generator import TrafficGenerator
from model import TrainModel
from training_simulation import Simulation
from memory import Memory
from visualization import Visualization

def main():
    """
    Multi-environment DQN training loop. Trains one DQN model on all
    intersection types by randomly picking an environment each episode,
    now using each environment's custom .sumocfg file.
    """

    # 1) Load your training config from .ini
    config = import_train_configuration("training_settings.ini")
    algorithm = config.get('algorithm', 'DQN')
    if algorithm != 'DQN':
        raise ValueError("This script is for DQN. Found algorithm=%s" % algorithm)

    # The intersection types you want to train on:
    possible_envs = ["cross", "roundabout", "T_intersection"]
    # (Add "Y_intersection" if you want, etc.)

    # 2) Find largest state/action spaces across those envs
    max_num_states = 0
    max_num_actions = 0
    for env_name in possible_envs:
        env_conf = INTERSECTION_CONFIGS[env_name]
        nS, nA = compute_environment_parameters(env_conf)
        nS += 9  # your existing offset
        if nS > max_num_states:
            max_num_states = nS
        if nA > max_num_actions:
            max_num_actions = nA

    config['num_states'] = max_num_states
    config['num_actions'] = max_num_actions

    # 3) Build a single DQN model big enough for all
    hidden_layers = config.get('dqn_hidden_layers', [64, 64])  # or from .ini
    dqn_model = build_dynamic_model(input_dim=max_num_states,
                                    output_dim=max_num_actions,
                                    hidden_layers=hidden_layers)

    Model = TrainModel(
        num_layers    = config['num_layers'],
        width         = config['width_layers'],
        batch_size    = config['batch_size'],
        learning_rate = config['learning_rate'],
        input_dim     = max_num_states,
        output_dim    = max_num_actions,
        model         = dqn_model
    )

    # One global replay memory
    MemoryInstance = Memory(config['memory_size_max'],
                            config['memory_size_min'])

    # 4) Build a Simulation() object per environment
    simulations = {}
    for env_name in possible_envs:
        local_conf = config.copy()
        local_conf['intersection_type'] = env_name
        # Recompute environment parameters for this env
        tmp_conf = INTERSECTION_CONFIGS[env_name]
        nS, nA = compute_environment_parameters(tmp_conf)
        nS += 9
        local_conf['num_states'] = nS
        local_conf['num_actions'] = nA

        # Pull the sumocfg_file from the intersection_config
        sumocfg_file = tmp_conf.get("sumocfg_file", "cross_intersection/cross_intersection.sumocfg")
        sumo_cmd = set_sumo(local_conf['gui'], sumocfg_file, local_conf['max_steps'])

        # The traffic generator
        TrafficGen = TrafficGenerator(
            max_steps         = local_conf['max_steps'],
            n_cars_generated  = local_conf['n_cars_generated'],
            intersection_type = env_name
        )

        # Create the DQN Simulation
        sim = Simulation(
            Model           = Model,
            Memory          = MemoryInstance,
            TrafficGen      = TrafficGen,
            sumo_cmd        = sumo_cmd,
            gamma           = local_conf['gamma'],
            max_steps       = local_conf['max_steps'],
            green_duration  = local_conf['green_duration'],
            yellow_duration = local_conf['yellow_duration'],
            num_states      = local_conf['num_states'],
            num_actions     = local_conf['num_actions'],
            training_epochs = local_conf['training_epochs'],
            intersection_type = env_name
        )
        simulations[env_name] = sim

    # 5) Main training loop
    total_episodes = config['total_episodes']
    start_time = datetime.datetime.now()
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    for ep in range(total_episodes):
        # pick environment type randomly
        chosen_env = random.choice(possible_envs)
        sim = simulations[chosen_env]

        print(f"----- Episode {ep + 1} of {total_episodes} on environment '{chosen_env}' -----")
        epsilon = 1.0 - (ep / total_episodes)  # naive linear decay
        sim_time, train_time = sim.run(episode=ep, epsilon=epsilon)
        print(f"Episode {ep + 1} done | env={chosen_env} | sim time={sim_time}s | train time={train_time}s\n")

    end_time = datetime.datetime.now()
    print("\n----- Start time:", start_time)
    print("----- End time:", end_time)

    # 6) Save the final model
    path = set_train_path(config['models_path_name'])
    Model.save_model(path)

    # (Optional) Visualization
    cross_sim = simulations["cross"]
    viz = Visualization(path, dpi=96)
    viz.save_data_and_plot(data=cross_sim.reward_store, filename="reward",
                           xlabel="Episode", ylabel="Cumulative negative reward")
    viz.save_data_and_plot(data=cross_sim.cumulative_wait_store, filename="delay",
                           xlabel="Episode", ylabel="Cumulative delay (s)")
    viz.save_data_and_plot(data=cross_sim.avg_queue_length_store, filename="queue",
                           xlabel="Episode", ylabel="Avg queue length (vehicles)")

    print("All done! Model + plots saved at:", path)


if __name__ == "__main__":
    main()
