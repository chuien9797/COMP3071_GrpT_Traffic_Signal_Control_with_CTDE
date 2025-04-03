import os
import sys
import random
import datetime
import numpy as np
import tensorflow as tf

# Your existing imports:
from utils import import_train_configuration, set_sumo, set_train_path
from intersection_config import INTERSECTION_CONFIGS
from generator import TrafficGenerator
from training_simulation import Simulation
from memory import Memory
from visualization import Visualization

# 1) Instead of your old "TrainModel" or "build_dynamic_model",
#    import our new aggregator class:
from model import TrainModelAggregator


def main():
    """
    Multi-environment DQN training loop using a per-lane embedding + aggregator
    approach. We do not rely on a single huge input_dim. Instead, each environment
    returns (num_lanes, lane_feature_dim) states that are aggregated inside the model.
    """

    # 1) Load config from .ini
    config = import_train_configuration("training_settings.ini")
    algorithm = config.get('algorithm', 'DQN')
    if algorithm != 'DQN':
        raise ValueError("This script is for DQN. Found algorithm=%s" % algorithm)

    # The intersection types you want to train on in one run:
    possible_envs = ["cross", "roundabout", "T_intersection"]
    # e.g. add "Y_intersection" if you like

    # 2) We only need the largest *action* space across those envs,
    #    because aggregator doesn't require a single 'num_states'.
    #    We'll pick the max # of actions we see among possible_envs.
    max_num_actions = 0
    for env_name in possible_envs:
        env_conf = INTERSECTION_CONFIGS[env_name]
        # The 'phase_mapping' typically has len() = # possible actions
        # Or you might read from 'num_actions' if stored in config
        nA = len(env_conf["phase_mapping"])
        if nA > max_num_actions:
            max_num_actions = nA

    # We'll store that in the config so our aggregator knows how many Q-values to produce
    config['num_actions'] = max_num_actions

    # 3) Build our aggregator model
    # Typically, lane_feature_dim is how many features you store PER lane.
    # For example, you might store 3 or 5 features per lane. We'll read from .ini if you want:
    # Or you can just hard-code a guess like 5
    lane_feature_dim = 5  # e.g. occupancy, waiting, emergency_flag, etc.
    aggregator_embedding_dim = 32  # for lane embeddings
    aggregator_final_hidden = 64   # final MLP hidden size

    Model = TrainModelAggregator(
        lane_feature_dim = lane_feature_dim,
        embedding_dim    = aggregator_embedding_dim,
        final_hidden     = aggregator_final_hidden,
        num_actions      = max_num_actions,
        batch_size       = config['batch_size'],
        learning_rate    = config['learning_rate']
    )

    TargetModel = TrainModelAggregator(
        lane_feature_dim    = lane_feature_dim,
        embedding_dim       = aggregator_embedding_dim,
        final_hidden        = aggregator_final_hidden,
        num_actions         = max_num_actions,
        batch_size          = config['batch_size'],
        learning_rate       = config['learning_rate']
    )
    TargetModel.set_weights(Model.get_weights())  # Initial sync

    # 4) Create a single replay Memory instance
    MemoryInstance = Memory(
        config['memory_size_max'],
        config['memory_size_min']
    )

    # 5) Build a Simulation() object per environment
    simulations = {}
    for env_name in possible_envs:
        # Copy config so we can tweak intersection_type
        local_conf = config.copy()
        local_conf['intersection_type'] = env_name

        # Retrieve the sumocfg_file from intersection_config
        tmp_conf = INTERSECTION_CONFIGS[env_name]
        sumocfg_file = tmp_conf.get("sumocfg_file", "cross_intersection/cross_intersection.sumocfg")
        sumo_cmd = set_sumo(local_conf['gui'], sumocfg_file, local_conf['max_steps'])

        # Traffic generator
        TrafficGen = TrafficGenerator(
            max_steps         = local_conf['max_steps'],
            n_cars_generated  = local_conf['n_cars_generated'],
            intersection_type = env_name
        )

        # Create the Simulation, referencing aggregator Model and shared Memory
        sim = Simulation(
            Model           = Model,
            TargetModel=TargetModel,
            Memory          = MemoryInstance,
            TrafficGen      = TrafficGen,
            sumo_cmd        = sumo_cmd,
            gamma           = local_conf['gamma'],
            max_steps       = local_conf['max_steps'],
            green_duration  = local_conf['green_duration'],
            yellow_duration = local_conf['yellow_duration'],
            # We can ignore num_states or set it arbitrarily:
            num_states      = 9999,  # not used by aggregator
            num_actions     = max_num_actions,  # from aggregator
            training_epochs = local_conf['training_epochs'],
            intersection_type = env_name
        )
        simulations[env_name] = sim

    # 6) Main training loop
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

    # 7) Save final aggregator model
    path = set_train_path(config['models_path_name'])
    Model.save_model(path)

    # 8) (Optional) Visualization. We'll pick "cross" as an example
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
