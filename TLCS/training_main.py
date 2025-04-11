import os
import sys
import random
import datetime
import numpy as np
import tensorflow as tf

# Import the PPO model wrapper.
from TLCS.rl_models.ppo_model import TrainModelPPO
# Your existing imports:
from utils import import_train_configuration, set_sumo, set_train_path
from intersection_config import INTERSECTION_CONFIGS
from generator import TrafficGenerator
from training_simulation import Simulation
from memory import Memory
from visualization import Visualization

# Instead of only DQN, import both DQN and PPO model wrappers.
from model import TrainModelAggregator

def main():
    """
    Multi-environment training loop using a per-lane embedding + aggregator approach.
    This version supports both DQN and a refined adaptive PPO algorithm.
    """
    # 1) Load configuration from .ini file.
    config = import_train_configuration("training_settings.ini")
    algorithm = config.get('algorithm', 'PPO').upper()

    # The intersection types you want to train on in one run:
    possible_envs = ["cross", "roundabout", "T_intersection"]

    # 2) Determine maximum number of actions across environments.
    max_num_actions = 0
    for env_name in possible_envs:
        env_conf = INTERSECTION_CONFIGS[env_name]
        nA = len(env_conf["phase_mapping"])
        if nA > max_num_actions:
            max_num_actions = nA
    config['num_actions'] = max_num_actions

    # 3) Build our model(s)
    if algorithm == "DQN":
        lane_feature_dim = 5      # used for DQN
        aggregator_embedding_dim = 32
        aggregator_final_hidden = 64

        Model = TrainModelAggregator(
            lane_feature_dim=lane_feature_dim,
            embedding_dim=aggregator_embedding_dim,
            final_hidden=aggregator_final_hidden,
            num_actions=max_num_actions,
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )
        TargetModel = TrainModelAggregator(
            lane_feature_dim=lane_feature_dim,
            embedding_dim=aggregator_embedding_dim,
            final_hidden=aggregator_final_hidden,
            num_actions=max_num_actions,
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )
        TargetModel.set_weights(Model.get_weights())  # Initial synchronization

    elif algorithm == "PPO":
        # Retrieve PPO parameters with default fallbacks.
        ppo_hidden_size = config.get('ppo_hidden_size', 64)
        ppo_learning_rate = config.get('ppo_learning_rate', 0.0003)
        if ppo_learning_rate is None:
            ppo_learning_rate = 0.0003
        ppo_clip_ratio = config.get('ppo_clip_ratio', 0.2)
        ppo_update_epochs = config.get('ppo_update_epochs', 10)
        ppo_training_epochs = config.get('ppo_training_epochs', 10)

        # For PPO, we use lane_feature_dim = 9 (to match _get_state()),
        # and the model is adaptive so we don't need to force a fixed padding at build-time.
        Model = TrainModelPPO(
            lane_feature_dim=9,  # Should match the state feature dimension from _get_state()
            hidden_size=ppo_hidden_size,
            learning_rate=ppo_learning_rate,
            clip_ratio=ppo_clip_ratio,
            update_epochs=ppo_update_epochs,
            training_epochs=ppo_training_epochs,
            num_actions=max_num_actions,
            use_priority = True,
            reward_scale = 2.0
        )
        # NOTE: Remove the explicit Model.build(...) call; the model now adapts to the input shape.
        TargetModel = None  # PPO does not need a separate target network.

    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # 4) Create a replay Memory instance (only needed for DQN).
    MemoryInstance = None
    if algorithm == "DQN":
        MemoryInstance = Memory(
            config['memory_size_max'],
            config['memory_size_min']
        )

    # 5) Build a Simulation object per environment.
    simulations = {}
    for env_name in possible_envs:
        local_conf = config.copy()
        local_conf['intersection_type'] = env_name
        tmp_conf = INTERSECTION_CONFIGS[env_name]
        sumocfg_file = tmp_conf.get("sumocfg_file", "cross_intersection/cross_intersection.sumocfg")
        sumo_cmd = set_sumo(local_conf['gui'], sumocfg_file, local_conf['max_steps'])
        TrafficGen = TrafficGenerator(
            max_steps=local_conf['max_steps'],
            n_cars_generated=local_conf['n_cars_generated'],
            intersection_type=env_name
        )
        sim = Simulation(
            Model=Model,
            TargetModel=TargetModel,
            Memory=MemoryInstance,
            TrafficGen=TrafficGen,
            sumo_cmd=sumo_cmd,
            gamma=local_conf['gamma'],
            max_steps=local_conf['max_steps'],
            green_duration=local_conf['green_duration'],
            yellow_duration=local_conf['yellow_duration'],
            num_states=9999,  # Not used by aggregator or PPO.
            num_actions=max_num_actions,
            training_epochs=local_conf['training_epochs'],
            intersection_type=env_name,
            algorithm=algorithm
        )
        simulations[env_name] = sim

    # 6) Main training loop.
    total_episodes = config['total_episodes']
    start_time_overall = datetime.datetime.now()
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    for ep in range(total_episodes):
        chosen_env = random.choice(possible_envs)
        sim = simulations[chosen_env]
        print(f"----- Episode {ep + 1} of {total_episodes} on environment '{chosen_env}' -----")
        epsilon = 1.0 - (ep / total_episodes) if algorithm == "DQN" else None
        sim_time, train_time = sim.run(episode=ep, epsilon=epsilon)
        print(f"Episode {ep + 1} done | env={chosen_env} | sim time={sim_time}s | train time={train_time}s\n")

    end_time_overall = datetime.datetime.now()
    print("\n----- Start time:", start_time_overall)
    print("----- End time:", end_time_overall)

    # 7) Save final model.
    path = set_train_path(config['models_path_name'])
    Model.save_model(path)

    # 8) Visualization (example using the "cross" environment).
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
