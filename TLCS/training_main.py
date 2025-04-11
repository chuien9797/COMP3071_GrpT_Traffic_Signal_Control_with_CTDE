import os
import sys
import random
import datetime
import numpy as np
import tensorflow as tf

from utils import import_train_configuration, set_sumo, set_train_path
from intersection_config import INTERSECTION_CONFIGS
from generator import TrafficGenerator
from training_simulation import Simulation  # see below for PPO changes
from visualization import Visualization

# Use our new PPO model wrapper instead of TrainModelAggregator (DQN version)
from model import TrainModelPPO


def main():
    """
    Multi-environment PPO training loop using an on-policy trajectory update.
    """
    # 1) Load config from .ini (ensure [agent] algorithm is set to PPO)
    config = import_train_configuration("training_settings.ini")
    algorithm = config.get('algorithm', 'PPO')
    if algorithm != 'PPO':
        raise ValueError("This script is for PPO. Found algorithm=%s" % algorithm)

    # The intersection types you want to train on in one run:
    possible_envs = ["cross", "roundabout", "T_intersection"]
    max_num_actions = max(len(INTERSECTION_CONFIGS[env]["phase_mapping"]) for env in possible_envs)
    config['num_actions'] = max_num_actions

    # 2) Build our PPO model – note: we use PPO-specific hyperparameters from config.
    lane_feature_dim = 5  # as before (or adjust to your state design)
    ppo_hidden_size = config['ppo_hidden_size']
    final_hidden = 64  # you can adjust or read from config too

    Model = TrainModelPPO(
        lane_feature_dim=lane_feature_dim,
        embedding_dim=ppo_hidden_size,
        final_hidden=final_hidden,
        num_actions=max_num_actions,
        batch_size=config['batch_size'],
        learning_rate=config['ppo_learning_rate']
    )

    # 3) Create a Simulation() object per environment (PPO version)
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
        # Pass the PPO model (and no memory buffer) into Simulation:
        sim = Simulation(
            Model=Model,
            sumo_cmd=sumo_cmd,
            gamma=config['ppo_gamma'],  # use the PPO gamma (often 0.99)
            max_steps=local_conf['max_steps'],
            green_duration=local_conf['green_duration'],
            yellow_duration=local_conf['yellow_duration'],
            num_actions=max_num_actions,
            training_epochs=config['ppo_training_epochs'],  # number of update epochs per episode
            intersection_type=env_name,
            ppo_clip_ratio=config['ppo_clip_ratio'],
            ppo_update_epochs=config['ppo_update_epochs'],
            gae_lambda=config.get('gae_lambda', 0.95)
        )
        simulations[env_name] = sim

    total_episodes = config['total_episodes']
    start_time = datetime.datetime.now()
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    combined_rewards = []

    for ep in range(total_episodes):
        # Choose a random environment for this episode:
        chosen_env = random.choice(possible_envs)
        sim = simulations[chosen_env]
        print(f"----- Episode {ep + 1} of {total_episodes} on environment '{chosen_env}' -----")

        # In PPO, exploration is implicit through sampling from the policy
        sim_time, train_time, ep_reward = sim.run(episode=ep)
        # Print total reward after running each episode:
        print(
            f"Episode {ep + 1} done | env={chosen_env} | sim time={sim_time}s | train time={train_time}s | Total Reward: {ep_reward:.2f}\n")
        combined_rewards.append(ep_reward)

    end_time = datetime.datetime.now()
    print("\n----- Start time:", start_time)
    print("----- End time:", end_time)
    path = set_train_path(config['models_path_name'])
    Model.save_model(path)

    viz = Visualization(path, dpi=96)
    for env_name, sim in simulations.items():
        viz.save_data_and_plot(data=sim.reward_store, filename=f"{env_name}_reward",
                               xlabel="Episode", ylabel="Cumulative reward")
        viz.save_data_and_plot(data=sim.cumulative_wait_store, filename=f"{env_name}_delay",
                               xlabel="Episode", ylabel="Cumulative delay (s)")
        viz.save_data_and_plot(data=sim.avg_queue_length_store, filename=f"{env_name}_queue",
                               xlabel="Episode", ylabel="Avg queue length (vehicles)")
    viz.save_data_and_plot(data=combined_rewards, filename="reward_all_envs",
                           xlabel="Episode", ylabel="Reward (all environments)")
    print("All done! Model + plots saved at:", path)


if __name__ == "__main__":
    main()
