import os
import sys
import random
import datetime
import numpy as np
import tensorflow as tf

from utils import import_train_configuration, set_sumo, set_train_path
from intersection_config import INTERSECTION_CONFIGS
from generator import TrafficGenerator
from training_simulation import Simulation
from memory import Memory
from visualization import Visualization
from model import TrainModelAggregator


def main():
    """
    Multi-agent DQN training loop using a per-lane embedding + aggregator approach.
    Each agent (for a given intersection) is instantiated separately with its own memory,
    and a corresponding Simulation instance is built.
    """

    # 1) Load configuration from .ini
    config = import_train_configuration("training_settings.ini")
    algorithm = config.get('algorithm', 'DQN')
    if algorithm != 'DQN':
        raise ValueError("This script is for DQN. Found algorithm=%s" % algorithm)

    # 2) Define the intersections (environments) to be controlled.
    # Modify this list to include the intersection types you want for your multi-agent system.
    multi_envs = ["T_intersection", "roundabout"]

    # 3) Determine the maximum action space across chosen environments.
    max_num_actions = 0
    for env_name in multi_envs:
        env_conf = INTERSECTION_CONFIGS[env_name]
        nA = len(env_conf["phase_mapping"])
        if nA > max_num_actions:
            max_num_actions = nA
    config['num_actions'] = max_num_actions

    # 4) Create separate agent & memory instances and build a Simulation object per environment.
    agents = {}       # dictionary to hold agent models per environment
    memories = {}     # dictionary to hold replay Memory per environment
    simulations = {}  # dictionary to hold Simulation objects

    # Define fixed hyperparameters for the aggregator network
    lane_feature_dim = 5           # e.g. occupancy, waiting, emergency_flag, etc.
    aggregator_embedding_dim = 32  # dimension for lane embeddings
    aggregator_final_hidden = 64   # size of final hidden layer

    for env_name in multi_envs:
        # Build an aggregator model for this environment
        agent = TrainModelAggregator(
            lane_feature_dim=lane_feature_dim,
            embedding_dim=aggregator_embedding_dim,
            final_hidden=aggregator_final_hidden,
            num_actions=max_num_actions,  # using the overall maximum number of actions
            batch_size=config['batch_size'],
            learning_rate=config['learning_rate']
        )
        agents[env_name] = agent

        # Create an independent replay Memory for this agent
        mem = Memory(config['memory_size_max'], config['memory_size_min'])
        memories[env_name] = mem

        # Prepare a copy of configuration for this environment and determine the SUMO configuration file
        local_conf = config.copy()
        local_conf['intersection_type'] = env_name
        tmp_conf = INTERSECTION_CONFIGS[env_name]
        sumocfg_file = tmp_conf.get("sumocfg_file", f"{env_name}/{env_name}.sumocfg")
        sumo_cmd = set_sumo(local_conf['gui'], sumocfg_file, local_conf['max_steps'])

        # Create TrafficGenerator for this environment
        TrafficGen = TrafficGenerator(
            max_steps=local_conf['max_steps'],
            n_cars_generated=local_conf['n_cars_generated'],
            intersection_type=env_name
        )

        # Build the Simulation object; here we use the agent for both the Model and the TargetModel
        sim = Simulation(
            Model=agent,
            TargetModel=agent,  # you can create a separate target if desired
            Memory=mem,
            TrafficGen=TrafficGen,
            sumo_cmd=sumo_cmd,
            gamma=local_conf['gamma'],
            max_steps=local_conf['max_steps'],
            green_duration=local_conf['green_duration'],
            yellow_duration=local_conf['yellow_duration'],
            num_states=9999,  # this parameter is not used by the aggregator
            training_epochs=local_conf['training_epochs'],
            intersection_type=env_name
        )
        simulations[env_name] = sim

    # 5) Main training loop for all agents
    total_episodes = config['total_episodes']
    start_time = datetime.datetime.now()
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    # To track per-agent rewards
    all_rewards = {env: [] for env in multi_envs}

    for ep in range(total_episodes):
        print(f"\n===== Episode {ep + 1}/{total_episodes} =====")
        # Use a naive linear decay for epsilon (exploration factor)
        epsilon = 1.0 - (ep / total_episodes)
        for env_name, sim in simulations.items():
            print(f"--- Running simulation for {env_name} with epsilon = {epsilon:.2f} ---")
            sim_time, train_time = sim.run(episode=ep, epsilon=epsilon)
            print(f"Episode {ep + 1} done | env = {env_name} | sim time = {sim_time}s | train time = {train_time}s\n")
            # Save the final episode reward for this agent
            all_rewards[env_name].append(sim.reward_store[-1])

    end_time = datetime.datetime.now()
    print("\n----- Start time:", start_time)
    print("----- End time:", end_time)

    # 6) Save each agent's model and plot results
    for env_name, agent in agents.items():
        save_path = set_train_path(config['models_path_name'] + f"/{env_name}")
        agent.save_model(save_path)
        print(f"Model for {env_name} saved at: {save_path}")
        viz = Visualization(save_path, dpi=96)
        viz.save_data_and_plot(
            data=simulations[env_name].reward_store,
            filename=f"{env_name}_reward",
            xlabel="Episode",
            ylabel="Cumulative negative reward"
        )
        viz.save_data_and_plot(
            data=simulations[env_name].cumulative_wait_store,
            filename=f"{env_name}_delay",
            xlabel="Episode",
            ylabel="Cumulative delay (s)"
        )
        viz.save_data_and_plot(
            data=simulations[env_name].avg_queue_length_store,
            filename=f"{env_name}_queue",
            xlabel="Episode",
            ylabel="Avg queue length (vehicles)"
        )

    # Optional: Plot a global reward curve (concatenating rewards of all agents)
    global_rewards = []
    for env in multi_envs:
        global_rewards.extend(all_rewards[env])
    global_viz = Visualization(save_path, dpi=96)
    global_viz.save_data_and_plot(
        data=global_rewards,
        filename="reward_all_agents",
        xlabel="Episode",
        ylabel="Reward (all agents)"
    )

    print("All done! Models and plots saved at:", save_path)


if __name__ == "__main__":
    main()
