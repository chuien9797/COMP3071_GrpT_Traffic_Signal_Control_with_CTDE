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
    Main multi-environment training loop:
      - Loads training configuration.
      - Constructs simulation environments for different intersection types.
      - Creates a per-intersection (agent) model.
      - Runs training episodes, randomly selecting an environment for each episode.
      - After simulation, performs per-agent updates using flexible reward sharing and CTDE.
      - Saves the final model and plots training statistics.
    """
    # 1. Load configuration.
    config = import_train_configuration("training_settings.ini")
    algorithm = config.get('algorithm', 'DQN')
    if algorithm != 'DQN':
        raise ValueError("This script is for DQN. Found algorithm=%s" % algorithm)

    # 2. Determine the maximum action space over all environments.
    possible_envs = ["cross", "roundabout", "2x2_grid", "1x2_grid"]
    max_num_actions = 0
    for env_name in possible_envs:
        env_conf = INTERSECTION_CONFIGS[env_name]
        nA = len(env_conf["phase_mapping"])
        if nA > max_num_actions:
            max_num_actions = nA
    config['num_actions'] = max_num_actions

    # 3. Set fixed hyperparameters for the aggregator network.
    lane_feature_dim = 9  # Must match state computation in Simulation._get_state()
    aggregator_embedding_dim = 32  # Per-lane embedding dimension.
    aggregator_final_hidden = 64     # Final hidden layer size for the aggregator.

    # 4. Create simulation environments.
    simulations = {}
    for env_name in possible_envs:
        # Create a local copy of the configuration and set the intersection type.
        local_conf = config.copy()
        local_conf['intersection_type'] = env_name

        # Retrieve environment-specific configuration.
        env_conf = INTERSECTION_CONFIGS[env_name]
        sumocfg_file = env_conf.get("sumocfg_file", f"{env_name}/{env_name}.sumocfg")
        sumo_cmd = set_sumo(local_conf['gui'], sumocfg_file, local_conf['max_steps'])

        # Create the TrafficGenerator for this environment.
        TrafficGen = TrafficGenerator(
            max_steps=local_conf['max_steps'],
            n_cars_generated=local_conf['n_cars_generated'],
            intersection_type=env_name
        )

        # Determine the number of intersections (agents) based on traffic_light_ids.
        if "traffic_light_ids" in env_conf:
            if isinstance(env_conf["traffic_light_ids"], list):
                num_intersections = len(env_conf["traffic_light_ids"])
            else:
                num_intersections = 1
        else:
            num_intersections = 1
        print(f"Environment '{env_name}' has {num_intersections} intersection(s).")

        # Create a list of agent models.
        agents = []
        for i in range(num_intersections):
            agent = TrainModelAggregator(
                lane_feature_dim=lane_feature_dim,
                embedding_dim=aggregator_embedding_dim,
                final_hidden=aggregator_final_hidden,
                num_actions=max_num_actions,
                batch_size=config['batch_size'],
                learning_rate=config['learning_rate']
            )
            agents.append(agent)

        # Create a shared replay memory instance.
        memory_instance = Memory(
            config['memory_size_max'],
            config['memory_size_min']
        )

        # Create the Simulation instance (with all modifications integrated).
        sim = Simulation(
            Models=agents,                # List of models (one per agent)
            TargetModels=agents,          # Target models (here, using the same list)
            Memory=memory_instance,
            TrafficGen=TrafficGen,
            sumo_cmd=sumo_cmd,
            gamma=local_conf['gamma'],
            max_steps=local_conf['max_steps'],
            green_duration=local_conf['green_duration'],
            yellow_duration=local_conf['yellow_duration'],
            num_states=9999,              # Arbitrary (not used by the aggregator)
            training_epochs=local_conf['training_epochs'],
            intersection_type=env_name,
            signal_fault_prob=local_conf.get('signal_fault_prob', 0.1)
        )
        simulations[env_name] = sim

    # 5. Main training loop.
    total_episodes = config['total_episodes']
    start_time = datetime.datetime.now()
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    combined_rewards = []
    for ep in range(total_episodes):
        # Randomly select one environment per episode.
        chosen_env = random.choice(possible_envs)
        sim = simulations[chosen_env]
        print(f"----- Episode {ep + 1}/{total_episodes} on environment '{chosen_env}' -----")
        # Linear decay of epsilon.
        epsilon = 1.0 - (ep / total_episodes)
        sim_time, train_time = sim.run(episode=ep, epsilon=epsilon)
        print(f"Episode {ep + 1} done | env='{chosen_env}' | sim time = {sim_time}s | train time = {train_time}s\n")
        combined_rewards.append(sim.reward_store[-1])

    end_time = datetime.datetime.now()
    print("\n----- Start time:", start_time)
    print("----- End time:", end_time)

    # 6. Save final model.
    model_save_path = set_train_path(config['models_path_name'])
    chosen_env = possible_envs[0]
    # Save the first agent from the chosen environment.
    agent = simulations[chosen_env]._Models[0]
    agent_save_path = os.path.join(model_save_path, f"{chosen_env}_agent_1")
    agent.save_model(agent_save_path)
    print(f"Model for {chosen_env} agent 1 saved at: {agent_save_path}")

    # 7. Save training plots.
    viz = Visualization(model_save_path, dpi=96)
    for env_name, sim in simulations.items():
        viz.save_data_and_plot(data=sim.reward_store, filename=f"{env_name}_reward",
                               xlabel="Episode", ylabel="Cumulative negative reward")
        viz.save_data_and_plot(data=sim.cumulative_wait_store, filename=f"{env_name}_delay",
                               xlabel="Episode", ylabel="Cumulative delay (s)")
        viz.save_data_and_plot(data=sim.avg_queue_length_store, filename=f"{env_name}_queue",
                               xlabel="Episode", ylabel="Avg queue length (vehicles)")
    viz.save_data_and_plot(data=combined_rewards, filename="reward_all_envs",
                           xlabel="Episode", ylabel="Reward (all environments)")

    print("All done! Model and plots saved at:", model_save_path)

if __name__ == "__main__":
    main()
