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
    Main multi-environment training loop with a single shared policy model:
      - Loads training configuration.
      - Determines action space and hyperparameters.
      - Instantiates one shared TrainModelAggregator for all agents.
      - Constructs Simulation instances for each intersection type, each
        referring to the same shared model.
      - Runs training episodes, randomly selecting an environment per episode.
      - Saves the shared model and training plots.
    """
    # 1. Load configuration.
    config = import_train_configuration("training_settings.ini")
    algorithm = config.get('algorithm', 'DQN')
    if algorithm != 'DQN':
        raise ValueError(f"This script is for DQN. Found algorithm={algorithm}")

    # 2. Determine maximum action space across all environments.
    possible_envs = ["cross", "roundabout", "1x2_grid"]
    max_num_actions = max(len(INTERSECTION_CONFIGS[env]["phase_mapping"]) for env in possible_envs)
    config['num_actions'] = max_num_actions

    # 3. Set fixed hyperparameters for the aggregator network.
    lane_feature_dim        = 9    # Must match Simulation._get_state()
    aggregator_embedding_dim = 32
    aggregator_final_hidden  = 64

    # 4. Create one shared policy model for all agents.
    shared_model = TrainModelAggregator(
        lane_feature_dim=lane_feature_dim,
        embedding_dim=aggregator_embedding_dim,
        final_hidden=aggregator_final_hidden,
        num_actions=max_num_actions,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )

    # 5. Build Simulation instances for each environment, all using shared_model.
    simulations = {}
    for env_name in possible_envs:
        local_conf = config.copy()
        local_conf['intersection_type'] = env_name

        env_conf = INTERSECTION_CONFIGS[env_name]
        sumocfg_file = env_conf.get("sumocfg_file", f"{env_name}/{env_name}.sumocfg")
        sumo_cmd = set_sumo(local_conf['gui'], sumocfg_file, local_conf['max_steps'])

        TrafficGen = TrafficGenerator(
            max_steps=local_conf['max_steps'],
            n_cars_generated=local_conf['n_cars_generated'],
            intersection_type=env_name
        )

        # number of agents = number of traffic lights in this env
        if isinstance(env_conf.get("traffic_light_ids", []), list):
            num_intersections = len(env_conf["traffic_light_ids"])
        else:
            num_intersections = 1
        print(f"Environment '{env_name}' has {num_intersections} intersection(s).")

        # all agents share the same policy model
        agents = [shared_model] * num_intersections
        # target models (for DQN target network) also shared
        target_agents = [shared_model] * num_intersections

        memory_instance = Memory(
            config['memory_size_max'],
            config['memory_size_min']
        )

        sim = Simulation(
            Models=agents,
            TargetModels=target_agents,
            Memory=memory_instance,
            TrafficGen=TrafficGen,
            sumo_cmd=sumo_cmd,
            gamma=local_conf['gamma'],
            max_steps=local_conf['max_steps'],
            green_duration=local_conf['green_duration'],
            yellow_duration=local_conf['yellow_duration'],
            num_states=9999,  # unused by aggregator
            training_epochs=local_conf['training_epochs'],
            intersection_type=env_name,
            signal_fault_prob=local_conf.get('signal_fault_prob', 0.1)
        )
        simulations[env_name] = sim

    # 6. Training loop.
    total_episodes = config['total_episodes']
    start_time = datetime.datetime.now()
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    combined_rewards = []
    for ep in range(total_episodes):
        chosen_env = random.choice(possible_envs)
        sim = simulations[chosen_env]
        print(f"----- Episode {ep+1}/{total_episodes} on '{chosen_env}' -----")
        epsilon = 1.0 - (ep / total_episodes)
        sim_time, train_time = sim.run(episode=ep, epsilon=epsilon)
        print(f"Episode {ep+1} done | sim_time={sim_time}s | train_time={train_time}s\n")
        combined_rewards.append(sim.reward_store[-1])

    end_time = datetime.datetime.now()
    print(f"\n----- Start: {start_time}  End: {end_time}")

    # 7. Save the shared model.
    model_save_path = set_train_path(config['models_path_name'])
    shared_model.save_model(os.path.join(model_save_path, "shared_policy"))
    print(f"Shared policy model saved at: {model_save_path}/shared_policy")

    # 8. Save training plots.
    viz = Visualization(model_save_path, dpi=96)
    for env_name, sim in simulations.items():
        viz.save_data_and_plot(sim.reward_store,    filename=f"{env_name}_reward",
                               xlabel="Episode", ylabel="Cumulative negative reward")
        viz.save_data_and_plot(sim.cumulative_wait_store, filename=f"{env_name}_delay",
                               xlabel="Episode", ylabel="Cumulative delay (s)")
        viz.save_data_and_plot(sim.avg_queue_length_store, filename=f"{env_name}_queue",
                               xlabel="Episode", ylabel="Avg queue length")
    viz.save_data_and_plot(combined_rewards, filename="reward_all_envs",
                           xlabel="Episode", ylabel="Reward (all envs)")

    print("All done! Models and plots saved at:", model_save_path)


if __name__ == "__main__":
    main()
