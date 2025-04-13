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
    Multi-environment DQN training loop using a per-lane embedding + aggregator approach.
    The code randomly selects one environment per episode. However, the simulation is now built
    to “sense” the environment configuration. Based on the intersection configuration, it creates
    as many agent controllers as there are intersections. For instance, if the chosen environment
    has 1 intersection then one agent is created; if 3 intersections are detected, 3 agents are created
    and they work cooperatively (by, for example, sharing parts of the state and reward calculations).
    """

    # 1) Load config from .ini
    config = import_train_configuration("training_settings.ini")
    algorithm = config.get('algorithm', 'DQN')
    if algorithm != 'DQN':
        raise ValueError("This script is for DQN. Found algorithm=%s" % algorithm)

    # The possible environment types to train on in a single run:
    possible_envs = ["cross", "roundabout", "T_intersection"]
    # (You can add more types if needed.)

    # 2) Determine the maximum action space across these environments.
    max_num_actions = 0
    for env_name in possible_envs:
        env_conf = INTERSECTION_CONFIGS[env_name]
        nA = len(env_conf["phase_mapping"])
        if nA > max_num_actions:
            max_num_actions = nA
    config['num_actions'] = max_num_actions

    # 3) Define fixed hyperparameters for the aggregator network
    lane_feature_dim = 5          # e.g. occupancy, waiting time, etc.
    aggregator_embedding_dim = 32 # dimension for per-lane embedding
    aggregator_final_hidden = 64  # final hidden layer size

    # 4) We create a Simulation instance for each environment type.
    #    Each Simulation instance will “scan” its environment configuration to determine
    #    how many intersections exist (via the 'traffic_light_ids' key in the config),
    #    then create that many agent controllers.
    simulations = {}

    for env_name in possible_envs:
        # Create a local copy of the configuration for this environment
        local_conf = config.copy()
        local_conf['intersection_type'] = env_name

        # Retrieve environment-specific configuration
        env_conf = INTERSECTION_CONFIGS[env_name]
        # Get sumocfg file name; if not present, fall back to a default name.
        sumocfg_file = env_conf.get("sumocfg_file", f"{env_name}/{env_name}.sumocfg")
        sumo_cmd = set_sumo(local_conf['gui'], sumocfg_file, local_conf['max_steps'])

        # Create a TrafficGenerator for this environment.
        TrafficGen = TrafficGenerator(
            max_steps         = local_conf['max_steps'],
            n_cars_generated  = local_conf['n_cars_generated'],
            intersection_type = env_name
        )

        # Determine number of intersections in this environment.
        # We assume the intersection configuration has a key "traffic_light_ids" which is a list.
        if "traffic_light_ids" in env_conf:
            if isinstance(env_conf["traffic_light_ids"], list):
                num_intersections = len(env_conf["traffic_light_ids"])
            else:
                num_intersections = 1
        else:
            num_intersections = 1

        print(f"Environment '{env_name}' has {num_intersections} intersection(s).")

        # Create a list of agent controllers for the environment.
        agents = []
        for i in range(num_intersections):
            agent = TrainModelAggregator(
                lane_feature_dim = lane_feature_dim,
                embedding_dim    = aggregator_embedding_dim,
                final_hidden     = aggregator_final_hidden,
                num_actions      = max_num_actions,
                batch_size       = config['batch_size'],
                learning_rate    = config['learning_rate']
            )
            agents.append(agent)

        # Create a single replay Memory instance for this environment.
        memory_instance = Memory(
            config['memory_size_max'],
            config['memory_size_min']
        )

        # Create the Simulation instance.
        # (Here we assume that the Simulation class has been updated to accept:
        #   - agents: a list of controllers,
        #   - memory: a Memory instance,
        #   - TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration,
        #   - num_states (may be ignored by the aggregator), training_epochs, and
        #   - intersection_type for scanning configuration.)
        sim = Simulation(
            agents,               # list of agent controllers (one per intersection)
            memory_instance,      # replay memory for the environment
            TrafficGen,           # traffic generator
            sumo_cmd,             # sumo command line
            gamma           = local_conf['gamma'],
            max_steps       = local_conf['max_steps'],
            green_duration  = local_conf['green_duration'],
            yellow_duration = local_conf['yellow_duration'],
            num_states      = 9999,          # not used by aggregator, so an arbitrary value
            training_epochs = local_conf['training_epochs'],
            intersection_type = env_name       # pass the type so Simulation can scan its config
        )
        simulations[env_name] = sim

    # 5) Main training loop (unchanged): randomly choose an environment per episode.
    total_episodes = config['total_episodes']
    start_time = datetime.datetime.now()
    print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
    print("Physical devices:", tf.config.list_physical_devices())

    combined_rewards = []

    for ep in range(total_episodes):
        chosen_env = random.choice(possible_envs)
        sim = simulations[chosen_env]
        print(f"----- Episode {ep + 1}/{total_episodes} on environment '{chosen_env}' -----")
        epsilon = 1.0 - (ep / total_episodes)  # linear decay of epsilon
        sim_time, train_time = sim.run(episode=ep, epsilon=epsilon)
        print(f"Episode {ep + 1} done | env='{chosen_env}' | sim time = {sim_time}s | train time = {train_time}s\n")
        combined_rewards.append(sim.reward_store[-1])

    end_time = datetime.datetime.now()
    print("\n----- Start time:", start_time)
    print("----- End time:", end_time)

    # 6) Save the final aggregator model(s).
    # In case of multiple intersections, you can choose to save a single combined model or
    # save each agent’s model separately.
    model_save_path = set_train_path(config['models_path_name'])
    # Here, for simplicity, we save the agents from the first environment.
    # You may extend this to save models for every environment if needed.
    chosen_env = possible_envs[0]
    agents = simulations[chosen_env]._agents  # assuming Simulation stores the agents list in attribute _agents
    for idx, agent in enumerate(agents):
        agent_save_path = os.path.join(model_save_path, f"{chosen_env}_agent_{idx+1}")
        agent.save_model(agent_save_path)
        print(f"Model for {chosen_env} agent {idx+1} saved at: {agent_save_path}")

    # 7) Save plots for each environment.
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
