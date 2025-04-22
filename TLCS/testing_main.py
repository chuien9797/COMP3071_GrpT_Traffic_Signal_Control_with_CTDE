# training_main.py

import os
import sys
import random
import datetime
import numpy as np
import tensorflow as tf

from TLCS.model import TrainModelAggregator
from TLCS.testing_simulation import TestingSimulation
from utils import import_train_configuration, set_sumo, set_train_path
from intersection_config import INTERSECTION_CONFIGS
from generator import TrafficGenerator
from memory import Memory
from visualization import Visualization

def run_training_once(seed: int, config_file: str = "training_settings.ini"):
    """
    Seeds all RNGs, runs the full training loop, and returns a dict of metrics.
    """
    # ----- 0. Seed everything for repeatability -----
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # If your TrafficGenerator or SUMO uses other RNGs, seed them here

    # ----- 1. Load config and build sims -----
    config = import_train_configuration(config_file)
    possible_envs = ["cross", "roundabout", "1x2_grid"]
    max_num_actions = max(len(INTERSECTION_CONFIGS[e]["phase_mapping"]) for e in possible_envs)
    config['num_actions'] = max_num_actions

    # shared model
    shared_model = TrainModelAggregator(
        lane_feature_dim=9,
        embedding_dim=32,
        final_hidden=64,
        num_actions=max_num_actions,
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )

    simulations = {}
    for env_name in possible_envs:
        local_conf = config.copy()
        local_conf['intersection_type'] = env_name
        env_conf = INTERSECTION_CONFIGS[env_name]
        sumo_cmd = set_sumo(local_conf['gui'], env_conf["sumocfg_file"], local_conf['max_steps'])

        tg = TrafficGenerator(
            max_steps=local_conf['max_steps'],
            n_cars_generated=local_conf['n_cars_generated'],
            intersection_type=env_name
        )

        num_agents = len(env_conf.get("traffic_light_ids", [])) or 1
        agents = [shared_model] * num_agents
        target_agents = [shared_model] * num_agents
        memory = Memory(local_conf['memory_size_max'], local_conf['memory_size_min'])

        sim = TestingSimulation(
            Models=agents,
            TargetModels=target_agents,
            Memory=memory,
            TrafficGen=tg,
            sumo_cmd=sumo_cmd,
            gamma=local_conf['gamma'],
            max_steps=local_conf['max_steps'],
            green_duration=local_conf['green_duration'],
            yellow_duration=local_conf['yellow_duration'],
            num_states=9999,
            training_epochs=local_conf['training_epochs'],
            intersection_type=env_name,
            signal_fault_prob=local_conf.get('signal_fault_prob',0.1)
        )
        simulations[env_name] = sim

    # ----- 2. Training loop -----
    total_episodes = config['total_episodes']
    start_time = datetime.datetime.now()
    combined_rewards = []

    for ep in range(total_episodes):
        chosen = random.choice(possible_envs)
        sim = simulations[chosen]
        epsilon = 1.0 - (ep / total_episodes)
        sim_time, train_time = sim.run(episode=ep, epsilon=epsilon)
        combined_rewards.append(sim.reward_store[-1])

    end_time = datetime.datetime.now()

    # ----- 3. (Optional) save model & plots -----
    # ... your existing save_model() and Visualization code ...
    # we skip here so that run_training_once returns clean metrics

    # ----- 4. Collect metrics -----
    # Total vehicles (standard + emergency):
    std_vehicles = config['n_cars_generated'] * total_episodes
    emg_vehicles = 3 * total_episodes  # since you spawn 3 emergency per ep
    total_vehicles = std_vehicles + emg_vehicles

    # Sum waiting times & queue lengths across all envs
    total_wait = sum(sim._sum_waiting_time for sim in simulations.values())
    avg_queue_per_step = np.mean([
        np.mean(sim._avg_queue_length_store)
        for sim in simulations.values()
    ])

    # Throughput = vehicles / total simulation wall‑clock
    total_sim_seconds = (end_time - start_time).total_seconds()
    throughput = total_vehicles / total_sim_seconds

    # Cumulative reward = sum over all episodes (across all envs)
    cumulative_reward = sum(combined_rewards)

    # Emergency clearance time = total emergency delay ÷ count
    # (you track _emergency_total_delay and _emergency_crossed per sim)
    total_emg_delay = sum(sim._emergency_total_delay for sim in simulations.values())
    total_emg_cross = sum(sim._emergency_crossed for sim in simulations.values())
    emergency_clearance = total_emg_delay / total_emg_cross if total_emg_cross else float('nan')

    # Fault recovery times: **you’ll need** to instrument Simulation to record
    #   a list sim.fault_recovery_times = [recover_step - inject_step, …]
    all_recovery = []
    for sim in simulations.values():
        all_recovery.extend(getattr(sim, 'fault_recovery_times', []))
    fault_recovery = float(np.mean(all_recovery)) if all_recovery else float('nan')

    return {
      'delay_per_vehicle': total_wait / total_vehicles,
      'avg_queue_length': avg_queue_per_step,
      'throughput': throughput,
      'cumulative_reward': cumulative_reward,
      'emergency_clearance_time': emergency_clearance,
      'fault_recovery_time': fault_recovery,
    }


if __name__ == "__main__":
    # keep your existing CLI behavior if you like,
    # or just run one seed=0 for quick sanity:
    metrics = run_training_once(seed=0)
    print("Metrics:", metrics)
