# test_seed_generalization.py

import os
import sys
import numpy as np

from utils import import_train_configuration, set_sumo
from intersection_config import INTERSECTION_CONFIGS
from generator import TrafficGenerator
from training_simulation import Simulation
from model import TrainModelAggregator
from memory import Memory

def make_cross_sim(model, config):
    """
    Build a Simulation instance for the 'cross' topology,
    with training disabled and no signal faults.
    """
    local_conf = config.copy()
    local_conf['intersection_type'] = 'cross'

    sumocfg       = INTERSECTION_CONFIGS['cross']["sumocfg_file"]
    base_sumo_cmd = set_sumo(local_conf['gui'], sumocfg, local_conf['max_steps'])

    traffic_gen = TrafficGenerator(
        max_steps=local_conf['max_steps'],
        n_cars_generated=local_conf['n_cars_generated'],
        intersection_type='cross'
    )

    sim = Simulation(
        Models=[model] * len(INTERSECTION_CONFIGS['cross']["traffic_light_ids"]),
        TargetModels=[model] * len(INTERSECTION_CONFIGS['cross']["traffic_light_ids"]),
        Memory=Memory(
            size_max=local_conf['memory_size_max'],
            size_min=local_conf['memory_size_min']
        ),
        TrafficGen=traffic_gen,
        sumo_cmd=base_sumo_cmd,
        gamma=local_conf['gamma'],
        max_steps=local_conf['max_steps'],
        green_duration=local_conf['green_duration'],
        yellow_duration=local_conf['yellow_duration'],
        num_states=local_conf['num_states'],
        training_epochs=0,           # no internal training
        intersection_type='cross',
        signal_fault_prob=0.0        # disable faults
    )
    return sim

def test_seed_generalization():
    # 1. Load training config
    config = import_train_configuration("training_settings.ini")

    # 2. Prepare model skeleton
    model = TrainModelAggregator(
        lane_feature_dim=9,
        embedding_dim=32,
        final_hidden=64,
        num_actions=config['num_actions'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )

    # 3. Build & load weights
    max_lanes = max(
        len(lanes)
        for env in INTERSECTION_CONFIGS
        for lanes in INTERSECTION_CONFIGS[env]["incoming_lanes"].values()
    )
    _ = model.model(np.zeros((1, max_lanes, 9), dtype=np.float32))

    model_h5 = os.path.join(
        os.getcwd(),
        config['models_path_name'],  # e.g. "models"
        "model_338",
        "shared_policy",
        "trained_model.h5"
    )
    print(f"🔍 Loading weights from: {model_h5}")
    model.model.load_weights(model_h5)
    print("✅ Weights loaded.\n")
    model.model.summary()

    # 4. Create a single Simulation for 'cross'
    sim = make_cross_sim(model, config)

    # 5. Loop over seeds 200–249
    stats = {'delay': [], 'queue': [], 'throughput': [], 'reward': []}
    for seed in range(200, 250):
        print(f"\n🚦 Evaluating seed {seed}...")
        # regenerate routes with this seed
        sim._TrafficGen.generate_routefile(seed=seed)
        # run with deterministic policy, no training
        sim.run(episode=seed, epsilon=0.0, train=False)

        # collect metrics
        avg_delay   = sim._sum_waiting_time / config['n_cars_generated']
        avg_queue   = sim._sum_queue_length  / config['max_steps']
        throughput  = config['n_cars_generated']
        cum_reward  = sim._sum_neg_reward

        stats['delay'].append(avg_delay)
        stats['queue'].append(avg_queue)
        stats['throughput'].append(throughput)
        stats['reward'].append(cum_reward)

    # 6. Report mean ± std
    print("\n=== Hold-Out Seed Generalization Results (200–249) ===")
    for m in ['delay','queue','throughput','reward']:
        arr = np.array(stats[m])
        print(f"{m.capitalize():<10}: {arr.mean():.2f} ± {arr.std():.2f}")

if __name__ == "__main__":
    test_seed_generalization()
