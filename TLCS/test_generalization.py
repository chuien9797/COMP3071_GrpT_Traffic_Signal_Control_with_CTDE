import os
import sys
import numpy as np
import tensorflow as tf

from utils import import_train_configuration, set_sumo
from intersection_config import INTERSECTION_CONFIGS
from generator import TrafficGenerator
from training_simulation import Simulation
from model import TrainModelAggregator
from memory import Memory


def make_sim(env_name, model, config):
    local_conf = config.copy()
    local_conf['intersection_type'] = env_name

    sumocfg = INTERSECTION_CONFIGS[env_name]["sumocfg_file"]
    base_sumo_cmd = set_sumo(local_conf['gui'], sumocfg, local_conf['max_steps'])

    traffic_gen = TrafficGenerator(
        max_steps=local_conf['max_steps'],
        n_cars_generated=local_conf['n_cars_generated'],
        intersection_type=env_name
    )

    sim = Simulation(
        Models=[model] * len(INTERSECTION_CONFIGS[env_name]["traffic_light_ids"]),
        TargetModels=[model] * len(INTERSECTION_CONFIGS[env_name]["traffic_light_ids"]),
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
        training_epochs=0,
        intersection_type=env_name,
        signal_fault_prob=0.0
    )

    # save base command for per-episode seeding
    sim.base_sumo_cmd = list(base_sumo_cmd)
    return sim


def test_generalization():
    # 1. Load train config
    config = import_train_configuration("training_settings.ini")

    # 2. Prepare wrapper (architecture only)
    model = TrainModelAggregator(
        lane_feature_dim=9,
        embedding_dim=32,
        final_hidden=64,
        num_actions=config['num_actions'],
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )

    # 3. Hard-code path to your saved weights
    model_h5 = os.path.join(
        os.getcwd(),
        config['models_path_name'],  # "models"
        "model_338",
        "shared_policy",
        "trained_model.h5"
    )

    # 4. BUILD for real by doing one dummy pass
    lane_feat = 9
    max_lanes = max(
        len(lanes)
        for env in INTERSECTION_CONFIGS
        for lanes in INTERSECTION_CONFIGS[env]["incoming_lanes"].values()
    )
    dummy = np.zeros((1, max_lanes, lane_feat), dtype=np.float32)
    _ = model.model(dummy)  # instantiate layers & weights

    # 5. Load weights & verify
    print(f"🔍 Loading weights from: {model_h5}")
    try:
        model.model.load_weights(model_h5)
        print("✅ Weights loaded successfully!")
        print("\n--- Model summary ---")
        model.model.summary()
        print("---------------------\n")
    except Exception as e:
        print(f"❌ Failed to load weights:\n{e}")
        sys.exit(1)

    # 6. Evaluate on unseen layouts
    unseen_envs = ["double_t", "1x2_grid"]
    episodes_per_env = 30

    for env in unseen_envs:
        stats = {'delay': [], 'queue': [], 'reward': []}
        print(f"\n=== Testing on unseen topology: {env} ===")
        sim = make_sim(env, model, config)

        for ep in range(episodes_per_env):
            # introduce variability: random SUMO seed per episode
            random_seed = np.random.randint(1, 1_000_000)
            sim.sumo_cmd = sim.base_sumo_cmd + ["--seed", str(random_seed), "--random", "true"]

            # run with deterministic policy (epsilon=0)
            sim.run(episode=ep, epsilon=0.0)

            # collect stats
            stats['delay'].append(sim.cumulative_wait_store[-1] / config['n_cars_generated'])
            stats['queue'].append(sim.avg_queue_length_store[-1])
            stats['reward'].append(sim.reward_store[-1])

        # report mean ± std
        for m in ['delay', 'queue', 'reward']:
            arr = np.array(stats[m])
            print(f"{m.capitalize():10s}: {arr.mean():.2f} ± {arr.std():.2f}")

if __name__ == "__main__":
    test_generalization()
