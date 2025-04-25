import os
import sys
import numpy as np
import pandas as pd

from generator import TrafficGenerator
from training_simulation import Simulation
from model import TrainModelAggregator
from memory import Memory
from intersection_config import INTERSECTION_CONFIGS
from utils import import_train_configuration, set_sumo

# ================================
# Configuration
# ================================
CONFIG_PATH      = "training_settings.ini"
INTERSECTION_TYPE = "roundabout"
TEST_SEEDS       = list(range(200, 230))  # valid 0 <= seed < 2**32

# ================================
# Load Config & SUMO Setup
# ================================
config = import_train_configuration(CONFIG_PATH)
config["intersection_type"] = INTERSECTION_TYPE
sumocfg     = INTERSECTION_CONFIGS[INTERSECTION_TYPE]["sumocfg_file"]
base_sumo   = set_sumo(config["gui"], sumocfg, config["max_steps"])

# ================================
# Build & Initialize Model
# ================================
model = TrainModelAggregator(
    lane_feature_dim=9,
    embedding_dim=32,
    final_hidden=64,
    num_actions=config["num_actions"],
    batch_size=config["batch_size"],
    learning_rate=config["learning_rate"]
)
# dummy pass to build
max_lanes = max(
    len(lanes)
    for env in INTERSECTION_CONFIGS
    for lanes in INTERSECTION_CONFIGS[env]["incoming_lanes"].values()
)
dummy_in = np.zeros((1, max_lanes, 9), dtype=np.float32)
_ = model.model(dummy_in)

# ================================
# Load Weights
# ================================
model_h5 = os.path.join(
    os.getcwd(),
    config["models_path_name"],   # e.g., "models"
    "model_339",                  # adjust as needed
    "shared_policy",
    "trained_model.h5"
)
print(f"🔍 Loading weights from: {model_h5}")
try:
    model.model.load_weights(model_h5)
    print("✅ Weights loaded.\n")
    model.model.summary()
except Exception as e:
    print(f"❌ Could not load weights:\n{e}")
    sys.exit(1)

# ================================
# Prepare Shared Structures
# ================================
env_conf         = INTERSECTION_CONFIGS[INTERSECTION_TYPE]
num_agents       = len(env_conf.get("traffic_light_ids", [])) or 1
agents           = [model] * num_agents
target_agents    = [model] * num_agents
memory_instance  = Memory(config["memory_size_max"], config["memory_size_min"])

# ================================
# Evaluation Loop
# ================================
results = []
for seed in TEST_SEEDS:
    print(f"🚦 Evaluating seed {seed}...")
    tg = TrafficGenerator(
        max_steps=config["max_steps"],
        n_cars_generated=config["n_cars_generated"],
        intersection_type=INTERSECTION_TYPE,
        inject_emergency = True
    )
    sim = Simulation(
        Models=agents,
        TargetModels=target_agents,
        Memory=memory_instance,
        TrafficGen=tg,
        sumo_cmd=base_sumo,
        gamma=config["gamma"],
        max_steps=config["max_steps"],
        green_duration=config["green_duration"],
        yellow_duration=config["yellow_duration"],
        num_states=9999,  # unused but required
        training_epochs=config["training_epochs"],
        intersection_type=INTERSECTION_TYPE,
        signal_fault_prob=config.get("signal_fault_prob", 0.1)
    )

    # Run one episode with this seed (no exploration, no training)
    sim.run(episode=seed, epsilon=0.0, train=False)


    # Extract metrics
    avg_delay = sim._sum_waiting_time / config["n_cars_generated"]
    avg_queue = sim._sum_queue_length / config["max_steps"]
    throughput = config["n_cars_generated"]
    cum_reward = sim._sum_neg_reward

    # 1) total delay suffered by all emergency vehicles
    total_emergency_delay = sim._emergency_total_delay
    # 2) how many emergency vehicles were injected
    n_emergencies = sim._n_emergencies
    # 3) average delay per emergency vehicle

    if n_emergencies > 0:
        avg_emergency_delay = total_emergency_delay / n_emergencies
    else:
        avg_emergency_delay = 0.0
     # 4) (optional) what fraction of all wait-time was emergency
    if sim._sum_waiting_time > 0:
        emergency_delay_ratio = total_emergency_delay / sim._sum_waiting_time
    else:
        emergency_delay_ratio = 0.0

    results.append({
        "seed": seed,
        "delay_per_vehicle": avg_delay,
        "avg_queue_length": avg_queue,
        "throughput": throughput,
        "cumulative_reward": cum_reward,
        "total_emergency_delay": total_emergency_delay,
        "avg_emergency_delay": avg_emergency_delay,
        "emergency_delay_ratio": emergency_delay_ratio
    })

# ================================
# Save & Summarize
# ================================
df = pd.DataFrame(results)
df.to_csv("holdout_seed_results_roundabout.csv", index=False)

print("\n📊 Generalization Results (Hold-Out Seeds):\n", df.describe())
