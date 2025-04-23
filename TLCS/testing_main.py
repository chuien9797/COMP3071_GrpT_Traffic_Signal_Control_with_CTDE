import os
import numpy as np
import tensorflow as tf

from utils import import_test_configuration, set_sumo, set_test_path
from intersection_config import INTERSECTION_CONFIGS
from generator_testing import TrafficGeneratorTesting
from testing_simulation import TestingSimulation
from model2 import TrainModelAggregator
from visualization import Visualization

# === Stress-test scenarios from your diagram
stress_scenarios = {
    "clean": {"emergency": False, "faults": False},
    "emergencies_only": {"emergency": True, "faults": False},
    "faults_only": {"emergency": False, "faults": True},
    "combined_stress": {"emergency": True, "faults": True},
}


def main():
    # === Load config
    config = import_test_configuration("testing_settings.ini")

    # === Get action space
    possible_envs = ["1x2_grid"]

    # possible_envs = ["double_t"]
    #max_num_actions = max(len(INTERSECTION_CONFIGS[env]["phase_mapping"]) for env in possible_envs)
    max_num_actions = 4
    config["num_actions"] = max_num_actions

    # === Load trained model
    model_path, plot_path = set_test_path(str(config["models_path_name"]), str(config["model_to_test"]))
    model_weights_path = os.path.join(model_path, "shared_policy", "trained_model.h5")

    shared_model = TrainModelAggregator(
        lane_feature_dim=9,
        embedding_dim=32,
        final_hidden=64,
        num_actions=max_num_actions,
        batch_size=1,
        learning_rate=0.001,
    )

    dummy_input = tf.random.uniform((1, 9, 9))
    shared_model._model(dummy_input)
    shared_model._model.load_weights(model_weights_path)
    print(f"✅ Weights loaded from: {model_weights_path}")

    # === Run evaluation for each stress scenario
    print("\n🔍 STARTING STRESS TEST EVALUATION...\n")
    for scenario_name, flags in stress_scenarios.items():
        print(f"\n🚨 SCENARIO: {scenario_name.upper()} | Emergency: {flags['emergency']} | Faults: {flags['faults']}")

        for env_name in possible_envs:
            print(f"→ Testing on environment: {env_name}")

            local_conf = config.copy()
            local_conf["intersection_type"] = env_name
            local_conf["inject_emergency"] = flags["emergency"]
            local_conf["inject_faults"] = flags["faults"]

            env_conf = INTERSECTION_CONFIGS[env_name]
            sumocfg_file = env_conf.get("sumocfg_file", f"{env_name}/{env_name}.sumocfg")
            sumo_cmd = set_sumo(local_conf["gui"], sumocfg_file, local_conf["max_steps"])

            TrafficGen = TrafficGeneratorTesting(
                max_steps=local_conf["max_steps"],
                n_cars_generated=local_conf["n_cars_generated"],
                intersection_type=env_name,
                inject_emergency=flags["emergency"]
            )

            num_intersections = len(env_conf["traffic_light_ids"])
            agents = [shared_model] * num_intersections

            sim = TestingSimulation(
                Models=agents,
                TrafficGen=TrafficGen,
                sumo_cmd=sumo_cmd,
                max_steps=local_conf["max_steps"],
                green_duration=local_conf["green_duration"],
                yellow_duration=local_conf["yellow_duration"],
                num_states=9999,
                intersection_type=env_name,
                inject_faults=flags["faults"]
            )

            sim_time = sim.run()
            print(f"✅ {env_name.upper()} finished | Sim Time: {sim_time}s")

            # Save plots
            scenario_folder = os.path.join(plot_path, scenario_name)
            os.makedirs(scenario_folder, exist_ok=True)
            tag = f"{env_name}_{scenario_name}"
            viz = Visualization(scenario_folder, dpi=96)

            viz.save_data_and_plot(sim.reward_store, filename=f"{tag}_reward",
                                   xlabel="Episode", ylabel="Reward")
            viz.save_data_and_plot(sim.cumulative_wait_store, filename=f"{tag}_delay",
                                   xlabel="Episode", ylabel="Cumulative Delay (s)")
            viz.save_data_and_plot(sim.avg_queue_length_store, filename=f"{tag}_queue",
                                   xlabel="Episode", ylabel="Avg Queue Length")

    print("\n🎯 ALL STRESS SCENARIOS COMPLETED. Plots saved at:", plot_path)


if __name__ == "__main__":
    main()
