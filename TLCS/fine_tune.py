import os
import datetime
import random
import numpy as np
import tensorflow as tf

# Import your project modules. Make sure these files are in your PYTHONPATH.
from utils import set_sumo, set_train_path
from intersection_config import INTERSECTION_CONFIGS
from generator import TrafficGenerator
from training_simulation import Simulation
from memory import Memory
from visualization import Visualization
from model import TrainModelAggregator


def main():
    """
    Fine-tuning the pre-trained model on a new environment (Y_intersection in this example).

    Steps performed:
    1. Specify the path to the pre-trained model.
    2. Create a new instance of your simulation for fine-tuning.
    3. Load the pre-trained model and adjust its learning rate.
    4. Run a short loop of fine-tuning episodes with reduced exploration (lower epsilon)
       and a lower learning rate.
    5. Save the fine-tuned model and plot the fine-tuning reward.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Specify the pre-trained model path.
    # -------------------------------------------------------------------------
    # Update this path to where your original training run saved the model.
    pretrained_model_path = os.path.join("models", "model_1")  # for example, adjust as needed

    # -------------------------------------------------------------------------
    # STEP 2: Setup the Fine-Tuning Environment.
    # -------------------------------------------------------------------------
    # In this example, we use the Y_intersection configuration.
    fine_tune_env = "Y_intersection"
    env_conf = INTERSECTION_CONFIGS[fine_tune_env]
    sumocfg_file = env_conf["sumocfg_file"]

    # Use your helper to configure SUMO. Note: the gui flag is set to False.
    sumo_cmd = set_sumo(gui=False, sumocfg_file_name=sumocfg_file, max_steps=5400)

    # Create Traffic Generator for the fine-tuning intersection.
    fine_tune_TrafficGen = TrafficGenerator(
        max_steps=5400,
        n_cars_generated=2000,
        intersection_type=fine_tune_env
    )

    # Create a new Memory instance.
    MemoryInstance = Memory(size_max=10000, size_min=600)

    # Determine the number of actions for this environment.
    # We use the number of actions defined in the phase mapping.
    max_num_actions = len(env_conf["phase_mapping"])

    # -------------------------------------------------------------------------
    # STEP 3: Create and Load the Pre-Trained Model, then update hyperparameters.
    # -------------------------------------------------------------------------
    # Here we create a new instance of your TrainModelAggregator.
    # Note: We set the learning rate lower for fine-tuning (e.g., 0.0001).
    Model = TrainModelAggregator(
        lane_feature_dim=5,  # Use your same lane feature dimension as before
        embedding_dim=32,  # The embedding dimension (same as training)
        final_hidden=64,  # Final hidden layer size (same as training)
        num_actions=max_num_actions,
        batch_size=100,
        learning_rate=0.0001  # Lower learning rate for fine-tuning
    )

    # Load the pre-trained model weights.
    # This will overwrite the randomly initialized weights with those learned from multi-env training.
    Model.load_from_disk(os.path.join(pretrained_model_path, "trained_model.h5"))
    print("Pre-trained model loaded from:", os.path.join(pretrained_model_path, "trained_model.h5"))

    # -------------------------------------------------------------------------
    # STEP 4: Setup the Fine-Tuning Simulation.
    # -------------------------------------------------------------------------
    # For fine-tuning, you might want to run fewer training epochs after each episode.
    # Here we set training_epochs=200 (which is typically much less than your main training).
    fine_tune_sim = Simulation(
        Model=Model,
        TargetModel=Model,  # You can use the same model as target for simplicity
        Memory=MemoryInstance,
        TrafficGen=fine_tune_TrafficGen,
        sumo_cmd=sumo_cmd,
        gamma=0.75,
        max_steps=5400,
        green_duration=20,
        yellow_duration=4,
        num_states=9999,  # This parameter is not used by the aggregator
        training_epochs=200,  # Reduced training epochs for fine-tuning
        intersection_type=fine_tune_env
    )

    # -------------------------------------------------------------------------
    # STEP 5: Run the Fine-Tuning Loop.
    # -------------------------------------------------------------------------
    # Here we run a small number of episodes with lower exploration.
    fine_tune_episodes = 10  # Fine-tuning typically uses a few episodes (e.g., 5–20)
    for ep in range(fine_tune_episodes):
        epsilon = 0.2  # Lower epsilon for fine-tuning (less random exploration)
        sim_time, train_time = fine_tune_sim.run(episode=ep, epsilon=epsilon)
        print(
            f"Fine-tune Episode {ep + 1}/{fine_tune_episodes} completed: sim_time = {sim_time}s, train_time = {train_time}s")

    # -------------------------------------------------------------------------
    # STEP 6: Save and Visualize the Fine-Tuned Model.
    # -------------------------------------------------------------------------
    # Save the new fine-tuned model in a separate folder.
    finetuned_model_path = set_train_path('fine_tuned_models')
    Model.save_model(finetuned_model_path)
    print("Fine-tuned model saved at:", finetuned_model_path)

    # Visualize the fine-tuning rewards.
    viz = Visualization(finetuned_model_path, dpi=96)
    viz.save_data_and_plot(
        data=fine_tune_sim.reward_store,
        filename=f"{fine_tune_env}_finetuned_reward",
        xlabel="Episode",
        ylabel="Cumulative negative reward"
    )

    print("Fine-tuning complete and visualization saved.")


if __name__ == "__main__":
    main()
