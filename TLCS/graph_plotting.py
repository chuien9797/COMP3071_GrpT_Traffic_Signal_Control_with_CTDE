import os 
import pandas as pd
import matplotlib.pyplot as plt

# List of route types (must match filenames)
route_types = [
    "1x2_grid",
    "roundabout",
]

# Create a base output folder
base_folder = "plots"
os.makedirs(base_folder, exist_ok=True)

# Loop through each route type
for route in route_types:
    filename = f"holdout_seed_results_{route}.csv"  # ✅ updated extension
    
    if not os.path.isfile(filename):
        print(f"❌ File not found: {filename}")
        continue

    df = pd.read_csv(filename)

    output_dir = os.path.join(base_folder, route)
    os.makedirs(output_dir, exist_ok=True)

    metrics = {
        "delay_per_vehicle": "Delay per Vehicle",
        "avg_queue_length": "Average Queue Length",
        "throughput": "Throughput",
        "cumulative_reward": "Cumulative Reward",
        "total_emergency_delay": "Total Emergency Delay",         # ✅ added
        "avg_emergency_delay": "Average Emergency Delay",         # ✅ added
        "emergency_delay_ratio": "Emergency Delay Ratio"          # ✅ added
    }

    for column, title in metrics.items():
        if column not in df.columns:
            print(f"⚠️ Column not found in {filename}: {column}")
            continue
        
        plt.figure()
        plt.plot(df["seed"], df[column], marker='o')
        plt.title(f"{title} vs Seed ({route})")
        plt.xlabel("Seed")
        plt.ylabel(title)
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{column}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"✅ Plots saved for: {route}")
