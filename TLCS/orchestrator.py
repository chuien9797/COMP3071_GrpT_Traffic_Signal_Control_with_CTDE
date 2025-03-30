import os
import subprocess
import sys
import configparser

def update_config(intersection_type, ini_path="training_settings.ini"):
    config = configparser.ConfigParser()
    config.read(ini_path)
    if "agent" not in config:
        config["agent"] = {}
    config["agent"]["intersection_type"] = intersection_type
    with open(ini_path, "w") as configfile:
        config.write(configfile)

def run_training(intersection_type):
    print("Starting training for intersection type:", intersection_type)
    # Update the configuration file with the new intersection type
    update_config(intersection_type)
    # Use sys.executable to ensure the same interpreter is used
    process = subprocess.Popen(
        [sys.executable, "training_main.py", intersection_type],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    # Stream output in real time
    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    print("Training process for", intersection_type, "ended with return code", rc)

def main():
    # List of intersection types to run sequentially
    intersection_types = ["cross", "roundabout", "T_intersection"]
    for itype in intersection_types:
        run_training(itype)

if __name__ == "__main__":
    main()
