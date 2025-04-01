import configparser
from sumolib import checkBinary
import os
import sys


def import_train_configuration(config_file):
    """
    Read the config file regarding training and import its content,
    including PPO‑specific hyperparameters.

    For multi-environment training (Method A), we do NOT rely on a single
    sumocfg_file from the .ini. Instead, each intersection's .sumocfg
    is stored in intersection_config.py under 'sumocfg_file'.
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    # Simulation
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')

    # Model (DQN)
    config['num_layers'] = content['model'].getint('num_layers')
    config['width_layers'] = content['model'].getint('width_layers')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['training_epochs'] = content['model'].getint('training_epochs')

    # Memory (DQN only)
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')

    # Agent (shared)
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['algorithm'] = content['agent'].get('algorithm')
    config['intersection_type'] = content['agent'].get('intersection_type', 'cross')

    # PPO-specific (if needed)
    if 'ppo' in content:
        config['ppo_hidden_size'] = content['ppo'].getint('hidden_size')
        config['ppo_learning_rate'] = content['ppo'].getfloat('learning_rate')
        config['ppo_clip_ratio'] = content['ppo'].getfloat('clip_ratio')
        config['ppo_update_epochs'] = content['ppo'].getint('update_epochs')
        config['ppo_training_epochs'] = content['ppo'].getint('training_epochs')

    # Paths
    config['models_path_name'] = content['dir']['models_path_name']

    # COMMENT OUT (or remove) to avoid forcing a single .sumocfg:
    # config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']

    return config


def import_test_configuration(config_file):
    """
    Read the config file regarding the testing and import its content.

    If you're doing single-environment testing, you might still want
    to use sumocfg_file_name from the .ini. Otherwise, you can also
    comment it out here if you're testing multiple intersections.
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    config['gui'] = content['simulation'].getboolean('gui')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['episode_seed'] = content['simulation'].getint('episode_seed')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['sumocfg_file_name'] = content['dir']['sumocfg_file_name']  # Keep for single-env test
    config['models_path_name'] = content['dir']['models_path_name']
    config['model_to_test'] = content['dir'].getint('model_to_test')

    return config


def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure various parameters of SUMO.
    If you're using multi-env training, sumocfg_file_name typically
    comes from intersection_config.py, not the .ini.
    """
    # sumo environment variable
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # either sumo (CMD) or sumo-gui
    if not gui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # build the command
    sumo_cmd = [
        sumoBinary,
        "-c", os.path.join('intersection', sumocfg_file_name),
        "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps)
    ]
    return sumo_cmd


def set_train_path(models_path_name):
    """
    Create a new model path with an incremental integer, also
    considering previously created model paths.
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)

    dir_content = os.listdir(models_path)
    if dir_content:
        # we expect folder names like "model_1", "model_2", ...
        previous_versions = [int(name.split("_")[1]) for name in dir_content]
        new_version = str(max(previous_versions) + 1)
    else:
        new_version = '1'

    data_path = os.path.join(models_path, 'model_' + new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def set_test_path(models_path_name, model_n):
    """
    Returns a model path for the tested model number, plus a 'test' subdir
    """
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_' + str(model_n), '')

    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else:
        sys.exit('The model number specified does not exist in the models folder')
