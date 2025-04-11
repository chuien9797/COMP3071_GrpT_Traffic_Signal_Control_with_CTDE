import configparser
import os
import sys
from sumolib import checkBinary


def import_train_configuration(config_file):
    """
    Read the training configuration file and import its content.
    This version supports PPO-specific hyperparameters alongside the shared settings.
    """
    content = configparser.ConfigParser()
    content.read(config_file)
    config = {}

    # Simulation settings
    config['gui'] = content['simulation'].getboolean('gui')
    config['total_episodes'] = content['simulation'].getint('total_episodes')
    config['max_steps'] = content['simulation'].getint('max_steps')
    config['n_cars_generated'] = content['simulation'].getint('n_cars_generated')
    config['green_duration'] = content['simulation'].getint('green_duration')
    config['yellow_duration'] = content['simulation'].getint('yellow_duration')

    # Model parameters (can be used as defaults or for architecture details)
    config['num_layers'] = content['model'].getint('num_layers')
    config['width_layers'] = content['model'].getint('width_layers')
    config['batch_size'] = content['model'].getint('batch_size')
    config['learning_rate'] = content['model'].getfloat('learning_rate')
    config['training_epochs'] = content['model'].getint('training_epochs')

    # Memory settings (for DQN; might not be used in PPO)
    config['memory_size_min'] = content['memory'].getint('memory_size_min')
    config['memory_size_max'] = content['memory'].getint('memory_size_max')

    # Agent settings (shared)
    config['num_states'] = content['agent'].getint('num_states')
    config['num_actions'] = content['agent'].getint('num_actions')
    config['gamma'] = content['agent'].getfloat('gamma')
    config['algorithm'] = content['agent'].get('algorithm')
    config['intersection_type'] = content['agent'].get('intersection_type', 'cross')

    # PPO-specific settings
    if 'ppo' in content:
        config['ppo_hidden_size'] = content['ppo'].getint('hidden_size')
        config['ppo_learning_rate'] = content['ppo'].getfloat('learning_rate')
        config['ppo_clip_ratio'] = content['ppo'].getfloat('clip_ratio')
        config['ppo_update_epochs'] = content['ppo'].getint('update_epochs')
        config['ppo_training_epochs'] = content['ppo'].getint('training_epochs')
        # Overwrite the gamma with a PPO-specific value if provided
        config['ppo_gamma'] = content['ppo'].getfloat('gamma')
        config['gae_lambda'] = content['ppo'].getfloat('gae_lambda')
        config['entropy_coef'] = content['ppo'].getfloat('entropy_coef')

    # Directory paths
    config['models_path_name'] = content['dir'].get('models_path_name')

    # Fault Curriculum settings (if applicable)
    config['FaultCurriculum'] = {}
    for key in content['FaultCurriculum']:
        config['FaultCurriculum'][key] = content['FaultCurriculum'][key]

    return config


def import_test_configuration(config_file):
    """
    Read the testing configuration file and import its content.
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
    config['sumocfg_file_name'] = content['dir'].get('sumocfg_file_name')
    config['models_path_name'] = content['dir'].get('models_path_name')
    config['model_to_test'] = content['dir'].getint('model_to_test')
    return config


def set_sumo(gui, sumocfg_file_name, max_steps):
    """
    Configure and return the SUMO command.
    """
    # Check for SUMO_HOME and append SUMO tools to the system path
    if 'SUMO_HOME' in os.environ:
        tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
        sys.path.append(tools)
    else:
        sys.exit("please declare environment variable 'SUMO_HOME'")

    # Determine which binary to use (with or without GUI)
    if not gui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    # Build the SUMO command
    sumo_cmd = [
        sumoBinary,
        "-c", os.path.join('intersection', sumocfg_file_name),
        "--no-step-log", "true",
        "--waiting-time-memory", str(max_steps)
    ]
    return sumo_cmd


def set_train_path(models_path_name):
    """
    Create a new training model folder with an incremental version number.
    """
    models_path = os.path.join(os.getcwd(), models_path_name, '')
    os.makedirs(os.path.dirname(models_path), exist_ok=True)
    dir_content = os.listdir(models_path)
    if dir_content:
        previous_versions = [int(name.split("_")[1]) for name in dir_content if name.startswith("model_")]
        new_version = str(max(previous_versions) + 1) if previous_versions else '1'
    else:
        new_version = '1'
    data_path = os.path.join(models_path, 'model_' + new_version, '')
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    return data_path


def set_test_path(models_path_name, model_n):
    """
    Return a model path for testing a particular model.
    """
    model_folder_path = os.path.join(os.getcwd(), models_path_name, 'model_' + str(model_n), '')
    if os.path.isdir(model_folder_path):
        plot_path = os.path.join(model_folder_path, 'test', '')
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        return model_folder_path, plot_path
    else:
        sys.exit('The model number specified does not exist in the models folder')
