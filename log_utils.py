# from config import config_enhanced
import os
from datetime import datetime

def name_env_dir(config_file):
    name = ""
    for k, v in config_file["env_settings"].items():
        name += (k + "=" + str(v) + "_")
    name += "seed=" + str(config_file['seed_env'])
    return name

def name_mod_dir(config_file):
    name = ""
    for k, v in config_file["network_parameters"].items():
        name += (k + "=" + str(v) + "_")
    name = name[:-1]
    return name

def name_dir(config_file):
    date_srt = str(datetime.today())[:-7].replace(' ', '_')
    env_str = name_env_dir(config_file)
    mod_str = name_mod_dir(config_file)
    return os.path.join(env_str, mod_str, date_srt)

def set_writer_dir(writer, config_file):
    name_dir = name_env_dir(config_file)
    current_dir = str(writer.get_logdir()).split('/')
    new_dir_path = os.path.join(current_dir[0], name_dir, current_dir[1])
    writer.log_dir = new_dir_path
# def name_model_sub_dir

def name_dir_logger(config_file):
    name = ""
    env_keys = ['env_type', 'n', 'nGPU', 'n_core', 'window', 'noise']
    for k in env_keys:
        v = config_file[k]
        name += (k + "=" + str(v) + "_")
    return name


# if __name__ == '__main__':
#     print(name_env_dir(config_enhanced))
#     print(str(datetime.today())[:-7].replace(' ', '_'))
#     print(name_mod_dir(config_enhanced))
#     print(name_dir(config_enhanced))