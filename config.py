# import os
import multiprocessing
import numpy as np
#
# # Computing env
seed = 42
num_cores = multiprocessing.cpu_count()-1
print("using {} CPUs".format(num_cores))
num_processes = "num_cpu"
num_processes_eval = "num_cpu"
#

# Agent
agent = 'A2C'
optimizer = 'adam'
value_loss_coef = 0.5
entropy_coef = 0.002
lr = 10 ** -2
eps = 10 ** -1
max_grad_norm = 10
acktr = False
alpha = 0.9
# use_clipped_value_loss = True

# # Training
# use_linear_lr_decay = True
gamma = 0.99
# use_gae = True
# gae_lambda = 0.95
# use_proper_time_limits = False
num_env_steps = 10 ** 4
trajectory_length = 40 # 256

scheduler = 'cyclic'
sched_ratio = config['sched_ratio']

# # logs
model_path = None

evaluate_every = 10
save_interval = 10
log_interval = 10 ** 2 #4

# World
env_type = 'QR' # chol or QR or LU
ncpu = 3
ngpu = 1
p = np.array([1] * ngpu + [0] * ncpu).astype(int)


A2C_settings = {
    "value_loss_coef": value_loss_coef,
    "entropy_coef": entropy_coef,
    "lr": lr,
    "eps": eps,
    "max_grad_norm": max_grad_norm,
    "acktr": acktr,
    "alpha": alpha
}
env_settings = {
    'n': 4,
    'node_types': p,
    'window': 1,
    'noise': 0,
}
env_settings2 = {
    'n': 4,
    'node_types': p,
    'window': 1,
    'noise': 0,
    'env_type': env_type,
}
seed_env = None

network_parameters = {"input_dim": 14}

config_enhanced = {
    'model_path': model_path,
    'num_env_steps': num_env_steps,
    'num_processes': num_processes,
    'lr': lr,
    'eps': eps,
    'optimizer': optimizer,
    'entropy_coef': entropy_coef,
    'seed': seed,
    'gamma': gamma,
    'loss_ratio': value_loss_coef,
    "evaluate_every": evaluate_every,
    'trajectory_length': trajectory_length,
    'log_interval': log_interval,
    'env_settings': env_settings,
    'env_settings2': env_settings2,
    'network_parameters': network_parameters,
    'agent': agent,
    'A2C_settings': A2C_settings,
    'seed_env': seed_env,
}
