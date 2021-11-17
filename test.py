import os
import numpy as np
import time
from env import DAGEnv
import heft
import string
import matplotlib.pyplot as plt
import pandas as pd
import torch
from model import Net, SimpleNet, SimpleNet2, ResNetG, SimpleNetMax, SimpleNetW, SimpleNetWSage
import pickle as pkl
import torch
from collections import namedtuple

if __name__ == '__main__':

    model = torch.load('model_examples/cholesky_n=8_nGPU=2_nCPU=2/model_window=0.pth')
    model.eval()

    w_list = []
    n_node_list = []
    tile_list = []
    ready_node_list = []
    num_node_observation = []
    mean_time = []


    env_type = 'chol'
    nGPU = 2
    window = 0
    for n in [2, 4, 6, 8, 10, 12]:
        p_input = np.array([1] * nGPU + [0] * (4 - nGPU))
        env = DAGEnv(n, p_input, window, env_type=env_type)
        print(env.is_homogene)
        print("|V|: ", len(env.task_data.x))
        observation = env.reset()
        print(observation.keys())
        print(observation['graph'].x.shape)
        done = False
        time_step = 0
        total_time = 0
        while (not done) :
            start_time = time.time()
            with torch.no_grad():
                policy, value = model(observation)
            action_raw = policy.argmax().detach().cpu().numpy()
            ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
            action = -1 if action_raw == policy.shape[-1] - 1 else \
            observation['node_num'][ready_nodes][action_raw].detach().numpy()[0]

            observation, reward, done, info = env.step(action)
            cur_time = time.time() - start_time
            total_time += cur_time
            time_step += 1
            w_list.append(window)
            n_node_list.append(env.num_nodes)
            tile_list.append(n)
            mean_time.append(cur_time)


        print('n_node:', env.num_nodes)
        print(total_time/time_step)

    execution_time = pd.DataFrame({'w': w_list,
                                   'n_node': n_node_list,
                                   'tiles': tile_list,
                                   'time': mean_time})
    execution_time.to_csv("results/time.csv")
