from torch_geometric.data import Data

import torch
import networkx as nx
from torch_geometric.utils.convert import to_networkx

# import pydot
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
import numpy as np
import re
import os
import pickle as pkl
# task type 0: POTRF 1:SYRK 2:TRSM 3: GEMMS

# Heterougenous case
durations_cpu = [18, 57, 52, 95, 0]
durations_gpu = [11, 2, 8, 3, 0]

durations_cpu_lu = [18, 95, 52, 52]
durations_gpu_lu = [11, 3, 8, 8]

durations_cpu_qr = [4, 6, 6, 10]
durations_gpu_qr = [3, 3, 1, 1]

# Homogenous case
# durations_cpu = [1, 3, 3, 6, 0]
# durations_gpu = [11, 2, 8, 3, 0]

# durations_cpu = [24, 52, 57, 95, 0]
# durations_gpu2 = [12, 1, 3, 2, 0]

simple_durations = [1, 3, 3, 6, 0]

colors = {0: [0, 0, 0], 1: [230, 190, 255], 2: [170, 255, 195], 3: [255, 250, 200],
          4: [255, 216, 177], 5: [250, 190, 190], 6: [240, 50, 230], 7: [145, 30, 180], 8: [67, 99, 216],
          9: [66, 212, 244], 10: [60, 180, 75], 11: [191, 239, 69], 12: [255, 255, 25], 13: [245, 130, 49],
          14: [230, 25, 75], 15: [128, 0, 0], 16: [154, 99, 36], 17: [128, 128, 0], 18: [70, 153, 144],
          19: [0, 0, 117]}
color_normalized = {i: list(np.array(colors[i])/255) for i in colors}


class Task():

    def __init__(self, barcode, noise=0, task_type='chol'):
        """
        task_type 0: POTRF 1:SYRK 2:TRSM 3: GEMMS
        0: LU 1: BMOD 2: BDIV 3: FW
        """

        self.type = barcode[0]
        if task_type=='chol':
            self.duration_cpu = durations_cpu[self.type]
            self.duration_gpu = durations_gpu[self.type]

        elif task_type == 'LU':
            self.duration_cpu = durations_cpu_lu[self.type]
            self.duration_gpu = durations_gpu_lu[self.type]

        elif task_type == 'QR':
            self.duration_cpu = durations_cpu_qr[self.type]
            self.duration_gpu = durations_gpu_qr[self.type]
        else:
            raise NotImplementedError('task type unknown')
        self.barcode = barcode
        self.durations = [self.duration_cpu, self.duration_gpu]
        # if noise and self.type == 3:
        #     if np.random.uniform() < 1/15:
        #         self.durations[-1] *= 3
        if noise > 0:
            self.durations[-1] += np.max([np.random.normal(0, noise), -2])


class TaskGraph(Data):

    def __init__(self, x, edge_index, task_list):
        Data.__init__(self, x, edge_index.to(torch.long))
        self.task_list = np.array(task_list)
        self.task_to_num = {v: k for (k, v) in enumerate(self.task_list)}
        self.n = len(self.x)

    def render(self, root=None):
        # graph = self.data
        task_list = [t.barcode for t in self.task_list]
        graph = to_networkx(Data(self.x, self.edge_index.contiguous()))
        pos = graphviz_layout(graph, prog='dot', root=root)
        # pos = graphviz_layout(G, prog='twopi')
        node_color = [color_normalized[task[0]] for task in task_list]
        # plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(graph, pos, node_color=node_color)
        nx.draw_networkx_edges(graph, pos)

    def remove_edges(self, node_list):
        # mask_node = torch.logical_not(isin(self.x, node_list))
        # self.x = self.x[mask_node]
        mask_edge = isin(self.edge_index[0, :], torch.tensor(node_list)) | \
                    isin(self.edge_index[1, :], torch.tensor(node_list))
        self.edge_index = self.edge_index[:, torch.logical_not(mask_edge)]

    def add_features_descendant(self):
        n = self.n
        x = self.x
        succ_features = torch.zeros((n, 4))
        succ_features_norm = torch.zeros((n, 4))
        edges = self.edge_index
        for i in reversed(range(n)):
            succ_i = edges[1][edges[0] == i]
            feat_i = x[i] + torch.sum(succ_features[succ_i], dim=0)
            n_pred_i = torch.FloatTensor([torch.sum(edges[1] == j) for j in succ_i])
            if len(n_pred_i) == 0:
                feat_i_norm = x[i]
            else:
                feat_i_norm = x[i] + torch.sum(succ_features_norm[succ_i] / n_pred_i.unsqueeze(1).repeat((1, 4)), dim=0)
            succ_features[i] = feat_i
            succ_features_norm[i] = feat_i_norm
        return succ_features_norm, succ_features
        # return succ_features_norm, succ_features/succ_features[0]

class Node():
    def __init__(self, type):
        self.type = type


class Cluster():
    def __init__(self, node_types, communication_cost):
        """
        :param node_types:
        :param communication_cost: [(u,v,w) with w weight]
        """
        self.node_types = node_types
        self.node_state = np.zeros(len(node_types))
        self.communication_cost = communication_cost


    def render(self):
        edges_list = [(u, v, {"cost": w}) for (u, v, w) in enumerate(self.communication_cost)]
        colors = ["k" if node_type == 0 else "red" for node_type in self.node_types]
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.node_types))))
        G.add_edges_from(edges_list)
        pos = graphviz_layout(G)

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors)
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos)


def succASAP(task, n, noise):
    tasktype = task.type
    i = task.barcode[1]
    j = task.barcode[2]
    k = task.barcode[3]
    listsucc = []
    if tasktype == 0:
        if i < n:
            for j in range(i + 1, n + 1, 1):
                y = (2, i, j, 0)
                listsucc.append(Task(y, noise))
        else:
            y = (4, 0, 0, 0)
            listsucc.append(Task(y))

    if tasktype == 1:
        if j < i - 1:
            y = (1, i, j + 1, 0)
            listsucc.append(Task(y))
        else:
            y = (0, i, 0, 0)
            listsucc.append(Task(y, noise))

    if tasktype == 2:
        if i <= n - 1:
            for k in range(i + 1, j):
                y = (3, k, j, i)
                listsucc.append(Task(y, noise))
            for k in range(j + 1, n + 1):
                y = (3, j, k, i)
                listsucc.append(Task(y, noise))
            y = (1, j, i, 0)
            listsucc.append(Task(y, noise))

    if tasktype == 3:
        if k < i - 1:
            y = (3, i, j, k + 1)
            listsucc.append(Task(y, noise))
        else:
            y = (2, i, j, 0)
            listsucc.append(Task(y, noise))

    return listsucc


def CPAndWorkBelow(x, n, durations):
    x_bar = x.barcode
    C = durations[0]
    S = durations[1]
    T = durations[2]
    G = durations[3]
    ReadyTasks = []
    ReadyTasks.append(x_bar)
    Seen = []
    ToVisit = []
    ToVisit.append(x_bar)
    TotalWork = durations[x_bar[0]]
    CPl = 0
    # while len(ToVisit) > 0:
    #     for t in ToVisit:
    #         for succ in succASAP(Task(t), n):
    #             if succ not in Seen:
    #                 succ = succ.barcode
    #                 TotalWork = TotalWork + durations[succ[0]]
    #                 Seen.append(succ)
    #                 ToVisit.append(succ)
    #         ToVisit.remove(t)

    tasktype = x_bar[0]
    if tasktype == 0:
        CPl = C + (n - x_bar[1]) * (T + S + C)
    if tasktype == 1:
        CPl = (x_bar[1] - x_bar[2]) * S + C + (n - x_bar[1]) * (T + S + C)
    if tasktype == 2:
        CPl = (x_bar[2] - x_bar[1] - 1) * (T + G) + (n - x_bar[2] + 1) * (T + S + C)
    if tasktype == 3:
        CPl = (x_bar[1] - x_bar[3]) * G + (x_bar[2] - x_bar[1] - 1) * (T + G) + (n - x_bar[2] + 1) * (T + S + C)

    return (CPl, TotalWork)

def _add_task(dic_already_seen, list_to_process, task):
    if task.barcode in dic_already_seen:
        pass
    else:
        dic_already_seen[task.barcode] = len(dic_already_seen)
        list_to_process.append(task)


def _add_node(dic_already_seen, list_to_process, node):
    if node in dic_already_seen:
        pass
    else:
        dic_already_seen[node] = True
        list_to_process.append(node)


def compute_graph(n, noise=False):
    root_nodes = []
    TaskList = {}
    EdgeList = []

    root_nodes.append(Task((0, 1, 0, 0), noise))
    TaskList[(0, 1, 0, 0)] = 0

    while len(root_nodes) > 0:
        task = root_nodes.pop()
        list_succ = succASAP(task, n, noise)
        for t_succ in list_succ:
            _add_task(TaskList, root_nodes, t_succ)
            EdgeList.append((TaskList[task.barcode], TaskList[t_succ.barcode]))

    # embeddings
    embeddings = [k for k in TaskList]

    data = Data(x=torch.tensor(embeddings, dtype=torch.float),
                edge_index=torch.tensor(EdgeList).t().contiguous())

    task_array = []
    for (k, v) in TaskList.items():
        task_array.append(Task(k, noise=noise))
    return TaskGraph(x=torch.tensor(embeddings, dtype=torch.float),
                edge_index=torch.tensor(EdgeList).t().contiguous(), task_list=task_array)
    # return data, task_array


def isin(ar1, ar2):
    return (ar1[..., None] == ar2).any(-1)

def remove_nodes(edge_index, mask, num_nodes):
    r"""Removes the isolated nodes from the graph given by :attr:`edge_index`
    with optional edge attributes :attr:`edge_attr`.
    In addition, returns a mask of shape :obj:`[num_nodes]` to manually filter
    out isolated node features later on.
    Self-loops are preserved for non-isolated nodes.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (LongTensor, Tensor, BoolTensor)
    """

    assoc = torch.full((num_nodes,), -1, dtype=torch.long, device=mask.device)
    assoc[mask] = torch.arange(mask.sum(), device=assoc.device)
    edge_index = assoc[edge_index]

    return edge_index

def compute_sub_graph(data, root_nodes, window):
    """
    :param data: the whole graph
    :param root_nodes: list of node numbers
    :param window: the max distance to go down from the root nodes
    :return: the sub graph with nodes at distance less than h from root_nodes
    """

    already_seen = torch.zeros(data.num_nodes, dtype=torch.bool)
    already_seen[root_nodes] = 1
    edge_list = torch.tensor([[], []], dtype=torch.long)

    i = 0
    while len(root_nodes) > 0 and i < window:
        mask = isin(data.edge_index[0], root_nodes)
        list_succ = data.edge_index[1][mask]
        list_pred = data.edge_index[0][mask]

        edge_list = torch.cat((edge_list, torch.stack((list_pred, list_succ))), dim=1)

        list_succ = torch.unique(list_succ)

        list_succ = list_succ[already_seen[list_succ] == 0]
        already_seen[list_succ] = 1
        root_nodes = list_succ
        i += 1

    assoc = torch.full((len(data.x),), -1, dtype=torch.long)
    assoc[already_seen] = torch.arange(already_seen.sum())

    node_num = torch.nonzero(already_seen)
    new_x = data.x[already_seen]
    new_edge_index = remove_nodes(data.edge_index, already_seen, len(data.x))
    mask_edge = (new_edge_index != -1).all(dim=0)
    new_edge_index = new_edge_index[:, mask_edge]
    new_task_list = data.task_list[already_seen]

    return TaskGraph(new_x, new_edge_index, new_task_list), node_num


def taskGraph2SLC(taskGraph, save_path):
    with open(save_path,"w") as file:
        file.write(str(len(taskGraph.task_list)))
        file.write('\n')
        for node, task in enumerate(taskGraph.task_list):
            line1 = str(node + 1) + " " + str(simple_durations[task.type]) + " 1"
            file.write(line1)
            file.write('\n')

            line2 = ""
            for n in taskGraph.edge_index[1][taskGraph.edge_index[0] == node]:
                line2 += str(n.item() + 1) + " 0 "
            line2 += "-1"
            file.write(line2)
            file.write('\n')
        # file.write("-1")

def random_ggen_fifo_edges(n_vertex, max_in, max_out):
    stream = os.popen("ggen generate-graph fifo {:d} {:d} {:d}".format(n_vertex, max_in, max_out))
    graph = stream.read()
    out_graph = graph.split('dag')[1].replace('\n', '').replace('\t', '').replace('{', '[[').replace('}', ']]')
    out_graph = out_graph.replace(' -> ', ', ')
    out_graph = out_graph.replace(';', '], [')
    out_graph = eval(out_graph)
    out_graph.pop()
    edge_index = np.transpose(np.array(out_graph))
    return edge_index


def random_ggen_fifo(n_vertex, max_in, max_out, noise=0):
    edges = random_ggen_fifo_edges(n_vertex, max_in, max_out)
    n = np.max(edges) + 1
    tasks = np.random.randint(0, 4, size=n)
    x = np.zeros((n, 4), dtype=int)
    x[np.arange(n), tasks] = 1
    task_list = []
    for t in tasks:
        task_list.append(Task((t), noise=noise))
    return TaskGraph(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edges), task_list=task_list)

def ggen_cholesky(n_vertex, noise=0):
    dic_task_ch = {'p': 0, 's': 1, 't': 2, 'g': 3}

    def parcours_and_purge(s):
        reg = re.compile('\[kernel=[\D]*\]')
        x = reg.findall(s)
        return np.array([dic_task_ch[subx[8:9]] for subx in x])

    def parcours_and_purge_edges(s):
        reg = re.compile('\\t[\d]+ -> [\d]+\\t')
        x = reg.findall(s)
        out = np.array([[int(subx.split(' -> ')[0][1:]), int(subx.split(' -> ')[1][:-1])] for subx in x])
        return out.transpose()

    file_path = 'graphs/cholesky_{}.txt'.format(n_vertex)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            graph = f.read()
    else:
        stream = os.popen("ggen dataflow-graph cholesky {:d}".format(n_vertex))
        graph = stream.read()

    edges = parcours_and_purge_edges(graph)
    tasks = parcours_and_purge(graph)
    n = len(tasks)
    x = np.zeros((n, 4), dtype=int)
    x[np.arange(n), tasks.astype(int)] = 1
    task_list = []
    for i, t in enumerate(tasks):
        task_list.append(Task((t, i), noise=noise, task_type='chol'))
    return TaskGraph(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edges), task_list=task_list)

def ggen_denselu(n_vertex, noise=0):
    dic_task_lu = {'lu': 0, 'bm': 1, 'bd': 2, 'fw': 3}

    def parcours_and_purge(s):
        reg = re.compile('\[kernel=[\D]*\]')
        x = reg.findall(s)
        return np.array([dic_task_lu[subx[8:10]] for subx in x])

    def parcours_and_purge_edges(s):
        reg = re.compile('\\t[\d]+ -> [\d]+\\t')
        x = reg.findall(s)
        out = np.array([[int(subx.split(' -> ')[0][1:]), int(subx.split(' -> ')[1][:-1])] for subx in x])
        return out.transpose()

    file_path = '/home/ngrinsztajn/HPC/graphs/denselu_{}.txt'.format(n_vertex)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            graph = f.read()
    else:
        stream = os.popen("ggen dataflow-graph denselu {:d}".format(n_vertex))
        graph = stream.read()
    edges = parcours_and_purge_edges(graph)
    tasks = parcours_and_purge(graph)
    n = len(tasks)
    x = np.zeros((n, 4), dtype=int)
    x[np.arange(n), tasks] = 1
    task_list = []
    for i, t in enumerate(tasks):
        task_list.append(Task((t, i), noise=noise, task_type='LU'))
    return TaskGraph(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edges), task_list=task_list)


def ggen_QR(n, noise=0):
    # file_path = '/home/ngrinsztajn/HPC/graphs/QR_{}.pkl'.format(n)
    file_path = '/home/nathan/PycharmProjects/HPC/graphs/QR_{}.pkl'.format(n)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            output = pkl.load(f)
            return output
        print('file loaded')

    numtask = 0

    numtasks = {}
    listtask = []

    for i in range(1, n + 1):
        numtasks[0, i, 0, 0] = numtask
        numtask = numtask + 1
        listtask.append(0)

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            numtasks[1, j, i, 0] = numtask
            numtask = numtask + 1
            listtask.append(1)

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            numtasks[2, i, j, 0] = numtask
            numtask = numtask + 1
            listtask.append(2)

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            for k in range(i + 1, n + 1):
                numtasks[3, i, j, k] = numtask
                numtask = numtask + 1
                listtask.append(3)

    listedges = []
    for i in range(1, n):
        source = numtasks[0, i, 0, 0]
        for j in range(i + 1, n + 1):
            dest = numtasks[1, j, i, 0]
            listedges.append((source, dest))

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            source = numtasks[1, j, i, 0]
            for k in range(i + 1, n + 1):
                dest = numtasks[3, i, j, k]
                listedges.append((source, dest))

    for i in range(1, n):
        for j in range(i + 1, n + 1):
            source = numtasks[2, i, j, 0]
            for k in range(i + 1, n + 1):
                dest = numtasks[3, i, k, j]
                listedges.append((source, dest))

    for i in range(1, n - 1):
        for j in range(i + 2, n + 1):
            for k in range(i + 2, n + 1):
                source = numtasks[3, i, j, k]
                dest = numtasks[3, i + 1, j, k]
                listedges.append((source, dest))

    for i in range(1, n - 1):
        for k in range(i + 2, n):
            source = numtasks[3, i, i + 1, k]
            dest = numtasks[2, i + 1, k, 0]
            listedges.append((source, dest))

    for i in range(1, n - 1):
        for j in range(i + 2, n):
            source = numtasks[3, i, j, i + 1]
            dest = numtasks[1, j, i + 1, 0]
            listedges.append((source, dest))

    for i in range(1, n - 1):
        source = numtasks[3, i, i + 1, i + 1]
        dest = numtasks[0, i + 1, 0, 0]
        listedges.append((source, dest))

    tasks = np.array(listtask)
    edges = np.array(listedges).transpose()

    n = len(tasks)
    x = np.zeros((n, 4), dtype=int)
    x[np.arange(n), tasks] = 1
    task_list = []
    for i, t in enumerate(tasks):
        task_list.append(Task((t, i), noise=noise, task_type='QR'))
    return TaskGraph(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(edges), task_list=task_list)

# def random_denselu(n_vertex, noise=0):
#     def parcours_and_purge(s):
#         x = []
#         for i in range(len(s) - 6):
#             if s[i:i + 7] == 'kernel=':
#                 x.append(dic_task[s[i + 7]])
#         return np.array([int(i) for i in x])
#
#     def parcours_and_purge_edges(s):
#         edge_index = []
#         for i in range(len(s) - 6):
#             if s[i:i + 2] == '->':
#                 partial_s = s[i - 4:i + 5]
#                 edge_index.append([int(j) for j in partial_s.split('\t')[1].split(' -> ')])
#         return np.array(edge_index).transpose()
#
#     stream = os.popen("ggen generate-graph fifo {:d} {:d} {:d}".format(n_vertex, max_in, max_out))
#     graph = stream.read()
#     edges = random_ggen_fifo_edges(n_vertex, max_in, max_out)
#     n = np.max(edges) + 1
#     tasks = np.random.randint(0, 4, size=n)
#     x = np.zeros((n, 4), dtype=int)
#     x[np.arange(n), tasks] = 1
#     task_list = []
#     for t in tasks:
#         task_list.append(Task([t], noise=noise))
#     return TaskGraph(x=torch.tensor(x, dtype=torch.float),
#                 edge_index=torch.tensor(edges), task_list=task_list)


# a = random_ggen_fifo(20, 5, 5, 0)
# print(a)
# ggen dataflow-graph cholesky 3