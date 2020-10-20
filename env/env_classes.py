import gym
from gym.spaces import Box, Dict
import string

from env.utils import *
from env.utils import compute_graph
import heft

class DAGEnv(gym.Env):
    def __init__(self, n, node_types, window, env_type, noise=False):
        if isinstance(node_types, int):
            p = node_types
            node_types = np.ones(p)
        else:
            p = len(node_types)

        self.observation_space = Dict
        self.action_space = "Graph"

        self.noise = noise
        self.time = 0
        self.num_steps = 0
        self.p = p
        self.n = n
        self.window = window
        self.env_type = env_type
        if self.env_type == 'LU':
            self.max_duration_cpu = max(durations_cpu_lu)
            self.max_duration_gpu = max(durations_gpu_lu)
            self.task_data = ggen_denselu(self.n, self.noise)
        elif self.env_type == 'QR':
            self.max_duration_cpu = max(durations_cpu_qr)
            self.max_duration_gpu = max(durations_gpu_qr)
            self.task_data = ggen_QR(self.n, self.noise)
        elif self.env_type == 'chol':
            self.max_duration_cpu = max(durations_cpu)
            self.max_duration_gpu = max(durations_gpu)
            self.task_data = ggen_cholesky(self.n, self.noise)
        else:
            raise EnvironmentError('not implemented')
        self.num_nodes = self.task_data.num_nodes
        self.sum_task = torch.sum(self.task_data.x, dim=0)
        self.norm_desc_features = self.task_data.add_features_descendant()[0] / self.sum_task
        self.cluster = Cluster(node_types=node_types.astype(int), communication_cost=np.zeros((p, p)))
        self.running = -1 * np.ones(p)  # array of task number
        self.running_task2proc = {}
        self.ready_proc = np.zeros(p)  # for each processor, the time where it becomes available
        self.ready_tasks = []
        self.processed = {}
        self.compeur_task = 0
        self.current_proc = 0
        self.is_homogene = (np.mean(self.cluster.node_types) - 1) * np.mean(self.cluster.node_types) == 0

        self.critic_path_duration = None
        self.total_work_normalized = None
        # self.task_to_CP = np.zeros(len(self.task_graph.task_list))

        # compute heft
        string_cluster = string.printable[:self.p]
        dic_heft = {}
        for edge in np.array(self.task_data.edge_index.t()):
            dic_heft[edge[0]] = dic_heft.get(edge[0], ()) + (edge[1],)

        def compcost(job, agent):
            idx = string_cluster.find(agent)
            duration = self.task_data.task_list[job].durations[self.cluster.node_types[idx]]
            return duration

        def commcost(ni, nj, A, B):
            return 0

        orders, jobson = heft.schedule(dic_heft, string_cluster, compcost, commcost)
        try:
            self.heft_time = orders[jobson[self.num_nodes - 1]][-1].end
        except:
            # ok if test
            self.heft_time = max([v[-1] for v in orders.values() if len(v) > 0])

    def reset(self):
        # self.task_data = random_ggen_fifo(self.n, self.max_in, self.max_out, self.noise)
        if self.env_type == 'LU':
            self.task_data = ggen_denselu(self.n, self.noise)
        elif self.env_type == 'QR':
            self.task_data = ggen_QR(self.n, self.noise)
        elif self.env_type == 'chol':
            self.task_data = ggen_cholesky(self.n, self.noise)
        else:
            raise EnvironmentError('not implemented')
        self.time = 0
        self.num_steps = 0
        self.running = -1 * np.ones(self.p).astype(int)
        self.running_task2proc = {}
        self.ready_proc = np.zeros(self.p)
        # self.ready_tasks.append(0)
        self.current_proc = 0

        # compute initial doable tasks

        new_ready_tasks = torch.arange(0, self.num_nodes)[torch.logical_not(isin(torch.arange(0, self.num_nodes), self.task_data.edge_index[1, :]))]
        self.ready_tasks = new_ready_tasks.tolist()

        self.processed = {}
        self.compeur_task = 0

        # if self.noise > 0:
        #     for i in range(self.task_data.num_nodes):
        #         self.task_data.task_list[i].durations[1] = self.task_data.task_list[i].duration_gpu + np.random.normal(0, self.noise)


        return self._compute_state()

    def step(self, action, render_before=False, render_after=False, enforce=True, speed=False):
        """
        first implementation, with only [-1, 0, ..., T] actions
        :param action: -1: does nothing. t: schedules t on the current available processor
        :return: next_state, reward, done, info
        """

        self.num_steps += 1

        self._find_available_proc()

        if action == -1 and enforce:
            if len(self.running_task2proc) == 0:
                # the agent does nothing but every proc is available: we enforce an arbitrary action
                action = self.ready_tasks[0]
        if action != -1:
            self.compeur_task += 1
        self._choose_task_processor(action, self.current_proc)

        if render_before:
            self.render()

        done = self._go_to_next_action(action, enforce)

        if render_after and not speed:
            self.render()

        reward = (self.heft_time - self.time)/self.heft_time if done else 0

        info = {'episode': {'r': reward, 'length': self.num_steps, 'time': self.time}, 'bad_transition': False}

        if speed:
            return 0, reward, done, info

        return self._compute_state(), reward, done, info

    def _find_available_proc(self):
        while (self.current_proc < self.p) and (self.running[self.current_proc] > -1):
            self.current_proc += 1
        if self.current_proc == self.p:
            # no new proc available
            self.current_proc == 0
            self._forward_in_time()
        while (self.current_proc < self.p) and (self.running[self.current_proc] > -1):
            self.current_proc += 1

    def _forward_in_time(self):
        if len(self.ready_proc[self.ready_proc > self.time]) > 0:
            min_time = np.min(self.ready_proc[self.ready_proc > self.time])
        else:
            min_time = 0

        self.time = min_time

        self.ready_proc[self.ready_proc < self.time] = self.time

        tasks_finished = self.running[np.logical_and(self.ready_proc == self.time, self.running > -1)].copy()

        self.running[self.ready_proc == self.time] = -1
        for task in tasks_finished:
            del self.running_task2proc[task]

        # compute successors of finished tasks
        mask = isin(self.task_data.edge_index[0], torch.tensor(tasks_finished))
        list_succ = self.task_data.edge_index[1][mask]
        list_succ = torch.unique(list_succ)

        # remove nodes
        self.task_data.remove_edges(tasks_finished)

        # compute new available tasks
        new_ready_tasks = list_succ[torch.logical_not(isin(list_succ, self.task_data.edge_index[1, :]))]
        self.ready_tasks += new_ready_tasks.tolist()


        self.current_proc = np.argmin(self.running)

    def _go_to_next_action(self, previous_action, enforce=True):
        has_just_passed = self.is_homogene and previous_action == -1 and enforce
        if has_just_passed:
            self._forward_in_time()
        elif previous_action == -1:
            self.current_proc += 1
        while len(self.ready_tasks) == 0:
            self._forward_in_time()
            if self._isdone():
                return True
        self._find_available_proc()
        return False

    def _choose_task_processor(self, action, processor):
        # assert action in self.ready_tasks

        if action != -1:
            self.ready_proc[processor] += self.task_data.task_list[action].durations[self.cluster.node_types[processor]]
            self.ready_tasks.remove(action)
            self.processed[self.task_data.task_list[action].barcode] = [processor, self.time]
            self.running_task2proc[action] = processor
            self.running[processor] = action

    def _compute_state(self):
        visible_graph, node_num = compute_sub_graph(self.task_data,
                                          torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                       self.ready_tasks)), dtype=torch.long),
                                          self.window)
        visible_graph.x, ready = self._compute_embeddings(node_num)
        return {'graph': visible_graph, 'node_num': node_num, 'ready': ready}

    def _remaining_time(self, running_tasks):
        return torch.tensor([self.ready_proc[self.running_task2proc[task.item()]] for task in running_tasks]) - self.time

    def _isdone(self):
        # return (self.task_data.edge_index.shape[-1] == 0) and (len(self.running_task2proc) == 0)
        return (self.compeur_task == self.num_nodes and (len(self.running_task2proc) == 0))

    def _compute_embeddings(self, tasks):

        ready = isin(tasks, torch.tensor(self.ready_tasks)).float()
        running = isin(tasks, torch.tensor(self.running[self.running > -1])).squeeze(-1)

        remaining_time = torch.zeros(tasks.shape[0])
        remaining_time[running] = self._remaining_time(tasks[running].squeeze(-1)).to(torch.float)
        remaining_time = remaining_time.unsqueeze(-1)

        n_succ = torch.sum((tasks == self.task_data.edge_index[0]).float(), dim=1).unsqueeze(-1)
        n_pred = torch.sum((tasks == self.task_data.edge_index[1]).float(), dim=1).unsqueeze(-1)

        task_num = self.task_data.task_list[tasks.squeeze(-1)]
        if isinstance(task_num, Task):
            task_type = torch.tensor([[4]])

        else:
            task_type = torch.tensor([task.type for task in task_num]).unsqueeze(-1)

        num_classes = 4
        one_hot_type = (task_type == torch.arange(num_classes).reshape(1, num_classes)).float()

        # add other embeddings

        descendant_features_norm = self.norm_desc_features[tasks].squeeze(1)

        # CP below task
        # cpl = torch.zeros(tasks.shape[0])
        # for i, task in enumerate(tasks):
        #     if self.task_to_CP[task] == 0:
        #         cpl[i] = CPAndWorkBelow(self.task_graph.task_list[task], self.n, durations_gpu)[0] / self.critic_path_duration
        #         self.task_to_CP[task] = cpl[i]
        #     else:
        #         cpl[i] = self.task_to_CP[task]
        # cpl = cpl.unsqueeze(-1)

        # add node type
        node_type = torch.ones(tasks.shape[0]) * self.cluster.node_types[self.current_proc]
        node_type = node_type.unsqueeze((-1))
        if sum(self.cluster.node_types == 1) == 0:
            min_ready_gpu = torch.FloatTensor([1]).repeat(tasks.shape[0]).unsqueeze((-1))
        else:
            min_ready_gpu = min(self.ready_proc[self.cluster.node_types == 1] - self.time)/self.max_duration_gpu
            min_ready_gpu = torch.FloatTensor([min_ready_gpu]).repeat(tasks.shape[0]).unsqueeze((-1))
        if sum(self.cluster.node_types == 0) == 0:
            min_ready_cpu = torch.FloatTensor([1]).repeat(tasks.shape[0]).unsqueeze((-1))
        else:
            min_ready_cpu = min(self.ready_proc[self.cluster.node_types == 0] - self.time) / self.max_duration_cpu
            min_ready_cpu = torch.FloatTensor([min_ready_cpu]).repeat(tasks.shape[0]).unsqueeze((-1))


        # if self.current_proc > 3:
            # print("what")

        # return (torch.cat((n_succ/10, n_pred/10, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time/10, cpl), dim=1),
        #         ready)

        # return (torch.cat((n_succ/10, n_pred/10, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time/10), dim=1),
        #         ready)

        # return (torch.cat((n_succ, n_pred, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time, cpl), dim=1),
        #         ready)

        return (torch.cat((n_succ, n_pred, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time,
                           descendant_features_norm, node_type, min_ready_gpu, min_ready_cpu), dim=1),
                ready)

        # return cpl, ready

        # return (torch.cat((one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time, cpl, node_type), dim=1),
        #         ready)

    # # Compute HEFT
    # def _compute_embeddings_heterogenous(self, tasks):


    def render(self):

        def color_task(task):
            colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            if task in self.running:
                time_proportion =1 - (self.ready_proc[self.running_task2proc[task]] - self.time)/\
                                  self.task_data.task_list[task].duration_cpu
                color_time = [1., time_proportion, time_proportion]
                return color_time
            elif task in self.ready_tasks:
                return colors[1]
            return colors[2]

        def color_processor(processor):
            if self.running[processor] == -1:
                return [0, 1, 0] if self.current_proc == processor else [0.7, 0.7, 0.7]
            else:
                time_proportion = (self.ready_proc[processor] - self.time) / \
                                  self.task_data.task_list[self.running[processor]].duration_cpu
            return [time_proportion, 0, 0]

        visible_graph, node_num = compute_sub_graph(self.task_data,
                                          torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                       self.ready_tasks)), dtype=torch.long),
                                          self.window)
        plt.figure(figsize=(8 , 8))
        plt.suptitle('time: {}'.format(self.time))
        plt.subplot(121)
        plt.box(on=None)
        visible_graph.render(root=list(self.running[self.running > -1]))
        # plt.title('time: {}'.format(self.time))
        # plt.show()

        plt.subplot(122)
        plt.box(on=None)
        graph = to_networkx(Data(visible_graph.x, visible_graph.edge_index.contiguous()))
        pos = graphviz_layout(graph, prog='dot', root=None)
        # pos = graphviz_layout(G, prog='tree')
        node_color = [color_task(task[0].item()) for task in node_num]
        # plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(graph, pos, node_color=node_color)
        nx.draw_networkx_edges(graph, pos)
        labels = {}
        for i, task in enumerate(node_num):
            if task[0].item() in self.ready_tasks:
                labels[i] = task[0].item()
        nx.draw_networkx_labels(graph, pos, labels, font_size=16)
        # plt.title('time: {}'.format(self.time))
        plt.show()

        # Cluster
        edges_list = [(u, v, {"cost": self.cluster.communication_cost[u, v]}) for u in range(self.p) for v in range(self.p) if u != v]
        colors = [color_processor(p) for p in range(self.p)]
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.cluster.node_types))))
        G.add_edges_from(edges_list)
        pos = graphviz_layout(G)
        node_labels = {}
        for i, node_type in enumerate(self.cluster.node_types):
            node_labels[i] = ["CPU", "GPU"][node_type]

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=1000)
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=16)
        plt.show()

    def visualize_schedule(self, figsize=(80, 30), fig_file=None, flip=False):

        def get_data(env):
            P = env.p
            Processed = env.processed
            for k, v in Processed.items():
                Processed[k] = [int(v[0]), int(v[1])]

            # makespan should be dicrete and durations should be discretized
            makespan = int(env.time)
            data = np.ones((P, makespan)) * (-1)
            data = data.astype(int)
            compl_data = [[] for _ in range(P)]
            for x, sched in Processed.items():
                tasktype = x[0]
                pr = sched[0]
                s_time = sched[1]
                e_time = s_time + Task(x).durations[env.cluster.node_types[pr]]
                data[pr, s_time:e_time] = tasktype
                if tasktype == 0:
                    compl_data[pr].insert(0, (x[1]))
                elif tasktype == 1:
                    compl_data[pr].insert(0, (x[1], x[2]))
                elif tasktype == 2:
                    compl_data[pr].insert(0, (x[1], x[2]))
                else:
                    compl_data[pr].insert(0, (x[1], x[2], x[3]))

            return data, compl_data

        def avg(a, b):
            return (a + b) / 2.0

        P = self.p
        data, compl_data = get_data(self)
        if flip:
            data = data[-1::-1, :]
            compl_data = compl_data[-1::-1]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)

        for y, row in enumerate(data):
            # for x, col in enumerate(row):
            x = 0
            i = 0
            indices_in_row = compl_data[y]
            while x < len(row):
                col = row[x]
                if col != -1:
                    shift = Task([col]).durations[self.cluster.node_types[y]]
                    indices = indices_in_row[i]
                else:
                    x = x + 1
                    continue
                x1 = [x, x + shift]
                y1 = np.array([y, y])
                y2 = y1 + 1
                if col == 0:
                    plt.fill_between(x1, y1, y2=y2, facecolor='green', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), 'C({})'.format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)

                if col == 1:
                    plt.fill_between(x1, y1, y2=y2, facecolor='red', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "S{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 2:
                    plt.fill_between(x1, y1, y2=y2, facecolor='orange', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "T{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 3:
                    plt.fill_between(x1, y1, y2=y2, facecolor='yellow', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "G{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                x = x + shift
                i = i + 1

        plt.ylim(P, 0)
        plt.xlim(-1e-3, data.shape[1] + 1e-3)
        plt.xticks(fontsize=50)
        if fig_file != None:
            plt.savefig(fig_file)
        return

    def export(self):
        se

class CholeskyTaskGraph(gym.Env):

    def __init__(self, n, node_types, window, noise=False):
        if isinstance(node_types, int):
            p = node_types
            node_types = np.ones(p)
        else:
            p = len(node_types)

        self.observation_space = Dict
        self.action_space = "Graph"

        self.noise = noise
        self.time = 0
        self.num_steps = 0
        self.n = n
        self.p = p
        self.window = window
        self.task_graph = compute_graph(n=n, noise=noise)
        self.task_data = TaskGraph(self.task_graph.x.clone(), self.task_graph.edge_index.clone(), self.task_graph.task_list.copy())
        # self.task_to_asap = {v: k for (k, v) in enumerate(self.task_data.task_list)}
        self.cluster = Cluster(node_types=node_types.astype(int), communication_cost=np.zeros((p, p)))
        self.running = -1 * np.ones(p)  # array of task number
        self.running_task2proc = {}
        self.ready_proc = np.zeros(p)  # for each processor, the time where it becomes available
        self.ready_tasks = []
        self.processed = {}
        self.current_proc = 0
        self.is_homogene = (np.mean(self.cluster.node_types) - 1) * np.mean(self.cluster.node_types) == 0

        self.critic_path_duration = sum(durations_gpu[:-2]) * (self.n - 1) + durations_gpu[0] # 158
        self.total_work_normalized = (n * durations_gpu[0] + n * (n - 1) / 2 * (durations_gpu[1] + durations_gpu[2]) + \
                          n * (n - 1) * (n - 2) / 6 * durations_gpu[3]) / p # 536 / p
        self.task_to_CP = np.zeros(len(self.task_graph.task_list))

    def reset(self):
        self.task_data = TaskGraph(self.task_graph.x.clone(), self.task_graph.edge_index.clone(), self.task_graph.task_list.copy())
        self.time = 0
        self.num_steps = 0
        self.running = -1 * np.ones(self.p).astype(int)
        self.running_task2proc = {}
        self.ready_proc = np.zeros(self.p)
        # self.ready_tasks.append(0)
        self.current_proc = 0

        self.ready_tasks = [0]
        self.processed = {}

        if self.noise > 0:
            for i in range(self.task_data.num_nodes):
                self.task_data.task_list[i].durations[1] = self.task_data.task_list[i].duration_gpu + np.random.normal(0, self.noise)


        return self._compute_state()

    def step(self, action, render_before=False, render_after=False):
        """
        first implementation, with only [-1, 0, ..., T] actions
        :param action: -1: does nothing. t: schedules t on the current available processor
        :return: next_state, reward, done, info
        """

        self.num_steps += 1

        self._find_available_proc()

        if action == -1:
            if len(self.running_task2proc) == 0:
                # the agent does nothing but every proc is available: we enforce an arbitrary action
                action = self.ready_tasks[0]

        self._choose_task_processor(action, self.current_proc)

        if render_before:
            self.render()

        done = self._go_to_next_action(action)

        if render_after:
            self.render()

        ref = max(self.critic_path_duration, self.total_work_normalized)
        reward = - (self.time - ref) / ref if done else 0

        info = {'episode': {'r': reward, 'length': self.num_steps, 'time': self.time}, 'bad_transition': False}

        return self._compute_state(), reward, done, info

    def _find_available_proc(self):
        while (self.current_proc < self.p) and (self.running[self.current_proc] > -1):
            self.current_proc += 1
        if self.current_proc == self.p:
            # no new proc available
            self.current_proc == 0
            self._forward_in_time()
        while (self.current_proc < self.p) and (self.running[self.current_proc] > -1):
            self.current_proc += 1

    def _forward_in_time(self):
        if len(self.ready_proc[self.ready_proc > self.time]) > 0:
            min_time = np.min(self.ready_proc[self.ready_proc > self.time])
        else:
            min_time = 0

        self.time = min_time

        self.ready_proc[self.ready_proc < self.time] = self.time

        tasks_finished = self.running[np.logical_and(self.ready_proc == self.time, self.running > -1)].copy()

        self.running[self.ready_proc == self.time] = -1
        for task in tasks_finished:
            del self.running_task2proc[task]

        # compute successors of finished tasks
        mask = isin(self.task_data.edge_index[0], torch.tensor(tasks_finished))
        list_succ = self.task_data.edge_index[1][mask]
        list_succ = torch.unique(list_succ)

        # remove nodes
        self.task_data.remove_edges(tasks_finished)

        # compute new available tasks
        new_ready_tasks = list_succ[torch.logical_not(isin(list_succ, self.task_data.edge_index[1, :]))]
        self.ready_tasks += new_ready_tasks.tolist()


        self.current_proc = np.argmin(self.running)

    def _go_to_next_action(self, previous_action):
        has_just_passed = self.is_homogene and previous_action == -1
        if has_just_passed:
            self._forward_in_time()
        elif previous_action == -1:
            self.current_proc += 1
        while len(self.ready_tasks) == 0:
            self._forward_in_time()
            if self._isdone():
                return True
        self._find_available_proc()
        return False

    def _choose_task_processor(self, action, processor):
        # assert action in self.ready_tasks

        if action != -1:
            self.ready_proc[processor] += self.task_data.task_list[action].durations[self.cluster.node_types[processor]]
            self.ready_tasks.remove(action)
            self.processed[self.task_data.task_list[action].barcode] = [processor, self.time]
            self.running_task2proc[action] = processor
            self.running[processor] = action

    def _compute_state(self):
        visible_graph, node_num = compute_sub_graph(self.task_data,
                                          torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                       self.ready_tasks)), dtype=torch.long),
                                          self.window)
        visible_graph.x, ready = self._compute_embeddings(node_num)
        return {'graph': visible_graph, 'node_num': node_num, 'ready': ready}

    def _remaining_time(self, running_tasks):
        return torch.tensor([self.ready_proc[self.running_task2proc[task.item()]] for task in running_tasks]) - self.time

    def _isdone(self):
        return (self.task_data.edge_index.shape[-1] == 0) and (len(self.running_task2proc) == 0)

    def _compute_embeddings(self, tasks):

        ready = isin(tasks, torch.tensor(self.ready_tasks)).float()
        running = isin(tasks, torch.tensor(self.running[self.running > -1])).squeeze(-1)

        remaining_time = torch.zeros(tasks.shape[0])
        remaining_time[running] = self._remaining_time(tasks[running].squeeze(-1))
        remaining_time = remaining_time.unsqueeze(-1)

        n_succ = torch.sum((tasks == self.task_data.edge_index[0]).float(), dim=1).unsqueeze(-1)
        n_pred = torch.sum((tasks == self.task_data.edge_index[1]).float(), dim=1).unsqueeze(-1)

        task_num = self.task_data.task_list[tasks.squeeze(-1)]
        if isinstance(task_num, Task):
            task_type = torch.tensor([[4]])

        else:
            task_type = torch.tensor([task.type for task in task_num]).unsqueeze(-1)

        num_classes = 5
        one_hot_type = (task_type == torch.arange(num_classes).reshape(1, num_classes)).float()

        # add other embeddings

        # CP below task
        cpl = torch.zeros(tasks.shape[0])
        for i, task in enumerate(tasks):
            if self.task_to_CP[task] == 0:
                cpl[i] = CPAndWorkBelow(self.task_graph.task_list[task], self.n, durations_gpu)[0] / self.critic_path_duration
                self.task_to_CP[task] = cpl[i]
            else:
                cpl[i] = self.task_to_CP[task]
        cpl = cpl.unsqueeze(-1)

        # add node type
        # node_type = torch.ones(tasks.shape[0]) * self.cluster.node_types[self.current_proc]
        # node_type = node_type.unsqueeze((-1))

        # if self.current_proc > 3:
        #     print("what")

        return (torch.cat((n_succ/10, n_pred/10, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time/10, cpl), dim=1),
                ready)

        # return (torch.cat((n_succ/10, n_pred/10, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time/10), dim=1),
        #         ready)

        # return (torch.cat((n_succ, n_pred, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time, cpl), dim=1),
        #         ready)

        # return (torch.cat((n_succ, n_pred, one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time), dim=1),
        #         ready)

        # return cpl, ready

        # return (torch.cat((one_hot_type, ready, running.unsqueeze(-1).float(), remaining_time, cpl, node_type), dim=1),
        #         ready)

    def render(self):

        def color_task(task):
            colors = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            if task in self.running:
                time_proportion =1 - (self.ready_proc[self.running_task2proc[task]] - self.time)/\
                                  self.task_data.task_list[task].duration_cpu
                color_time = [1., time_proportion, time_proportion]
                return color_time
            elif task in self.ready_tasks:
                return colors[1]
            return colors[2]

        def color_processor(processor):
            if self.running[processor] == -1:
                return [0, 1, 0] if self.current_proc == processor else [0.7, 0.7, 0.7]
            else:
                time_proportion = (self.ready_proc[processor] - self.time) / \
                                  self.task_data.task_list[self.running[processor]].duration_cpu
            return [time_proportion, 0, 0]

        visible_graph, node_num = compute_sub_graph(self.task_data,
                                          torch.tensor(np.concatenate((self.running[self.running > -1],
                                                                       self.ready_tasks)), dtype=torch.long),
                                          self.window)
        plt.figure(figsize=(8 , 8))
        plt.suptitle('time: {}'.format(self.time))
        plt.subplot(121)
        plt.box(on=None)
        visible_graph.render(root=list(self.running[self.running > -1]))
        # plt.title('time: {}'.format(self.time))
        # plt.show()

        plt.subplot(122)
        plt.box(on=None)
        graph = to_networkx(Data(visible_graph.x, visible_graph.edge_index.contiguous()))
        pos = graphviz_layout(graph, prog='dot', root=None)
        # pos = graphviz_layout(G, prog='tree')
        node_color = [color_task(task[0].item()) for task in node_num]
        # plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(graph, pos, node_color=node_color)
        nx.draw_networkx_edges(graph, pos)
        labels = {}
        for i, task in enumerate(node_num):
            if task[0].item() in self.ready_tasks:
                labels[i] = task[0].item()
        nx.draw_networkx_labels(graph, pos, labels, font_size=16)
        # plt.title('time: {}'.format(self.time))
        plt.show()

        # Cluster
        edges_list = [(u, v, {"cost": self.cluster.communication_cost[u, v]}) for u in range(self.p) for v in range(self.p) if u != v]
        colors = [color_processor(p) for p in range(self.p)]
        G = nx.Graph()
        G.add_nodes_from(list(range(len(self.cluster.node_types))))
        G.add_edges_from(edges_list)
        pos = graphviz_layout(G)
        node_labels = {}
        for i, node_type in enumerate(self.cluster.node_types):
            node_labels[i] = ["CPU", "GPU"][node_type]

        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos=pos, node_color=colors, node_size=1000)
        nx.draw_networkx_edges(G, pos=pos)
        nx.draw_networkx_edge_labels(G, pos=pos)
        nx.draw_networkx_labels(G, pos, node_labels, font_size=16)
        plt.show()

    def visualize_schedule(self, figsize=(80, 30), fig_file=None, flip=False):

        def get_data(env):
            P = env.p
            Processed = env.processed
            for k, v in Processed.items():
                Processed[k] = [int(v[0]), int(v[1])]

            # makespan should be dicrete and durations should be discretized
            makespan = int(env.time)
            data = np.ones((P, makespan)) * (-1)
            data = data.astype(int)
            compl_data = [[] for _ in range(P)]
            for x, sched in Processed.items():
                tasktype = x[0]
                pr = sched[0]
                s_time = sched[1]
                e_time = s_time + Task(x).durations[env.cluster.node_types[pr]]
                data[pr, s_time:e_time] = tasktype
                if tasktype == 0:
                    compl_data[pr].insert(0, (x[1]))
                elif tasktype == 1:
                    compl_data[pr].insert(0, (x[1], x[2]))
                elif tasktype == 2:
                    compl_data[pr].insert(0, (x[1], x[2]))
                else:
                    compl_data[pr].insert(0, (x[1], x[2], x[3]))

            return data, compl_data

        def avg(a, b):
            return (a + b) / 2.0

        P = self.p
        data, compl_data = get_data(self)
        if flip:
            data = data[-1::-1, :]
            compl_data = compl_data[-1::-1]

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_aspect(1)

        for y, row in enumerate(data):
            # for x, col in enumerate(row):
            x = 0
            i = 0
            indices_in_row = compl_data[y]
            while x < len(row):
                col = row[x]
                if col != -1:
                    shift = Task([col]).durations[self.cluster.node_types[y]]
                    indices = indices_in_row[i]
                else:
                    x = x + 1
                    continue
                x1 = [x, x + shift]
                y1 = np.array([y, y])
                y2 = y1 + 1
                if col == 0:
                    plt.fill_between(x1, y1, y2=y2, facecolor='green', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), 'C({})'.format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)

                if col == 1:
                    plt.fill_between(x1, y1, y2=y2, facecolor='red', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "S{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 2:
                    plt.fill_between(x1, y1, y2=y2, facecolor='orange', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "T{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                if col == 3:
                    plt.fill_between(x1, y1, y2=y2, facecolor='yellow', edgecolor='Black')
                    plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "G{}".format(indices),
                             horizontalalignment='center',
                             verticalalignment='center', fontsize=30)
                x = x + shift
                i = i + 1

        plt.ylim(P, 0)
        plt.xlim(-1e-3, data.shape[1] + 1e-3)
        plt.xticks(fontsize=50)
        if fig_file != None:
            plt.savefig(fig_file)
        return

if __name__ == "__main__":
    import torch
    from env import CholeskyTaskGraph
    import networkx as nx
    from torch_geometric.utils.convert import to_networkx

    import pydot
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_pydot import graphviz_layout
    import numpy as np

    from model import *

    # env = DAGEnv(2, np.array([1, 1, 0, 0]), 1)
    # obs = env.reset()
    # obs2 = env.step(0)
    # obs3 = env.step(1)
    # obs4 = env.step(2)
    # obs5 = env.step(3)
    # print('ok')
    # env.task_data.add_features_descendant()
#
#     env = CholeskyTaskGraph(8, np.array([1,1,1,1]), 1, noise=2)
# #     print(len(env.task_data.x))
#     obs = env.reset()
#     obs = env.reset()
#     done = False
#     env.render()
#     env.step(0, render_before=True, render_after=True)
#     env.render()
#     env.step(1, render_before=True, render_after=True)
#     env.render()
    # env.step(2, render_before=True, render_after=True)
    # # env.render()
    # env.step(-1, render_before=True, render_after=True)
    # # env.render()
    # env.step(-1, render_before=True, render_after=True)
    # # env.render()
#     while not done:
#         action = env.ready_tasks[0]
#         observation, reward, done, info = env.step(action)
#     print(reward)
#     print(env.time)

    # model = torch.load('/home/nathan/PycharmProjects/HPC/runs/Apr13_14-24-24_nathan-Latitude-7490/model.pth')
    # env = CholeskyTaskGraph(8, 4, 2)
    # print(len(env.task_data.x))
    # observation = env.reset()
    # done = False
    #
    # while not done:
    #     policy, value = model(observation)
    #     # action_raw = torch.multinomial(policy, 1).detach().cpu().numpy()
    #     action_raw = policy.argmax().detach().cpu().numpy()
    #     ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
    #     # action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw].detach().numpy()[0][0]
    #     action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw].detach().numpy()[0]
    #     observation, reward, done, info = env.step(action)
    # print(reward)
    # print(env.time)

# learn not to pass : -0.3670886075949367, 216.0

# model = torch.load('/home/nathan/PycharmProjects/HPC/runs/Apr25_20-07-37_nathan-Latitude-7490/model.pth')
# #
#
# # "/home/ngrinsztajn/HPC/runs/Apr20_05-12-44_chifflot-4.lille.grid5000.fr/model.pth"
# env = CholeskyTaskGraph(8, np.array([0, 0, 1, 1]), 1)
# print(len(env.task_data.x))
# observation = env.reset()
# done = False
#
# while not done:
#     policy, value = model(observation)
# #     new_policy = policy ** temp / ((policy ** temp).sum())
# #     action_raw = torch.multinomial(F.relu(new_policy), 1).detach().cpu().numpy()[0]
#     action_raw = policy.argmax().detach().cpu().numpy()
#     ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
# #     action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw].detach().numpy()[0][0]
#     action = -1 if action_raw == policy.shape[-1] - 1 else observation['node_num'][ready_nodes][action_raw].detach().numpy()[0]
#     observation, reward, done, info = env.step(action)
#
# env.visualize_schedule(figsize=(200, 100), fig_file='/home/nathan/PycharmProjects/HPC/img/test.pdf')
