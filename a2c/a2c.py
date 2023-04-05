import time
import numpy as np
import os
from copy import deepcopy
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CyclicLR, LambdaLR
from collections import deque


def make_seed(seed):
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)


# use_cuda = torch.cuda.is_available()
use_cuda = False
if use_cuda:
    device = torch.device('cuda')
    print("using GPU")
else:
    device = torch.device('cpu')
    print("using CPU")


class A2C:
    def __init__(self, config, env, model, writer=None):
        self.config = config
        self.env = env
        make_seed(config['seed'])
        # self.env.seed(config['seed'])
        self.gamma = config['gamma']
        self.entropy_cost = config["entropy_coef"]
        self.noise = config['noise'] if 'noise' in config.keys() else config['env_settings']['noise']
        self.random_id = str(np.random.randint(0, 9, 10)).replace(' ', '_')
        # Our network
        # self.network = model(**config["network_parameters"]).to(device)
        # model = model(11)

        # if 'network_parameters' in config.keys():
        #     model = model(config['network_parameters']["input_dim"])
        # else:
        #     model = model(config['input_dim'])

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     model = torch.nn.DataParallel(model)

        self.network = model.to(device)
        if config["model_path"] is not None and config["model_path"] != 'none':
            # self.network.load_state_dict(torch.load(config['model_path']))
            self.network = torch.load(config['model_path'])
        # Their optimizers
        if config['optimizer'] == "sgd":
            self.optimizer = optim.SGD(self.network.parameters(), config['lr'])
        elif config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=config['lr'])
        else:
            self.optimizer = optim.RMSprop(self.network.parameters(), config['lr'], eps=config['eps'])
        self.writer = writer

        if config['scheduler'] == 'cyclic':
            ratio = config['sched_ratio']
            self.scheduler = CyclicLR(self.optimizer, base_lr=config['lr']/ratio, max_lr=config['lr']*ratio,
                                      step_size_up=config['step_up'])
        elif config['scheduler'] == 'lambda':
            lambda2 = lambda epoch: 0.99 ** epoch
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=[lambda2])
        else:
            self.scheduler = None

    # Hint: use it during training_batch
    def _returns_advantages(self, rewards, dones, values, next_value):
        """Returns the cumulative discounted rewards at each time step

        Parameters
        ----------
        rewards : array
            An array of shape (batch_size,) containing the rewards given by the env
        dones : array
            An array of shape (batch_size,) containing the done bool indicator given by the env
        values : array
            An array of shape (batch_size,) containing the values given by the value network
        next_value : float
            The value of the next state given by the value network

        Returns
        -------
        returns : array
            The cumulative discounted rewards
        advantages : array
            The advantages
        """

        returns = np.append(np.zeros_like(rewards), [next_value], axis=0)

        for t in reversed(range(rewards.shape[0])):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - dones[t])

        returns = returns[:-1]
        advantages = returns - values
        return returns, advantages

    def training_batch(self):
        """Perform a training by batch

        Parameters
        ----------
        steps : int
            Number of steps
        batch_size : int
            The size of a batch
        """
        start = time.time()
        reward_log = deque(maxlen=10)
        time_log = deque(maxlen=10)

        batch_size = self.config['trajectory_length']

        actions = np.empty((batch_size,), dtype=np.int)
        dones = np.empty((batch_size,), dtype=np.bool)
        rewards, values = np.empty((2, batch_size), dtype=np.float)
        observations = []
        observation = self.env.reset()
        observation['graph'] = observation['graph'].to(device)
        rewards_test = []
        best_reward_mean = -1000

        n_step = 0
        log_ratio = 0
        best_time = 100000

        while n_step < self.config['num_env_steps']:
            # Lets collect one batch

            probs = torch.zeros(batch_size, dtype=torch.float, device=device)
            vals = torch.zeros(batch_size, dtype=torch.float, device=device)
            probs_entropy = torch.zeros(batch_size, dtype=torch.float, device=device)

            for i in range(batch_size):
                observations.append(observation['graph'])
                policy, value = self.network(observation)
                values[i] = value.detach().cpu().numpy()
                vals[i] = value
                probs_entropy[i] = - (policy * policy.log()).sum(-1)
                try:
                    action_raw = torch.multinomial(policy, 1).detach().cpu().numpy()
                except:
                    print("chelou")
                probs[i] = policy[action_raw]
                ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
                actions[i] = -1 if action_raw == policy.shape[-1] -1 else observation['node_num'][ready_nodes][action_raw]
                observation, rewards[i], dones[i], info = self.env.step(actions[i])
                observation['graph'] = observation['graph'].to(device)
                n_step += 1

                if dones[i]:
                    observation = self.env.reset()
                    observation['graph'] = observation['graph'].to(device)
                    reward_log.append(rewards[i])
                    time_log.append(info['episode']['time'])

            # If our episode didn't end on the last step we need to compute the value for the last state
            if dones[i] and not info['bad_transition']:
                next_value = 0
            else:
                next_value = self.network(observation)[1].detach().cpu().numpy()[0]

            # Update episode_count

            # Compute returns and advantages
            returns, advantages = self._returns_advantages(rewards, dones, values, next_value)

            # TO DO: use rewards for train rewards

            # Learning step !
            loss_value, loss_actor, loss_entropy = self.optimize_model(observations, actions, probs, probs_entropy, vals, returns, advantages, step=n_step)
            if self.writer is not None and log_ratio * self.config['log_interval'] < n_step:
                print('saving model if better than the previous one')
                log_ratio += 1
                self.writer.add_scalar('reward', np.mean(reward_log), n_step)
                self.writer.add_scalar('time', np.mean(time_log), n_step)
                self.writer.add_scalar('critic_loss', loss_value, n_step)
                self.writer.add_scalar('actor_loss', loss_actor, n_step)
                self.writer.add_scalar('entropy', loss_entropy, n_step)

                if self.noise > 0:
                    current_time = np.mean([self.evaluate(), self.evaluate(), self.evaluate()])
                else:
                    current_time = self.evaluate()
                self.writer.add_scalar('test time', current_time, n_step)
                print("comparing current time: {} with previous best: {}".format(current_time, best_time))
                if current_time < best_time:
                    print("saving model")
                    string_save = os.path.join(str(self.writer.get_logdir()), 'model{}.pth'.format(self.random_id))
                    torch.save(self.network, string_save)
                    best_time = current_time
                    # current_tab = []
                    # for _ in range(10):
                    #     current_tab.append(self.evaluate())
                    # current_mean = np.mean(current_tab)

            if len(reward_log) > 0:
                end = time.time()
                print('step ', n_step, '\n reward: ', np.mean(reward_log))
                print('FPS: ', int(n_step / (end - start)))

            if self.scheduler is not None:
                print(self.scheduler.get_lr())
                self.scheduler.step(int(n_step/batch_size))

        self.network = torch.load(string_save)
        results_last_model = []
        if self.noise > 0:
            for _ in range(5):
                results_last_model.append(self.evaluate())
        else:
            results_last_model.append(self.evaluate())
        torch.save(self.network, os.path.join(str(self.writer.get_logdir()), 'model_{}.pth'.format(str(np.mean(results_last_model)))))
        os.remove(string_save)
        return best_time, np.mean(results_last_model)

        #     # Test it every "evaluate_every" steps
        #     if n_step > self.config['evaluate_every'] * (log_ratio + 1):
        #         rewards_test.append(np.array([self.evaluate() for _ in range(50)]))
        #         print(
        #             f"""Steps {n_step}/{self.config['num_env_steps']}: Mean rewards: {round(rewards_test[-1].mean(), 2)}, Std: {round(rewards_test[-1].std(), 2)}""")
        #         if self.writer:
        #             self.writer.add_scalar('mean_reward', round(rewards_test[-1].mean(), 2), n_step)
        #
        #         if rewards_test[-1].mean() >= best_reward_mean:
        #             best_reward_mean = rewards_test[-1].mean()
        #             str_file = str(self.writer.get_logdir()).split('/')[1]
        #             torch.save(self.network.state_dict(), os.path.join(str(self.writer.get_logdir()),
        #                                                                'model.pth'))
        #
        #         observation = self.env.reset()
        #
        # # Plotting
        # r = pd.DataFrame(
        #     (itertools.chain(*(itertools.product([i], rewards_test[i]) for i in range(len(rewards_test))))),
        #     columns=['Steps', 'Reward'])
        # sns.lineplot(x="steps", y="Reward", data=r, ci='sd');
        #
        # print(f'The trainnig was done over a total of {n_step} steps')

    def optimize_model(self, observations, actions, probs, entropies, vals, returns, advantages, step=None):
        # actions = F.one_hot(torch.tensor(actions, device=device), self.env.action_space.n)
        returns = torch.tensor(returns[:, None], dtype=torch.float, device=device)
        advantages = torch.tensor(advantages, dtype=torch.float, device=device)
        # observations = torch.tensor(observations, dtype=torch.float, device=device)
        # observations = Batch().from_data_list(observations)


        # reset
        # self.network_optimizer.zero_grad()
        # policies, values = self.network(observations)

        # MSE for the values
        loss_value = 1 * F.mse_loss(vals.unsqueeze(-1), returns)
        if self.writer:
            self.writer.add_scalar('critic_loss', loss_value.data.item(), step)

        # Actor loss
        # loss_policy = ((probs.log()).sum(-1) * advantages).mean()
        loss_policy = ((probs.log()) * advantages).mean()
        loss_entropy = entropies.mean()
        loss_actor = - loss_policy - self.entropy_cost * loss_entropy
        if self.writer:
            self.writer.add_scalar('actor_loss', loss_actor.data.item(), step)

        total_loss = self.config["loss_ratio"] * loss_value + loss_actor
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 10)
        self.optimizer.step()
        return loss_value.data.item(), loss_actor.data.item(), loss_entropy.data.item()

    def evaluate(self, render=False):
        env = self.monitor_env if render else deepcopy(self.env)

        observation = env.reset()
        done = False

        while not done:
            observation['graph'] = observation['graph'].to(device)
            policy, value = self.network(observation)
            # action_raw = torch.multinomial(policy, 1).detach().cpu().numpy()
            action_raw = policy.argmax().detach().cpu().numpy()
            ready_nodes = observation['ready'].squeeze(1).to(torch.bool)
            action = -1 if action_raw == policy.shape[-1] - 1 else \
                observation['node_num'][ready_nodes][action_raw].detach().numpy()[0]
            try :
                observation, reward, done, info = env.step(action)
            except KeyError:
                print(chelou)
        return env.time

