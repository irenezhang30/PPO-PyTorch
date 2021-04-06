# PPO-LSTM
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from envs.acrobot_simulator import AcrobotSimulator
from envs.acrobot_simulator_po import AcrobotSimulator_po

import time
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 2
T_horizon = 20


class PPO(nn.Module):
    def __init__(self, action_dim=3, state_dim=4):
        super(PPO, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(state_dim, 64)
        self.lstm = nn.LSTM(64, 32)
        self.fc_pi = nn.Linear(32, action_dim)
        self.fc_v = nn.Linear(32, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def pi(self, x, hidden):
        import pdb; pdb.set_trace()
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=2)
        return prob, lstm_hidden

    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                              torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                              torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        s = torch.FloatTensor(s).to(device)
        r = r.float().to(device)
        a = a.to(device)
        s_prime = torch.FloatTensor(s_prime).to(device)
        done_mask = torch.FloatTensor(done_mask).to(device)
        prob_a = torch.FloatTensor(prob_a).to(device)


        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().cpu().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            pi, _ = self.pi(s, first_hidden)
            pi_a = pi.squeeze(1).gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            # import pdb; pdb.set_trace()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())


            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()


def main():
    exp_name = "lstmppo_acrobot_fo_discrete"
    env = AcrobotSimulator_po()
    model = PPO(action_dim=3, state_dim=5).to(device)

    # exp_name = "lstmppo_acrobot_po_discrete"
    # env = AcrobotSimulator_po(continuous_time=False)
    # model = PPO(action_dim=3, state_dim=2).to(device)

    # exp_name = "lstmppo_acrobot_po_continuous"
    # env = AcrobotSimulator_po(continuous_time=True)
    # model = PPO(action_dim=3, state_dim=2).to(device)

    # exp_name = "lstmppo_acrobot_fo_continuous"
    # env = AcrobotSimulator_po(continuous_time=True, partially_observable=False)
    # model = PPO(action_dim=3, state_dim=4).to(device)

    score = 0.0
    print_interval = 10
    results = []
    for n_epi in range(10000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float).to(device), torch.zeros([1, 1, 32], dtype=torch.float).to(device))
        s = env.reset()
        done = False

        while not done:
            for t in range(T_horizon):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float().to(device), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            results.append([n_epi, score / print_interval])
            np.save(exp_name, np.array(results))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()