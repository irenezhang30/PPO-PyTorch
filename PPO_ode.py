import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# import wandb
from envs.acrobot_simulator import AcrobotSimulator
from envs.acrobot_simulator_po import AcrobotSimulator_po

from torchdiffeq import odeint as odeint
import time
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# wandb.init(project="ppo-ode")
# config = wandb.config

# Hyperparameters
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.2
# K_epoch = 4
# T_horizon = 300
K_epoch = 2
T_horizon = 20


ode_method = "euler"
atol = rtol = 1e-2
latent_dim = 32
update_timestep = 1
alpha=0.05

# config.learning_rate = learning_rate
# config.gamma = gamma
# config.lmbda = lmbda
# config.eps_clip = eps_clip
# config.K_epoch = K_epoch
# config.T_horizon = T_horizon
# config.ode_method = ode_method
# config.atol = atol
# config.rtol = rtol
# config.latent_dim = latent_dim
# config.update_timestep = update_timestep
# config.alpha = alpha

class ODEFunc(nn.Module):
    def __init__(self, input_dim, ode_dim=20):
        super(ODEFunc, self).__init__()
        self.l1 = nn.Linear(input_dim, ode_dim)
        self.l2 = nn.Linear(ode_dim, ode_dim)
        self.l3 = nn.Linear(ode_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, t, x):
        """
        Perform one step in solving ODE.
        """
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

class PPO(nn.Module):
    def __init__(self, action_dim=3, state_dim=4, latent_dim=64):
        super(PPO, self).__init__()
        self.data = []
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 64)
        self.lstm = nn.LSTM(64, latent_dim)
        self.fc_pi = nn.Linear(latent_dim, action_dim)
        self.fc_v = nn.Linear(latent_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        self.fc_xi = nn.Linear(latent_dim * 2 + action_dim, latent_dim * 2)
        self.ode_func = ODEFunc(latent_dim * 2, ode_dim=20)


    def xi(self, hidden_tild, act):
#         import pdb; pdb.set_trace()
        act = torch.tensor(act).to(device).long()

        act = F.one_hot(act, num_classes=self.action_dim)
        input = torch.cat((hidden_tild[0].flatten(), hidden_tild[1].flatten(), act), dim=-1) # cell state, hidden_state
        hidden_prime = self.fc_xi(input)
        return hidden_prime

    def shared(self, obs, hidden):
        if len(obs.shape) > 1:
            outs = []
            for o in obs:
                ts = o[...,-3:-1]
                act = o[...,-1]
                ob = o[...,:-3]
                if ts.sum():
                    z_prime = self.xi(hidden, act)
                    z = odeint(self.ode_func, z_prime, ts, rtol=rtol, atol=atol, method=ode_method)[-1]
                    z = torch.split(z.reshape(1, 1, -1), split_size_or_sections=32, dim=-1)
                else:
                    z = hidden

                ob = self.fc1(ob)
                ob = ob.view(-1, 1, 64)

                out, hidden = self.lstm(ob, z)
                outs.append(out)
            outs = torch.stack(outs).squeeze(1)

        else:
            ts = obs[...,-3:-1]
            act = obs[...,-1]
            ob = obs[...,:-3]
            if ts.sum():
                z_prime = self.xi(hidden, act)
                z = odeint(self.ode_func, z_prime, ts, rtol=rtol, atol=atol, method=ode_method)[-1]
                z = torch.split(z.reshape(1, 1, -1), split_size_or_sections=32, dim=-1)
            else:
                z = hidden
            # print(obs.shape, 'obs shape')
            # print(ob.shape, 'ob shape')
            ob = self.fc1(ob)
            ob = ob.view(-1, 1, 64)

            outs, hidden = self.lstm(ob, z)


        return outs, hidden


    def pi(self, obs, hidden):
        outs, hidden = self.shared(obs, hidden)
        outs = self.fc_pi(outs)
        prob = F.softmax(outs, dim=2)
        return prob, hidden

    def v(self, obs, hidden):
        outs, _ = self.shared(obs, hidden)
        v = self.fc_v(outs)
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
            pi, _ = self.pi(s, first_hidden)
            m = Categorical(pi)
            entropy = m.entropy()
            td_target = r + gamma * v_prime * done_mask #+ alpha * entropy
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



            pi_a = pi.squeeze(1).gather(1, a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            # import pdb; pdb.set_trace()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())


            # self.optimizer.zero_grad()
            loss_mean = loss.mean()
            loss_mean.backward(retain_graph=True)
#             wandb.log({"Mean Loss": loss_mean})
            # self.optimizer.step()
            self.optimizer.step()
            self.optimizer.zero_grad()


def main():
    exp_name = "lstmppo_acrobot_po_continuous"
    env = AcrobotSimulator_po()
    model = PPO(action_dim=3, state_dim=2, latent_dim=latent_dim).to(device)

#     wandb.watch(model)

    score = 0.0
    print_interval = 10
    results = []
    timestep = 0
    for n_epi in range(1000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float).to(device), torch.zeros([1, 1, 32], dtype=torch.float).to(device))
        s = env.reset()
        done = False
        t = 0
        timestep
        while not done:
            for t in range(T_horizon):

                timestep += 1
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float().to(device), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
#                 prev_t = t
#                 t = prev_t + info["dt"]
#                 z_prime = model.xi(z_tild, a)
#                 time = torch.FloatTensor([prev_t, t]).to(device)
#                 z = odeint(model.ode_func, z_prime, time, rtol=rtol, atol=atol, method=ode_method)[-1]
#                 z = torch.split(z.reshape(1, 1, -1), split_size_or_sections=32, dim=-1)


                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime
                # wandb.log({'Rewards': r})
                score += r
                # print(r, 'r')
                # if timestep % update_timestep == 0:

                if done:
                    break
            model.train_net()
        print("WAT")
        if n_epi % 10 == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            results.append([n_epi, score / print_interval])
#             wandb_score = score / print_interval
#             wandb.log({'Score': wandb_score})
            np.save(exp_name, np.array(results))
            score = 0.0
            print(torch.norm(h_out[0]))

    env.close()

if __name__ == '__main__':
    main()
