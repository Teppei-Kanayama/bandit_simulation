import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import argparse


class User(object):
    def __init__(self):
        self.open_probability = np.array([0.030, 0.0015, 0.0015])  # after ab test
        #self.open_probability = np.array([0.050, 0.0015, 0.0015])  # previous
        self.open_count = 0

    def is_open(self, send_timing):
        is_open = np.random.binomial(1, self.open_probability[send_timing], 1)[0]
        self.open_count += is_open
        return is_open

    def get_prob(self, send_timing):
        return self.open_probability[send_timing]


class Mailer(object):
    def __init__(self):
        self.distribution_recode = np.array([1, 1, 1])
        self.open_recode = np.array([0, 0, 0])

    def recode_result(self, sent_timing, is_opened):
        self.distribution_recode[sent_timing] += 1
        self.open_recode[sent_timing] += is_opened

    def get_max_probability(self):
        estimated_open_probability = self.open_recode.astype(np.float32) / self.distribution_recode.astype(np.float32)
        max_probability = np.where(estimated_open_probability == estimated_open_probability.max())[0]
        return max_probability


class EpsilonGreedyMailer(Mailer):
    def __init__(self, epsilon=0.3):
        self.epsilon = epsilon
        super().__init__()

    def decide_send_timing(self):
        is_random = np.random.binomial(1, self.epsilon, 1)
        if is_random:
            return np.random.randint(3)
        else:
            max_probability = self.get_max_probability()
            return np.random.choice(max_probability)


class OneStateMarcovMailer(Mailer):
    def __init__(self):
        self.previous_distribution = np.random.randint(3)
        self.is_opened = False
        super().__init__()

    def decide_send_timing(self):
        if self.is_opened:
            return self.previous_distribution
        else:
            candidate = [0, 1, 2]
            candidate.remove(self.previous_distribution)
            return np.random.choice(candidate)


class DiscountModel(Mailer):
    def __init__(self, discount_rate=0.99):
        self.open_log = []
        self.distribution_log = []
        self.discount_rate = discount_rate
        super().__init__()

    def decide_send_timing(self):
        if not (1 in self.open_log):
            return np.random.randint(3)
        else:
            df = pd.DataFrame(dict(open=self.open_log, timing=self.distribution_log))
            df['exploration_probability'] = [(1 - self.discount_rate ** i) for i in range(df.shape[0])][::-1]

            prob = df[df['open'] == 1]['exploration_probability'].iloc[0]
            timing = df[df['open'] == 1]['timing'].iloc[0]

            is_random = np.random.binomial(1, prob, 1)
            if is_random:
                return np.random.randint(3)
            else:
                return timing

    def recode_result(self, sent_timing, is_opened):
        self.distribution_recode[sent_timing] += 1
        self.open_recode[sent_timing] += is_opened
        self.open_log.append(is_opened)
        self.distribution_log.append(sent_timing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-users', default=1000, type=int)
    parser.add_argument('--send-times', default=60, type=int)
    parser.add_argument('--model', default='discount', type=str)
    args = parser.parse_args()

    n_users = args.n_users
    send_times = args.send_times
    print(f'n_users: {n_users}, send_times: {send_times}')

    models = dict(
        egreedy=EpsilonGreedyMailer,
        discount=DiscountModel
        )

    users = [User() for _ in range(n_users)]
    mailers = [models[args.model]() for _ in range(n_users)]

    send_users_by_time = []
    open_count = np.zeros(send_times)
    sum_open_prob = np.zeros(send_times)
    for i in range(send_times):
        print(i)
        send_users = np.asarray([0, 0, 0])
        for j in range(n_users):
            user = users[j]
            mailer = mailers[j]
            send_timing = mailer.decide_send_timing()
            send_users[send_timing] += 1
            is_open = user.is_open(send_timing)
            mailer.recode_result(send_timing, is_open)
            open_count[i] += is_open
            sum_open_prob[i] += user.get_prob(send_timing)
        send_users_by_time.append(send_users)

    never_open_user_count = 0
    for user in users:
        if user.open_count == 0:
            never_open_user_count += 1
    print(never_open_user_count)
    print(never_open_user_count / n_users)

    # open_rate = open_count / n_users
    open_prob_rate = sum_open_prob / n_users

    df = pd.DataFrame(dict(open_prob=open_prob_rate))
    df.to_csv('open_rate_old.csv')

    # left = np.arange(len(open_prob_rate))
    # plt.plot(left, open_prob_rate * 100)
    # plt.xlabel("times")
    # plt.savefig('open_rate.png')


if __name__ == '__main__':
    main()
