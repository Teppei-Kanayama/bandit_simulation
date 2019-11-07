import numpy as np
import matplotlib.pyplot as plt

import argparse


class User(object):
    def __init__(self):
        self.open_probability = np.array([0.050, 0.0015, 0.0015])

    def is_open(self, send_timing):
        return np.random.binomial(1, self.open_probability[send_timing], 1)[0]


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
    def __init__(self):
        open_log = []
        super().__init__()

    def decide_send_timing(self):
        import pdb; pdb.set_trace()

    def recode_result(self, sent_timing, is_opened):
        self.distribution_recode[sent_timing] += 1
        self.open_recode[sent_timing] += is_opened
        self.open_log.append(is_opened)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-users', default=1000, type=int)
    parser.add_argument('--send-times', default=60, type=int)
    parser.add_argument('--model', default='egreedy', type=str)
    args = parser.parse_args()

    n_users = args.n_users
    send_times = args.send_times
    print(f'n_users: {n_users}, send_times: {send_times}')

    models = dict(
        egreedy=EpsilonGreedyMailer
        )

    users = [User() for _ in range(n_users)]
    mailers = [models[args.model]() for _ in range(n_users)]

    send_users_by_time = []
    open_count = np.zeros(send_times)
    for i in range(send_times):
        send_users = np.asarray([0, 0, 0])
        for j in range(n_users):
            user = users[j]
            mailer = mailers[j]
            send_timing = mailer.decide_send_timing()
            send_users[send_timing] += 1
            is_open = user.is_open(send_timing)
            mailer.recode_result(send_timing, is_open)
            open_count[i] += is_open
        send_users_by_time.append(send_users)

    # data = np.asarray(send_users_by_time)
    open_rate = open_count / n_users

    left = np.arange(len(open_rate))
    plt.plot(left, open_rate * 100)
    plt.xlabel("times")
    plt.ylabel("open rate (%)")
    plt.savefig('open_rate.png')

    # p = []
    # for i in range(3):
    #     height = np.array(data[:, i])
    #     left = np.arange(len(height))
    #     p.append(plt.plot(left, height))
    #
    # plt.legend((p[0][0], p[1][0], p[2][0]), (f"Suitable timebox (p={users[0].open_probability[0]})",
    #                                          f"Unsuitable timebox (p={users[0].open_probability[1]})",
    #                                          f"Unsuitable timebox (p={users[0].open_probability[2]})"),
    #            loc=2)
    # plt.xlabel("times")
    # plt.ylabel("the number of users")
    # plt.savefig('send_timing.png')


if __name__ == '__main__':
    main()
