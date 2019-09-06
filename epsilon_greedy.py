import numpy as np
import matplotlib.pyplot as plt


class User(object):

    def __init__(self):
        self.__open_probability = np.array([0.010, 0.0015, 0.0015])

    def is_open(self, send_timing):
        return np.random.binomial(1, self.__open_probability[send_timing], 1)[0]


class Mailer(object):

    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.distribution_recode = np.array([1, 1, 1])
        self.open_recode = np.array([0, 0, 0])

    def decide_send_timing(self):
        is_random = np.random.binomial(1, self.epsilon, 1)
        if is_random:
            return np.random.randint(3)
        else:
            max_probability = self.get_max_probability()
            return np.random.choice(max_probability)

    def recode_result(self, sent_timing, is_opened):
        self.distribution_recode[sent_timing] += 1
        self.open_recode[sent_timing] += is_opened

    def print_result(self):
        estimated_open_probability = self.open_recode.astype(np.float32) / self.distribution_recode.astype(np.float32)
        print(estimated_open_probability)

    def get_max_probability(self):
        estimated_open_probability = self.open_recode.astype(np.float32) / self.distribution_recode.astype(np.float32)
        max_probability = np.where(estimated_open_probability == estimated_open_probability.max())[0]
        return max_probability


def main():
    n_users = 1000
    send_days = 60

    users = [User() for _ in range(n_users)]
    mailers = [Mailer(epsilon=0.1) for _ in range(n_users)]

    send_users_by_time = []
    for i in range(send_days):
        send_users = np.asarray([0, 0, 0])
        for j in range(n_users):
            user = users[j]
            mailer = mailers[j]
            send_timing = mailer.decide_send_timing()
            send_users[send_timing] += 1
            is_open = user.is_open(send_timing)
            mailer.recode_result(send_timing, is_open)
        send_users_by_time.append(send_users)

    data = np.asarray(send_users_by_time)
    p = []
    for i in range(3):
        height = np.array(data[:, i])
        left = np.arange(len(height))
        p.append(plt.plot(left, height))

    plt.legend((p[0][0], p[1][0], p[2][0]), ("Suitable timebox (p=0.015)", "Unsuitable timebox (p=0.0015)", "Unsuitable timebox (p=0.0015)"), loc=2)
    plt.xlabel("days")
    plt.ylabel("the number of users")
    plt.savefig('send_timing.png')


class Mailer2(object):

    def __init__(self):
        self.previous_distribution = np.random.randint(3)
        self.is_opened = False

    def decide_send_timing(self):
        if self.is_opened:
            return self.previous_distribution
        else:
            candidate = [0, 1, 2]
            candidate.remove(self.previous_distribution)
            return np.random.choice(candidate)

    def recode_result(self, sent_timing, is_opened):
        self.previous_distribution = sent_timing
        self.is_opened += is_opened


def main2():
    n_users = 1000
    send_days = 60

    users = [User() for _ in range(n_users)]
    mailers = [Mailer2() for _ in range(n_users)]

    send_users_by_time = []
    for i in range(send_days):
        send_users = np.asarray([0, 0, 0])
        for j in range(n_users):
            user = users[j]
            mailer = mailers[j]
            send_timing = mailer.decide_send_timing()
            send_users[send_timing] += 1
            is_open = user.is_open(send_timing)
            mailer.recode_result(send_timing, is_open)
        send_users_by_time.append(send_users)
        print(send_users)

    data = np.asarray(send_users_by_time)
    p = []
    for i in range(3):
        height = np.array(data[:, i])
        left = np.arange(len(height))
        p.append(plt.plot(left, height))

    plt.legend((p[0][0], p[1][0], p[2][0]), ("Suitable timebox (p=0.015)", "Unsuitable timebox (p=0.0015)", "Unsuitable timebox (p=0.0015)"), loc=2)
    plt.xlabel("days")
    plt.ylabel("the number of users")
    plt.savefig('send_timing_1state_markov.png')


if __name__ == '__main__':
    main()
    main2()
