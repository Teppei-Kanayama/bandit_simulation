import numpy as np


class User(object):

    def __init__(self):
        self.__open_probability = np.array([0.01, 0.001, 0.001])

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
    n_users = 10000
    send_days = 30

    users = [User() for _ in range(n_users)]
    mailers = [Mailer(epsilon=0.5) for _ in range(n_users)]

    for i in range(send_days):
        for j in range(n_users):
            user = users[j]
            mailer = mailers[j]
            send_timing = mailer.decide_send_timing()
            is_open = user.is_open(send_timing)
            mailer.recode_result(send_timing, is_open)

    cnt = 0
    for j in range(n_users):
        max_probability = mailers[j].get_max_probability()
        if len(max_probability) == 1 and max_probability[0] == 0:
            cnt += 1
    print(cnt)
    print(float(cnt) / n_users)


if __name__ == '__main__':
    main()
