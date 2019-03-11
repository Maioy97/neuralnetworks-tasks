
import numpy as np


def signum(num):
    if num > 0:
        return 1

    else:
        return -1


class tr:
    weight = []
    bias = 1

    def __init__(self):
        self.weights = np.random.rand(3, )
        print(self.weights)

    def train(self, labels, stopping_condition, bias, x1, x2, LR):
        error = 0   # (di-yi)^2
        mse = 50    # mean square error
        n = len(labels)
        while mse > stopping_condition:
            for j in range(0, n):
                yi = bias*self.weights[0]+self.weights[1]*x1[j]+self.weights[2]*x2[j]
                loss = (labels[j] - yi)
                # if loss > 0:
                w1 = self.weights[0] + LR * loss * self.bias
                w2 = self.weights[1] + LR * loss * x1[j]
                w3 = self.weights[2] + LR * loss * x2[j]
                self.weights[0] = w1
                self.weights[1] = w2
                self.weights[2] = w3

            for label in labels:
                yi = bias * self.weights[0] + self.weights[1] * x1[j] + self.weights[2] * x2[j]
                e = yi - label
                error += (e*e)   # calculate mean square error
            mse = 1/n * error
        return self.weights

    def test(self, labels, x1, x2):
        miss = 0
        conf_mat = np.zeros((2, 3))
        for i in range(0, len(labels)):
            y = (self.bias * self.weights[0] + x1[i] * self.weights[1] + x2[i] * self.weights[2])
            yi = signum(y)
            if yi != labels[i]:
                miss += 1

            if labels[i] == 1 and yi == 1:
                conf_mat[0][0] += 1
            elif labels[i] == -1 and yi == -1:
                conf_mat[1][1] += 1
            elif labels[i] == -1 and yi == 1:
                conf_mat[1][0] += 1
            else:
                # label = 1 y = -1
                conf_mat[0][1] += 1

        conf_mat[0][2] = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
        conf_mat[1][2] = conf_mat[1][1]/(conf_mat[1][0]+conf_mat[1][1])
        total_error = miss/len(labels)
        return conf_mat, total_error

