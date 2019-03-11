
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
        mse = 50    # mean square error
        min_mse = 50
        n = len(labels)
        epoch = 1
        counter = 0
        self.bias = bias
        pos_range = 0.00001

        while mse > stopping_condition and counter < 10:
            error = 0  # (di-yi)^2
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

            for i in range(0, n):
                yi = bias * self.weights[0] + self.weights[1] * x1[i] + self.weights[2] * x2[i]
                e = labels[i] - yi
                error += 1/2*(e*e)   # calculate mean square error
            mse = 1/n * error

            condition_var = mse - min_mse
            if -pos_range > condition_var > pos_range:   # -range>condition>range
                min_mse = mse
                counter = 0
            else:           #  if condition_var < pos_range and condition_var > - pos_range: # condition
                # too little change
                counter += 1
            print("end of epoch:", epoch, "mse =", mse, "counter:", counter)
            epoch += 1
        print("mse:", mse)
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

