import random
import numpy as np

class tr:
    def __init__(self):
        weights = np.random.rand(3, )
        weights[0] = 1


    def train(self,labels, epo, bias, x1, x2, LR):

        for i in range(0, epo):
            for j in range(0, len(labels)):
                yi = signum((bias*weights[j][0]+x1[j]*weights[j][1]+x2[j]*weights[j][2]))
                loss = labels[j]-yi
                if loss > 0:
                    w1 = weights[0]+LR * loss * bias
                    w2 = weights[1]+LR * loss * x1[j]
                    w3 = weights[2]+LR * loss * x2[j]
                    weights[0] = w1
                    weights[1] = w2
                    weights[2] = w3
        return weights
    def test(self,labels,x1,x2):
        miss=0
        for i in range(0,len(labels)):
            yi = signum((bias * weights[0] + x1[i] * weights[1] + x2[i] * weights[2]))
            if yi != labels[i]:
                miss +=1
            