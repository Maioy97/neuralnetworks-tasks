
import numpy as np


def signum(num):
    if num > 0:
        return 1

    else:
        return -1

class tr:
    def __init__(self):
        weights = np.random.rand(3, )
        weights[0] = 1
        bias = 1

    def train(self, labels, epo, bias, x1, x2, LR):

        for i in range(0, epo):
            for j in range(0, len(labels)):
                yi = signum((bias*self.weights[0]+x1[j]*self.weights[1]+x2[j]*self.weights[2]))
                loss = labels[j]-yi
                if loss > 0:
                    w1 = self.weights[0] + LR * loss * self.bias
                    w2 = self.weights[1] + LR * loss * x1[j]
                    w3 = self.weights[2] + LR * loss * x2[j]
                    self.weights[0] = w1
                    self.weights[1] = w2
                    self.weights[2] = w3
                return self.weights
    def test(self,labels,x1,x2):
        miss=0
        conmat=np.empty((2,3))
        for i in range(0,len(labels)):
            yi = signum((bias * self.weights[0] + x1[i] * self.weights[1] + x2[i] * self.weights[2]))
            if yi != labels[i]:
                miss +=1
            if yi==1 & labels[i]==1:
                conmat[0][0]+=1
            elif yi==1 & labels[i]==-1:
                conmat[0][1]+=1
            elif yi==-1 & labels[i]==-1:
                conmat[1][0]+=1
            else :
                conmat[1][1]+=1
        conmat[0][2]=conmat[0][0]/(conmat[0][0]+conmat[0][1])
        conmat[1][2]=conmat[1][0]/(conmat[1][0]+conmat[1][1])
        TotalError=miss/len(labels)
        return conmat,TotalError

