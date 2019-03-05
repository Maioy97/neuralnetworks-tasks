import random
import numpy as np

def getclass(str):
    if str=="Iris-setosa" :
        return 0
    elif str == "Iris-versicolor" :
        return 1
    else :
        return -1

def signum(num):
    if num>0 :
        return 1
    elif num ==0:
        return 0
    else :
        return -1

#train would take 30 for the count and 20 for test
#n is the class num 0 ,1 or 2
def fetchingdata (n,count):
    x1=[], x2=[], x3=[], x4=[] ,data=[]
    with open('IrisData') as f:
        lines =f.readlines()
        for i in range(n*50,n*50+count):
            data2=lines[i]
            data1=data2.split(',')
            x1.append(data1[0])
            x2.append(data1[1])
            x3.append(data1[2])
            x4.append(data1[3])
            data.append(x1)
            data.append(x2)
            data.append(x3)
            data.append(x4)
            data.append(getclass(data1[4]))
        np.reshape(data,(5,count))
    return data , x1,x2,x3,x4
def getweight (labels,epo,bias,x1 ,x2,LR):

    weig1 = np.random.rand(len(labels), 3)
    for i in range(0,len(labels)):
        weig1[i][0]=1
    for i in range(0,epo):
        for j in range(0,len(labels)):
            yi=signum( (bias*weig1[j][0]+x1[j]* weig1[j][1]+x2[j]*weig1[j][2]))
            loss=getclass(labels[j])-yi
            if loss>0:
                w1=weig1[j][0]+LR *loss *bias
                w2=weig1[j][1]+LR *loss *x1[j]
                w3=weig1[j][2]+LR *loss *x2[j]
                weig1[j][0]=w1
                weig1[j][1] = w2
                weig1[j][2] = w3
    return weig1
