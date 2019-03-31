import numpy as np


class BackPropagation:
    weight = {}  # []
    bias = 1
    layer_error = {}
    net_Input = {}
    num_layers = 1
    learning_rate = .1
    function_number = 0
    layers_shape = []
    output_y = []
    def __init__(self, num_layers, layer_shapes, bias):
        layer_shapes.append(3)
        self.layers_shape=layer_shapes
        self.bias = bias
        self.num_layers = num_layers
        # should be changed to receive the desired network shape and initialise the weight values in it
        if bias == 0:
            self.weights[0] = np.random.rand(4, layer_shapes[0])  # loop over layer matrix s and initialise them
            for i in range(0, num_layers):
                self.weight[i+1] = np.random.rand(layer_shapes[i], layer_shapes[i+1])
                self.net_Input[i] = np.empty((layer_shapes[i]))
                self.layer_error[i] = np.empty((layer_shapes[i]))
            self.weights[0] = np.random.rand(5, layer_shapes[0])  # loop over layer matrix s and initialise them
            for i in range(0, num_layers):
                self.weight[i+1] = np.random.rand(layer_shapes[i]+1, layer_shapes[i+1])
                self.net_Input[i] = np.empty((layer_shapes[i]))
                self.layer_error[i] = np.empty((layer_shapes[i]))
        print(self.weights)  # remove when done

    def sigmoid(self, num):
        a = 1
        return 1/(1+np.exp(-a*num))

    def hyperbolic(self, num):
        a = 1
        return (1-np.exp(-a*num))/(1+np.exp(-a*num))

    def gradient(self, num):
        return num * (1-num)

    def activation_func(self, net):
        a = 1
        if self.function_number == 0:
            return (1 - np.exp(-a * net)) / (1 + np.exp(-a * net))
        else:
            return 1 / (1 + np.exp(-a * net))

    def network_output(self, f1, f2, f3, f4):
        # Calculate the dot product between the inputs & the weights
        self.net_Input[0] = np.dot([f1, f2, f3, f4], self.weight[0])
        for i in range(1, self.num_layers+2):
            self.net_Input[i] = np.dot(self.net_Input[i-1], self.weight[i])

        self.output_y = self.activation_func(self.net_Input[self.num_layers+1])
        print("network output")

    # use it to calculate error for each node in each layer(and save them with weights)
    def calculate_error(self, label):
        #  go through the network backward
        # for the output layer calculate square error
        # sigma at output layer = (t-y)*f'(net_input for output layer)   | num_layer+1  -> output layer
        self.error[self.num_layers+1] = (label - self.output_y) * self.gradient(self.net_Input[self.num_layers+1])
        for i in range(self.num_layers, 0, -1):
            self.error[i] = np.dot(self.weight[i+1], np.transpose(self.error[i+1])) * self.gradient(self.net_Input[i])

    def update_weights(self,input):
        print("update weights")
        # loop forward and change weights using error previously calculated
        self.weight[0] = self.weight[0] + self.learning_rate * self.error[0] * np.transpose(input)
        for i in range(1, self.num_layers+2):
            self.weight[i] = self.weight[i] + self.learning_rate * self.error[i] * np.transpose(self.net_Input[i-1])

    def train(self, labels, epochs, bias, x1, x2, x3, x4, lr, function_num):
        n = len(labels)
        self.bias = bias
        self.function_number = function_num
        self.learning_rate = lr

        for i in range(0, epochs):
            for j in range(0, n):
                # phase 1 : forward phase, get Y
                self.network_output(x1[j], x2[j], x3[j], x4[j])
                # phase 2 : backwards phase,error
                self.calculate_error(labels[j])
                # phase 3 : update weights
                self.update_weights([x1[j], x2[j], x3[j], x4[j]])
            print(" ")

        # return self.weights matOfNeur=[]
        # for k in range(1,self.num_layers+1):
        #                     for s in range(0,self.layers_shape[k]):
        #                         for h in range(0,self.layers_shape[k]):
        #                             matOfNeur[k*s+s]+=self.weight[k][h]*matOfNeur[(k-1)*h+h]
        #                 matOfNeur[len(matOfNeur)]

    def test(self, labels, x1, x2, x3, x4):
        miss = 0
        conf_mat = np.zeros((3, 3))
        for i in range(0, len(labels)):
            # y = get y through phase1 function
            self.network_output(x1[i], x2[i], x3[i], x4[i])
            # calculate error for the last layer only
            # deal with labels having 3 indexes

            '''yi=0 
            if yi != labels[i]:
                miss += 1

            # handel 3x3 matrix
            if labels[i] == 1 and yi == 1:
                conf_mat[0][0] += 1
            elif labels[i] == -1 and yi == -1:
                conf_mat[1][1] += 1
            elif labels[i] == -1 and yi == 1:
                conf_mat[1][0] += 1
            else:
                # label = 1 y = -1
                conf_mat[0][1] += 1

        # conf_mat[0][2] = conf_mat[0][0]/(conf_mat[0][0]+conf_mat[0][1])
        # conf_mat[1][2] = conf_mat[1][1]/(conf_mat[1][0]+conf_mat[1][1])
        total_error = miss/len(labels)
        return conf_mat, total_error'''
model = BackPropagation(BackPropagation, num_layers, layer_shapes, bias)



