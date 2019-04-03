import numpy as np


class BackPropagation:
    weights = {}  # []
    bias = None
    layer_error = {}
    net_Input = {}
    num_layers = None
    learning_rate = None
    function_number = None
    layers_shape = []
    output_y = []

    def __init__(self, num_layers, layer_shapes, bias):
        layer_shapes = np.append(layer_shapes, 3)
        self.layers_shapes = layer_shapes
        self.bias = bias
        self.num_layers = num_layers
        # should be changed to receive the desired network shape and initialise the weight values in it
        if self.bias == 0:
            self.weights[0] = np.random.rand(4, layer_shapes[0])  # loop over layer matrix s and initialise them
            for i in range(0, num_layers):
                self.weights[i+1] = np.random.rand(layer_shapes[i], layer_shapes[i+1])
                self.net_Input[i] = np.empty((layer_shapes[i], 1))
                self.layer_error[i] = np.empty((layer_shapes[i], 1))
        else:
            self.weights[0] = np.random.rand(5, layer_shapes[0])
            for i in range(0, num_layers):
                self.weights[i+1] = np.random.rand(layer_shapes[i]+1, layer_shapes[i+1])
                self.net_Input[i] = np.empty((layer_shapes[i], 1))
                self.layer_error[i] = np.empty((layer_shapes[i], 1))

        self.net_Input[num_layers] = np.empty(3)  # net input for the final layer
        print(self.net_Input)

    @staticmethod
    def sigmoid(num):
        a = 1
        return 1/(1+np.exp(-a*num))

    @staticmethod
    def hyperbolic(num):
        a = 1
        return (1-np.exp(-a*num))/(1+np.exp(-a*num))

    @staticmethod
    def gradient(num):
        return num * (1-num)

    def activation_func(self, net):
        a = 1
        if self.function_number == 0:
            return (1 - np.exp(-a * net)) / (1 + np.exp(-a * net))
        else:
            return 1 / (1 + np.exp(-a * net))

    def network_output(self, current_input):
        # Calculate the dot product between the inputs & the weights
        self.net_Input[0] = np.dot(current_input, self.weights[0])
        for i in range(1, self.num_layers+1):
            if self.bias != 0:
                self.net_Input[i-1] = np.append(self.bias, self.net_Input[i-1])
            current_net_input = self.net_Input[i-1]
            self.net_Input[i] = np.dot(current_net_input, self.weights[i])
            # print(i)
        self.output_y = self.activation_func(self.net_Input[self.num_layers])
        print("network output calculated")

    # use it to calculate error for each node in each layer(and save them with weights)
    def calculate_error(self, label):
        # go through the network backward
        # for the output layer calculate square error
        # sigma at output layer = (t-y)*f'(net_input for output layer)   | num_layer -> output layer
        self.layer_error[self.num_layers] = (label - self.output_y) * self.gradient(self.net_Input[self.num_layers])
        for i in range(self.num_layers-1, 0, -1):
            print(i)
            self.layer_error[i] = np.dot(self.weights[i+1], self.layer_error[i+1])
            self.layer_error[i] = self.layer_error[i] * np.transpose(self.gradient(self.net_Input[i]))

    def update_weights(self, current_input):
        print("updating weights")
        # loop forward and change weights using error previously calculated
        current_input = np.transpose(current_input)
        self.weights[0] = self.weights[0] + self.learning_rate * np.dot(current_input, np.transpose(self.layer_error[0]))
        for i in range(1, self.num_layers+1):
            self.weights[i] = self.weights[i] + self.learning_rate * self.layer_error[i] * np.transpose(self.net_Input[i-1])
        print("updating weights")

    def train(self, labels, epochs, x1, x2, x3, x4, lr, function_num):
        n = len(labels)
        self.function_number = function_num
        self.learning_rate = lr

        for i in range(0, epochs):
            for j in range(0, n):
                if self.bias == 0:
                    current_input = [[x1[j], x2[j], x3[j], x4[j]]]
                else:
                    current_input = [[self.bias, x1[j], x2[j], x3[j], x4[j]]]
                # phase 1 : forward phase, get Y
                self.network_output(current_input)
                # phase 2 : backwards phase, calculate error
                self.calculate_error(labels[j])
                # phase 3 : update weights
                self.update_weights(current_input)
            print(" ")

    @staticmethod
    def max_index(vector, length):
        max_index = 0
        print(vector)
        vector.flatten()
        print(vector)
        for i in range(0, length):
            if vector[i] > vector[max_index]:
                print (vector[i])
                max_index = i
        return max_index

    def test(self, labels, x1, x2, x3, x4):
        # calculate error for the last layer only
        # deal with labels having 3 indexes
        miss = 0
        conf_mat = np.zeros((3, 3))
        for i in range(0, len(labels)):
            # y = get y through phase1 function
            if self.bias == 0:
                current_input = [[x1[i], x2[i], x3[i], x4[i]]]
            else:
                current_input = [[self.bias, x1[i], x2[i], x3[i], x4[i]]]
            self.network_output(current_input)
            y = self.max_index(self.output_y, 3)
            t = self.max_index(labels[i], 3)
            if y == t:
                conf_mat[y][y] += 1
            else:
                conf_mat[y][t] += 1
                miss += 1

        total_error = miss/len(labels)
        return conf_mat, total_error


def test_model():
    num_layers = 1
    layer_shapes = [2]
    bias = 1
    labels = [[1, 0, 0],
              [1, 0, 0],
              [1, 0, 0],
              [1, 0, 0],
              [1, 0, 0],
              [1, 0, 0]]
    x1 = [5.1, 4.9, 4.7, 4.6, 5.0, 5.4]
    x2 = [3.5, 3.0, 3.2, 3.1, 3.6, 3.9]
    x3 = [1.4, 1.4, 1.3, 1.5, 1.4, 1.7]
    x4 = [0.2, 0.2, 0.2, 0.2, 0.2, 0.4]
    lr = .01
    epochs = 10
    model = BackPropagation(num_layers, layer_shapes, bias)
    model.train(labels, epochs, bias, x1, x2, x3, x4, lr, 0)
    conf, error = model.test(labels, x1, x2, x3, x4)
    print("confusion matrix\n ", conf)
    print("\nerror = ", error)


# test_model()

