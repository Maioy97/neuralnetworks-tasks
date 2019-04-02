import tkinter as tk
import tkinter.messagebox
import matplotlib.pyplot as plt
import numpy as np
from task3_backpropagation import classification


class GUI:
    window = tk.Tk()
    #class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    #features_list = ["X1", "X2", "X3", "X4"]
    module = classification.tr()
    txtbx_numOfLayers = ''
    txtbx_numOfNeurons = ''
    txtbx_rate = ''
    txtbx_numOfEpoces = ''
    chbttn_bias = ''
    selection = ''
    selectionf = ''
    bias = tk.IntVar()
    rate = tk.StringVar()
    numOfEpoces = tk.StringVar()
    numOfLayers = ''
    numOfNeurons = ''

    def __init__(self):
        self.setup()
        self.show()

    def bttn_start_onclick(self, a):
        # check which classes and features are selected
        numOfEpoces = float(self.numOfEpoces.get())
        l_bias = int(self.bias.get())
        l_rate = float(self.rate.get())
        # call read data with said classes and features
        x1features, x2features, x3features,x4features, labels, class_names = self.read_data_()
        # organise data : divide it into train and test data
        x1features = np.array(x1features)

        tr_x1 = np.array(x1features[0:30])
        tr_x1 = np.append(tr_x1, x1features[50:80])
        tr_x1 = np.append(tr_x1, x1features[100:130])

        ts_1 = np.array(x1features[30:50].astype(float))
        ts_1 = np.append(ts_1, x1features[80:100].astype(float))
        ts_1=np.append(ts_1,x1features[130:150])

        x2features = np.array(x2features).astype(float)
        tr_x2 = np.array(x2features[0:30].astype(float))
        tr_x2 = np.append(tr_x2, x2features[50:80].astype(float))
        tr_x2 = np.append(tr_x2, x1features[100:130])

        ts_2 = np.array(x2features[30:50].astype(float))
        ts_2 = np.append(ts_2, x2features[80:100].astype(float))
        ts_2 = np.append(ts_2, x2features[130:150].astype(float))

        x3features = np.array(x3features)
        tr_x3 = np.array(x3features[0:30])
        tr_x3 = np.append(tr_x3, x3features[50:80])
        tr_x3 = np.append(tr_x3, x3features[100:130])

        ts_3 = np.array(x3features[30:50].astype(float))
        ts_3 = np.append(ts_3, x3features[80:100].astype(float))
        ts_3 = np.append(ts_3, x3features[130:150])

        x4features = np.array(x4features)
        tr_x4 = np.array(x4features[0:30])
        tr_x4 = np.append(tr_x4, x4features[50:80])
        tr_x4 = np.append(tr_x4, x4features[100:130])

        ts_4 = np.array(x4features[30:50].astype(float))
        ts_4 = np.append(ts_4, x4features[80:100].astype(float))
        ts_4 = np.append(ts_4, x4features[130:150])

        labels = np.array(labels).astype(int)
        train_labels = np.array(labels[0:30].astype(int))
        train_labels = np.append(train_labels, labels[50:80].astype(int))
        train_labels = np.append(train_labels,labels[100:130])

        test_labels = np.array(labels[30:50].astype(int))
        test_labels = np.append(test_labels, labels[80:100].astype(int))
        test_labels = np.append(test_labels,labels[130:150])
        weights = self.module.train(train_labels, numOfEpoces, l_bias, tr_x1, tr_x2,tr_x3,tr_x4, l_rate)
        # get line points
        decision_line = []
        x = max(tr_x2)
        # x2 = - (w[1] x1 + b)/w[2]
        y = -(weights[2]*x + weights[0]*l_bias) / weights[1]
        decision_line.append((x, y))

        x = min(tr_x2)
        y = -(weights[2] * x + weights[0]*l_bias) / weights[1]
        decision_line.append((x, y))
        print(decision_line)
        # output graph (decision boundary visible)
        feature1 = self.selected_features[0]
        feature2 = self.selected_features[1]
        # call test and output the percentage
        conmat, error = self.module.test(test_labels, ts_1, ts_2,ts_3,ts_4)
        # show accuracy
        accuracy = (1 - error)*100
        msg_str = 'model is ,', accuracy, ' % accurate'
        msg = tk.messagebox.showinfo("model accuracy", msg_str)
        print(conmat)
        #self.plot_class_([feature1, feature2], x1features, x2features, decision_line, class_names)





    def setup(self):

        self.window.geometry("550x300")

        lbl_numOfLayers = tk.Label(self.window, text="Enter Number Of Layers")
        lbl_numOfLayers.place(x=25, y=25)

        self.txtbx_numOfLayers = tk.Entry(self.window, textvariable=self.numOfLayers)
        self.txtbx_numOfLayers.place(x=25, y=50)

        lbl_numOfNeurons = tk.Label(self.window,text="Enter Number Of Neurons Per Layer .. comma Seprate them")
        lbl_numOfNeurons.place(x=150,y=25)


        self.txtbx_numOfNeurons = tk.Entry(self.window, textvariable=self.numOfNeurons)
        self.txtbx_numOfNeurons.place(x=150, y=50)

        lbl_rate = tk.Label(self.window, text="learning rate")
        lbl_rate.place(x=300, y=50)
        self.txtbx_rate = tk.Entry(self.window, textvariable=self.rate)
        self.txtbx_rate.place(x=375, y=50)

        lbl_nEpoces = tk.Label(self.window, text="Epochesnum")
        lbl_nEpoces.place(x=300, y=75)
        self.txtbx_numOfEpoces = tk.Entry(self.window, textvariable=self.numOfEpoces)
        self.txtbx_numOfEpoces.place(x=375, y=75)

        lbl_bias = tk.Label(self.window, text="bias")
        lbl_bias.place(x=300, y=100)

        self.chbttn_bias = tk.Checkbutton(self.window, variable=self.bias)
        self.chbttn_bias.place(x=350, y=100)

        bttn_start = tk.Button(self.window, text="start")
        bttn_start.place(x=300, y=150)
        bttn_start.bind("<Button-1>", self.bttn_start_onclick)  # <Button-1> = left click

    def show(self):
        self.window.mainloop()

    def plot_class_(self, feature_num, feature1, feature2, decision_line, class_names):
        # for the showing output
        plt.figure(class_names[0] + " vs " + class_names[1])
        plt.xlabel('X%d' % (feature_num[0] + 1))
        plt.ylabel('X%d' % (feature_num[1] + 1))
        x = np.array(feature1)
        y = np.array(feature2)
        plt.scatter(x[0:50], y[0:50])
        plt.scatter(x[50:100], y[50:100])
        print(decision_line)
        plt.plot([decision_line[0][1], decision_line[1][1]], [decision_line[0][0], decision_line[1][0]])
        # consists of two points x1 x2 y1 y2

        plt.show()

    def plot_class(self, features1, features2):
        # for the first part of the task
        # will be called 6 times
        x, y, labels, class_names = self.read_data(GUI, features1, features2)
        plt.figure(class_names[0]+" vs "+class_names[1]+" vs "+class_names[2])
        plt.xlabel('X%d' % (features1 + 1))
        plt.ylabel('X%d' % (features2 + 1))
        x = np.array(x)
        y = np.array(y)
        plt.scatter(x[0:50], y[0:50])
        plt.scatter(x[50:100], y[50:100])
        plt.scatter(x[100:150], y[100:150])

        plt.show()

    def read_data_(self, class1, class2, features1, features2):

        # reads data based on class number and feature number
        # reads selected features starting at row classnumber*50 till row class number*50+50
        Xfeatures = []
        Yfeatures = []
        labels = []

        fp = open('IrisData.txt')  # Open file on read mode
        lines = fp.read().split("\n")  # Create a list containing all lines
        fp.close()  # Close file
        class_names = ['', '']
        start = class1*50+1   # since line 1 is table labels
        end = start+50
        line = lines[start].split(',')
        class_names[0] = line[4]
        for i in range(start, end):
            line = lines[i].split(',')
            Xfeatures.append(float(line[features1]))
            Yfeatures.append(float(line[features2]))
            labels.append(1)

        start = class2 * 50+1
        end = start + 50
        line = lines[start].split(',')
        class_names[1] = line[4]
        for i in range(start, end):
            line = lines[i].split(',')
            Xfeatures.append(float(line[features1]))
            Yfeatures.append(float(line[features2]))
            labels.append(-1)

        return Xfeatures, Yfeatures, labels, class_names

    def read_data(self):

        # reads data based on feature number only (all classes)
        # reads selected features starting at row classnumber*50 till row class number*50+50
        X1features = []
        X2features = []
        X3features = []
        X4features = []
        labels = []

        fp = open('IrisData.txt')  # Open file on read mode
        lines = fp.read().split("\n")  # Create a list containing all lines
        fp.close()  # Close file
        class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
        for line in lines:

            line = line.split(',')
            if line[0] == "X1":
                continue
            X1features.append(float(line[0]))
            X2features.append(float(line[1]))
            X3features.append(float(line[2]))
            X4features.append(float(line[3]))
            if line[4] == "Iris-setosa":
                labels.append(0)
            elif line[4] == "Iris-versicolor":
                labels.append(1)
            elif line[4] == "Iris-virginica":
                labels.append(2)
        return X1features, X2features, X3features, X4features, labels, class_names


def test_training_only():
    bias = 1
    mod = classification.tr()
    x1features, x2features, labels, class_names = GUI.read_data_(GUI, 0, 1, 0, 1)
    # organise data : divide it into train and test data
    x1features = np.array(x1features).astype(float)

    tr_x1 = np.array(x1features[0:31])   # .astype(float))
    tr_x1 = np.append(tr_x1, x1features[50:81])  # .astype(float))

    ts_1 = np.array(x1features[31:50].astype(float))
    ts_1 = np.append(ts_1, x1features[81::].astype(float))

    x2features = np.array(x2features).astype(float)
    tr_x2 = np.array(x2features[0:31].astype(float))
    tr_x2 = np.append(tr_x2, x2features[50:81].astype(float))

    ts_2 = np.array(x2features[31:50].astype(float))
    ts_2 = np.append(ts_2, x2features[81::].astype(float))

    labels = np.array(labels).astype(int)
    train_labels = np.array(labels[0:31].astype(int))
    train_labels = np.append(train_labels, labels[50:81].astype(int))

    test_labels = np.array(labels[31:50].astype(int))
    test_labels = np.append(test_labels, labels[81::].astype(int))

    weights = mod.train(train_labels, 50, bias, tr_x1, tr_x2, .2)
    print("weight:", weights)
    # get line points
    decision_line = []
    x = x2features[0]
    y = (-weights[0] * x - 1) / weights[1]
    decision_line.append((x, y))
    x = x2features[50]
    y = (-weights[0] * x - bias) / weights[1]
    decision_line.append((x, y))
    # output graph (decision boundary visible)
    feature1 = 0
    feature2 = 1
    GUI.plot_class_(GUI,[feature1, feature2], x1features, x2features, decision_line, class_names)
    # call test and output the percentage
    conmat, error = mod.test(test_labels, ts_1, ts_2)
    # show accuracy
    accuracy = (1 - error) * 100
    msg_str = "model is %d % accurate", accuracy
    msg = tk.messagebox.showinfo("model accuracy", msg_str)
    print(conmat)


#  test_training_only()
window = GUI()
'''GUI.plot_class(GUI, 0, 1)  # x1 x2
GUI.plot_class(GUI, 0, 2)  # x1 x3
GUI.plot_class(GUI, 0, 3)  # x1 x4
GUI.plot_class(GUI, 1, 2)  # x2 x3
GUI.plot_class(GUI, 1, 3)  # x2 x4
GUI.plot_class(GUI, 2, 3)  # x3 x4'''
