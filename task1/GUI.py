import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
from task1 import classification


class GUI:
    window = tk.Tk()
    class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    features_list = ["X1", "X2", "X3", "X4"]
    module = classification.tr()
    lstbx_class = ''
    lstbx_feature = ''
    txtbx_rate = ''
    txtbx_epochs = ''
    chbttn_bias = ''
    selection = ''
    selectionf = ''
    bias = tk.IntVar()
    rate = tk.StringVar()
    epochs = tk.StringVar()
    selected_classes = [0, 1]
    selected_features = [0, 1]

    def __init__(self):
        self.setup()
        self.show()

    def bttn_start_onclick(self, a):

        # check which classes and features are selected
        self.selected_classes = self.lstbx_class.curselection()
        print("classes:", self.selected_classes)
        self.selected_features = self.lstbx_feature.curselection()
        print("features:", self.selected_features)
        l_epochs = int(self.txtbx_epochs.get())
        l_bias = int(self.bias.get())
        l_rate = float(self.rate.get())
        # call read data with said classes and features
        x1features, x2features, labels, class_names = self.read_data_(self.selected_classes[0],
                                                                      self.selected_classes[1],
                                                                      self.selected_features[0],
                                                                      self.selected_features[1])
        # organise data : divide it into train and test data
        x1features = np.array(x1features)
        tr_x1 = ts_1 = tr_x2 = ts_2 = train_labels = test_labels = []

        tr_x1.append(x1features[0:31])
        tr_x1.append(x1features[50:81])

        ts_1.append(x1features[31:50])
        ts_1.append(x1features[81::])

        x2features = np.array(x2features)
        tr_x2.append(x2features[0:31])
        tr_x2.append(x2features[50:81])

        ts_2.append(x2features[31:50])
        ts_2.append(x2features[81::])

        labels = np.array(labels)
        train_labels.append(labels[0:31])
        train_labels.append(x1features[50:81])

        test_labels.append(labels[31:50])
        test_labels.append(labels[81::])

        weights = self.module.train(train_labels, l_epochs, l_bias, tr_x1, tr_x2, l_rate)
        # get line points
        decision_line = []
        x = x2features[0]
        y = (-weights[0]*x - self.bias)/weights[1]
        decision_line.add((x, y))
        x = x2features[50]
        y = (-weights[0] * x - self.bias) / weights[1]
        decision_line.add((x, y))
        # output graph (decision boundary visible)
        feature1 = self.selected_features[0]
        feature2 = self.selected_features[1]
        self.plot_class([feature1, feature2], x1features, x2features, decision_line, class_names)
        # call test and output the percentage
        error = self.module.test(test_labels, ts_x1, ts_x2)
        accuracy = 1 - error
        # show accuracy

    def callback(self, a):
        if len(self.lstbx_class.curselection()) > 2:
            for i in self.lstbx_class.curselection():
                if i not in self.selection:
                    self.lstbx_class.selection_clear(i)
        self.selection = self.lstbx_class.curselection()

    def callback_features(self, a):
        if len(self.lstbx_feature.curselection()) > 2:
            for i in self.lstbx_feature.curselection():
                if i not in self.selection:
                    self.lstbx_feature.selection_clear(i)
        self.selectionf = self.lstbx_feature.curselection()

    def setup(self):

        self.window.geometry("550x300")

        lbl_class = tk.Label(self.window, text="Pick classes and features :")
        lbl_class.place(x=25, y=25)

        self.lstbx_class = tk.Listbox(self.window, selectmode="multiple", exportselection=0,
                                      listvariable=self.selected_classes)
        self.lstbx_class.place(x=25, y=50)
        self.lstbx_class.bind("<ButtonRelease-1>", self.callback)
        for item in self.class_names:
            self.lstbx_class.insert('end', item)

        self.lstbx_feature = tk.Listbox(self.window, selectmode="multiple", exportselection=0)
                                        #, listvariable=self.selected_features)
        self.lstbx_feature.place(x=150, y=50)
        self.lstbx_feature.bind("<ButtonRelease-1>", self.callback_features)
        for item in self.features_list:
            self.lstbx_feature.insert('end', item)

        lbl_rate = tk.Label(self.window, text="learning rate")
        lbl_rate.place(x=300, y=50)
        self.txtbx_rate = tk.Entry(self.window, textvariable=self.rate)
        self.txtbx_rate.place(x=375, y=50)

        lbl_epochs = tk.Label(self.window, text="epochs")
        lbl_epochs.place(x=300, y=75)
        self.txtbx_epochs = tk.Entry(self.window, textvariable=self.epochs)
        self.txtbx_epochs.place(x=375, y=75)

        lbl_bias = tk.Label(self.window, text="bias")
        lbl_bias.place(x=300, y=100)

        self.chbttn_bias = tk.Checkbutton(self.window, variable=self.bias)
        self.chbttn_bias.place(x=350, y=100)

        bttn_start = tk.Button(self.window, text="start")
        bttn_start.place(x=300, y=150)
        bttn_start.bind("<Button-1>", self.bttn_start_onclick)  # <Button-1> = left click

    def show(self):
        self.window.mainloop()

    def plot_class(self, feature_num, feature1, feature2, decision_line, class_names):
        # for the showing output
        plt.figure(class_names[0] + " vs " + class_names[1])
        plt.xlabel('X%d' % (feature_num[0] + 1))
        plt.ylabel('X%d' % (feature_num[1] + 1))
        x = np.array(feature1)
        y = np.array(feature2)
        plt.scatter(x[0:50], y[0:50])
        plt.scatter(x[50:100], y[50:100])
        plt.plot(decision_line[0], decision_line[1])  # consists of two points

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
            labels.append(class1)

        start = class2 * 50+1
        end = start + 50
        line = lines[start].split(',')
        class_names[1] = line[4]
        for i in range(start, end):
            line = lines[i].split(',')
            Xfeatures.append(float(line[features1]))
            Yfeatures.append(float(line[features2]))
            labels.append(class1)

        return Xfeatures, Yfeatures, labels, class_names

    def read_data(self, features1, features2):

        # reads data based on feature number only (all classes)
        # reads selected features starting at row classnumber*50 till row class number*50+50
        Xfeatures = []
        Yfeatures = []
        labels = []

        fp = open('IrisData.txt')  # Open file on read mode
        lines = fp.read().split("\n")  # Create a list containing all lines
        fp.close()  # Close file
        class_names = ["Iris-setosa", "Iris-versicolor","Iris-virginica"]
        for line in lines:

            line = line.split(',')
            if line[0] == "X1":
                continue
            Xfeatures.append(float(line[features1]))
            Yfeatures.append(float(line[features2]))
            if line[4] == "Iris-setosa":
                labels.append(0)
            elif line[4] == "Iris-versicolor":
                labels.append(1)
            elif line[4] == "Iris-virginica":
                labels.append(1)
        return Xfeatures, Yfeatures, labels, class_names


window = GUI()
'''GUI.plot_class(GUI, 0, 1)  # x1 x2
GUI.plot_class(GUI, 0, 2)  # x1 x3
GUI.plot_class(GUI, 0, 3)  # x1 x4
GUI.plot_class(GUI, 1, 2)  # x2 x3
GUI.plot_class(GUI, 1, 3)  # x2 x4
GUI.plot_class(GUI, 2, 3)  # x3 x4'''

