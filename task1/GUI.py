import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np


class GUI:
    window = tk.Tk()
    class_names = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    def setup(self):
        window = tk.Tk()
        window.geometry("500x500")
        lbl_classes = tk.Label(window, text="classes")
        lbl_classes.place(x=25, y=25)
        lstbx_class1 = tk.Listbox(window, selectmode="single")
        lstbx_class1.place(x=25, y=50)
        lstbx_class2 = tk.Listbox(window, selectmode="single")
        lstbx_class2.place(x=150, y=50)
        for item in self.class_names:
            lstbx_class1.insert('end', item)
        lbl_features = tk.Label(window, text="features")
        lbl_features.place(x=25, y=225)
        lstbx_feature1 = tk.Listbox(window, selectmode="single")
        lstbx_feature1.place(x=25, y=250)
        lstbx_feature2 = tk.Listbox(window, selectmode="single")
        lstbx_feature2.place(x=150, y=250)
        lbl_rate = tk.Label(window, text="learning rate")
        lbl_rate.place(x=300, y=25)
        lbl_epochs = tk.Label(window, text="epochs")
        lbl_epochs.place(x=300, y=50)
        lbl_bias = tk.Label(window, text="bias")
        lbl_bias.place(x=300, y=75)
        chbttn_bias= tk.Checkbutton(window)
        chbttn_bias.place(x=350,y=75)
        bttn_start = tk.Button(window, text="start")
        bttn_start.place(x=300, y=100)
        for item in self.class_names:
            lstbx_feature2.insert('end', item)


    def show(self):
        self.window.mainloop()

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

    def read_data(self, class1, class2, features1, features2):

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

        # reads data based on class number and feature number
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


GUI.setup(GUI)
GUI.show(GUI)
'''GUI.plot_class(GUI, 0, 1)  # x1 x2
GUI.plot_class(GUI, 0, 2)  # x1 x3
GUI.plot_class(GUI, 0, 3)  # x1 x4
GUI.plot_class(GUI, 1, 2)  # x2 x3
GUI.plot_class(GUI, 1, 3)  # x2 x4
GUI.plot_class(GUI, 2, 3)  # x3 x4'''