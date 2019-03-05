import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np


class GUI:
    window = tk.Tk()
    def setup(self):
        window = tk.Tk()

    def show(self):
        self.window.mainloop()

    def plot_class(self, class1, class2, features1, features2):
        # for the first part of the task
        # will be called 6 times
        x, y, labels, class_names = self.read_data(GUI, class1, class2, features1, features2)
        plt.figure(class_names[0]+" vs "+class_names[1])
        plt.xlabel('X%d' % (features1 + 1))
        plt.ylabel('X%d' % (features2 + 1))
        x = np.array(x)
        y = np.array(y)
        plt.scatter(x[0:50], y[0:50])
        plt.scatter(x[50::], y[50::])

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


GUI.plot_class(GUI, 0, 1, 0, 1)  # x1 x2
GUI.plot_class(GUI, 0, 1, 0, 2)  # x1 x3
GUI.plot_class(GUI, 0, 1, 0, 3)  # x1 x4
GUI.plot_class(GUI, 0, 1, 1, 2)  # x2 x3
GUI.plot_class(GUI, 0, 1, 1, 3)  # x2 x4
GUI.plot_class(GUI, 0, 1, 2, 3)  # x3 x4