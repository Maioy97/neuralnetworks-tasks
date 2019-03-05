import tkinter as tk
import matplotlib as plt
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
        x, y, lables, class_names = self.read_data(class1, class2, features1, features2)
        x = np.array(x)
        y = np.array(y)
        plt.scatter(x[0:50], y[0:50])
        plt.scatter(x[50::], y[50::])
        plt.figure(class_names[0]+" vs "+class_names[1])
        plt.xlable('feature 1')
        plt.ylable('feature 2')

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
            Xfeatures.append(line[features1])
            Yfeatures.append(line[features2])
            labels.append(class1)

        start = class2 * 50+1
        end = start + 50
        line = lines[start].split(',')
        class_names[1] = line[4]
        for i in range(start, end):
            line = lines[i].split(',')
            Xfeatures.append(line[features1])
            Yfeatures.append(line[features2])
            labels.append(class1)

        return Xfeatures, Yfeatures, labels, class_names


GUI.read_data(GUI, 0, 1, 0, 1)
