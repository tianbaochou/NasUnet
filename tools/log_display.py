import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':
    csvfile = open('log.csv', 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))

    plt.plot(x, y)

    plt.xlabel('epoch')
    plt.ylabel('mIoU')
    plt.legend()
    plt.show()