import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('tkagg')


def main():
    path = 'trained_models/trained_losses/'


    total_loss = np.loadtxt(open(path+"l_total_history.csv", "rb"), delimiter=",")
    feat_loss = np.loadtxt(open(path+"l_feat_history.csv", "rb"), delimiter=",")
    style_loss = np.loadtxt(open(path+"l_style_history.csv", "rb"), delimiter=",")

    total_total_loss = np.loadtxt(open(path+"l_total_total_history.csv", "rb"), delimiter=",")
    total_feat_loss = np.loadtxt(open(path+"l_feat_total_history.csv", "rb"), delimiter=",")
    total_style_loss = np.loadtxt(open(path+"l_style_total_history.csv", "rb"), delimiter=",")

    k = 10

    x = np.arange(len(total_loss[k:]))


    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(4)

    plt.subplot(1, 3, 1, )
    plt.plot(x, total_loss[k:])
    plt.xlabel('Batches')
    plt.ylabel('Total Loss')

    plt.subplot(1, 3, 2)
    plt.plot(x, feat_loss[k:])
    plt.xlabel('Batches')
    plt.ylabel('Total feature Loss')

    plt.subplot(1, 3, 3)
    plt.plot(x, style_loss[k:])
    plt.xlabel('Batches')
    plt.ylabel('Total style Loss')

    plt.tight_layout()
    plt.show()


    f = plt.figure()
    f.set_figwidth(15)
    f.set_figheight(4)

    plt.subplot(1, 3, 1, )
    plt.plot(x, total_total_loss[k:])
    plt.xlabel('Batches')
    plt.ylabel('Aggregated Loss')

    plt.subplot(1, 3, 2)
    plt.plot(x, total_feat_loss[k:])
    plt.xlabel('Batches')
    plt.ylabel('Aggregated feature Loss')

    plt.subplot(1, 3, 3)
    plt.plot(x, total_style_loss[k:])
    plt.xlabel('Batches')
    plt.ylabel('Aggregated style Loss')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()