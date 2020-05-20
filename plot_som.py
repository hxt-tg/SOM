import numpy as np
import matplotlib.pyplot as plt

GRID_H, GRID_W = 30, 30

def plot_and_save(iter_num):
    data = np.loadtxt(f'output/weight_{iter_num:03d}.csv', delimiter=',')
    plt.figure(figsize=(20, 20), facecolor='white')
    cnt = 0
    for j in range(GRID_W):  # images mosaic
        for i in range(GRID_H):
            plt.subplot(GRID_H, GRID_W, cnt+1, frameon=False,  xticks=[],  yticks=[])
            plt.imshow(data[i*10+j].reshape(-1, 28),
                           cmap='Greys', interpolation='nearest')
            cnt = cnt + 1

    plt.tight_layout()
    plt.savefig(f'images/weight_{iter_num:03d}.png')
    plt.close()

#for i in list(range(0, 50, 1)):
for i in list(range(50, 2000, 50)):
    plot_and_save(i)
    print(f'Save: {i}')
