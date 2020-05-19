import numpy as np
import matplotlib.pyplot as plt

def plot_and_save(iter_num):
    data = np.loadtxt(f'output/weight_{iter_num:03d}.csv', delimiter=',')
    plt.figure(figsize=(10, 10), facecolor='white')
    cnt = 0
    for j in range(10):  # images mosaic
        for i in range(10):
            plt.subplot(10, 10, cnt+1, frameon=False,  xticks=[],  yticks=[])
            plt.imshow(data[i*10+j].reshape(-1, 28),
                           cmap='Greys', interpolation='nearest')
            cnt = cnt + 1

    plt.tight_layout()
    plt.savefig(f'images/weight_{iter_num:03d}.png')
    plt.close()

for i in list(range(0, 50, 1)):
# for i in list(range(50, 2000, 50)):
    plot_and_save(i)
    print(f'Save: {i}')
