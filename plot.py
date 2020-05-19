import numpy as np
import matplotlib.pyplot as plt

with open('mnist/mnist_np.npy', 'rb') as f:
    data = np.load(f)

data = data[:100, :]
img = data[0].reshape(-1, 28)

##plt.imshow(img, cmap='gray')
##plt.show()

with open('output/activate_result.csv', 'r') as f:
    f.readline()
    win_coor = dict()
    for i, line in enumerate(f.readlines()):
        _, x, y = line.strip().split(',')
        win_coor[(int(x), int(y))] = i
print(win_coor)


plt.figure(figsize=(5, 5), facecolor='white')
cnt = 0
for j in range(10):  # images mosaic
    for i in range(10):
        plt.subplot(10, 10, cnt+1, frameon=False,  xticks=[],  yticks=[])
        if (i, j) in win_coor:
            plt.imshow(data[win_coor[(i, j)]].reshape(-1, 28).T,
                       cmap='Greys', interpolation='nearest')
        else:
            plt.imshow(np.zeros((28, 28)),  cmap='Greys')
        cnt = cnt + 1

plt.tight_layout()
# plt.savefig('resulting_images/som_digts_imgs.png')
plt.show()

