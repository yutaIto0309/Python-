#%%
from keras.datasets import mnist
# MNISTの読み込み
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# 行列の変換
X_train = X_train.reshape([X_train.shape[0], X_train.shape[1]*X_train.shape[2]])
X_test = X_test.reshape([X_test.shape[0], X_test.shape[1]*X_test.shape[2]])

import matplotlib.pyplot as plt 
# subplotsで描画を設定
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(10):
    img = X_train[y_train == i][0].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()
for i in range(5*5):
    img = X_train[y_train == 7][i].reshape(28,28)
    ax[i].imshow(img, cmap='Greys')
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.show()
# %%

# %%
