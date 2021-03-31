#%%
import tensorflow as tf 
import numpy as np 
np.set_printoptions(precision=3)
a = np.array([1, 2, 3], dtype=np.int32)
b = [4,5,6]
# テンソル
t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)
print(t_a)
print(t_b)
t_ones = tf.ones((2,3))
print(t_ones.shape)
print(t_ones.numpy())
# キャスト
t_a_new = tf.cast(t_a, tf.int64)
print(t_a_new.dtype)
# 転置
t = tf.random.uniform(shape=(3,5))
t_tf = tf.transpose(t)
print(t_tf.shape)
# 形状変換
t = tf.zeros((30,))
t_reshape = tf.reshape(t, shape=(5, 6))
print(t_reshape.shape)
# 不要な次元の削除
t = tf.zeros((1,2,1,4,1))
t_sgz = tf.squeeze(t, axis=(2,4))
print(t_sgz.shape)
# %%
tf.random.set_seed(1)
t1 = tf.random.uniform(shape=(5,2), minval=-1.0, maxval=1.0)
t2 = tf.random.normal(shape=(5,2), mean=0.0,stddev=1.0)
print(t1)
print(t2)
# 要素ごとの積
t3 = tf.multiply(t1, t2).numpy()
print(t3)
# 各列の平均
t4 = tf.math.reduce_mean(t1, axis=0)
print(t4)
# 行列同士の積
t5 = tf.linalg.matmul(t1, t2, transpose_b=True)
print(t5.numpy())
t6 = tf.linalg.matmul(t1, t2, transpose_a=True)
print(t6.numpy())
# %%
