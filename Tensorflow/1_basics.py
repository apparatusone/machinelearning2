import tensorflow as tf

print(tf.version)

# create a tensor filled with zeros
t = tf.zeros([3,3])

print(t)

# tf.Tensor(
# [[0. 0. 0.]
#  [0. 0. 0.]
#  [0. 0. 0.]], shape=(3, 3), dtype=float32)

# reshapes the tensor 
# -1  tells TensorFlow to automatically infer the size of the 
# corresponding dimension based on the other dimensions and 
# the total number of elements in the tensor.

t = tf.reshape(t, [1, -1])
print(t)

# tf.Tensor(
# [[0. 0. 0. 0. 0. 0. 0. 0. 0.]], shape=(1, 9), dtype=float32)