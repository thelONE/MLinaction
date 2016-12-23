import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns

### Preparation of data

num_points = 2000
vectors_set = []
for i in xrange(num_points) :
    if np.random.random() > 0.5 :
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else :
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

### VISUALIZATION of the original data

# df = pd.DataFrame({"x" : [v[0] for v in vectors_set], "y" : [v[1] for v in vectors_set]})
# sns.lmplot("x", "y", data = df, fit_reg = False, size = 6)
# plt.show()

### Prepare variables

vectors = tf.constant(vectors_set)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

print "shape of vectors = ", vectors.get_shape()
print "shape of centroides = ", centroides.get_shape()
print "shape of expanded_vectors = ", expanded_vectors.get_shape()
print "shape of expanded_centroides = ", expanded_centroides.get_shape()

# diff = tf.sub(expanded_vectors, expanded_centroides)
# print "shape of sub(expanded_vectors, expanded_centroides) = ", diff.get_shape()
# shape = (4, 2000, 2)

# (centroides - vector)^2
assignments = tf.argmin( tf.reduce_sum( tf.square( tf.sub(expanded_vectors, expanded_centroides) ), 2), 0 )

# equal = tf.equal(assignments, 0)
# print "shape of equal(assignments, 0) = ", equal.get_shape()
# where = tf.where(equal)
# print "shape of where(equal) = ", where.get_shape()

means = tf.concat(0, 
    [ tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c) ), [1, -1])), reduction_indices=[1]) 
        for c in xrange(k) ])

update_centroides = tf.assign(centroides, means)

init_op = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_op)

steps = 100
for step in xrange(steps) :
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])

#### ViSUALIZATION of result

data = {"x" : [], "y" : [], "cluster" : []}

for i in xrange(len(assignment_values)) :
    data["x"].append(vectors_set[i][0])
    data["y"].append(vectors_set[i][1])
    data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()


