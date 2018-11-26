#This is to show that everytime a session is run, a new set of random numbers is genrated
import tensorflow as tf

vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(tf.Session().run({'vec':vec}))              #When session run for the first time
print(tf.Session().run({'vec':vec}))              #When session run for the second time 
print(tf.Session().run({'out2':out2, 'out1':out1, 'vec':vec}))              #When session run for the third time 									 and other two variables printed too
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())




