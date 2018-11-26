#This code can be run even without the tensorboard lines to get correct operation but will not get tensorboard visualization

import tensorflow as tf
from datetime import datetime      #for tensorboard
LOG_PATH = './tmp/main/' + datetime.now().isoformat()  #for tensorboard



#A graph can be parameterized to accept external inputs, known as placeholders. A placeholder is a promise to provide a value later, like a function argument.


a = tf.placeholder("float",name = 'a')
b = tf.placeholder("float",name = 'b')

y = tf.multiply(a,b,name = 'y')

with tf.Session() as sess:
#create log writer object - for tensorboard
	writer = tf.summary.FileWriter(LOG_PATH, graph = sess.graph)

#perform calculation
#. We can evaluate this graph with multiple inputs by using the feed_dict argument of the tf.Session.run method to feed concrete values to the placeholders:
#The only difference between placeholders and other tf.Tensors is that placeholders throw an error if no value is fed to them.
	result = sess.run(y,feed_dict={a:6,b:7})
	print(result)


