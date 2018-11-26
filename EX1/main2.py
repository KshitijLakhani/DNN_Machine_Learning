import tensorflow as tf

a =tf.constant(3.0,dtype=tf.float32)   # by default it is float32 but you can explicitly mention too
b = tf.constant(4.0)
total = a+b
print(a)
print(b)
print(total)
#All this part above does not generate your output. It only generates a computation graph and assigns names for each operation in it.To obtain the output we need to run a session 
#The output that you get when you run the above code only is that you get a list of three tf.Tensor type objects printed out along with their attribute details


#Next part is for tensorboard - to view the computation graph
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

#Next we write stuff for running the tensors / computation graph to obtsin output i.e. the session
#A session encapsulates the state of the TensorFlow runtime, and runs TensorFlow operations.
#If a tf.Graph is like a .py file, a tf.Session is like the python executable.
#The following code creates a tf.Session object and then invokes its run method to evaluate the total tensor we created above:
sess = tf.Session()
print(sess.run(total))
#print statement can also be written as 
print(sess.run({'a,b':(a,b), 'total':total}))








