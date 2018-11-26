import tensorflow as tf

alpha =0.02
x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

optimizer = tf.train.GradientDescentOptimizer(alpha)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

sess = tf.Session()
sess.run(init)
iter_num = 1000
#file = open("out1.txt","w")
for i in range(iter_num):
	_, loss_value = sess.run((train, loss))
	print('loss_value =',loss_value)
#	file.write("loss_value[{}] = {}\n".format(i,loss_value))

print('\n \n If no. of iterations = {} \n'.format(iter_num))
print(sess.run({'y_true':y_true,'\n y_pred':y_pred}))


