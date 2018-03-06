'''

This trains a very simple SNN on mnist data using distributed TF APIs.
Refer to slide# 17 in https://www.slideshare.net/ManiGoswami/into-to-tensorflow-architecture-v2


To run use below commands (I am simulating on my local laptop on independent ports):
If you have three independent hosts, replace the 127.0.0.1 with respective hostnames/IP
 
  """
python distributed.py  --ps_hosts=127.0.0.1:2222  --worker_hosts=127.0.0.1:2223,127.0.0.1:2224  --job_name=ps --task_index=0
python distributed.py  --ps_hosts=127.0.0.1:2222  --worker_hosts=127.0.0.1:2223,127.0.0.1:2224  --job_name=worker --task_index=0
python distributed.py  --ps_hosts=127.0.0.1:2222  --worker_hosts=127.0.0.1:2223,127.0.0.1:2224  --job_name=worker --task_index=1
  """

__author__ = 'manishankargoswami'

'''

import tensorflow as tf
import time

# flags for input defined here
flags = tf.app.flags
tf.app.flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_string("worker_hosts", None,
                    "The worker url list, separated by comma (e.g. tf-worker1:2222,1.2.3.4:2222)")

flags.DEFINE_string("ps_hosts", None,
                    "The ps url list, separated by comma (e.g. tf-ps2:2222,1.2.3.5:2222)")
FLAGS = tf.app.flags.FLAGS


# creating a cluster using the flags defined above
# (slide# 16 in https://www.slideshare.net/ManiGoswami/into-to-tensorflow-architecture-v2).
cluster = tf.train.ClusterSpec({"ps": FLAGS.ps_hosts.split(","), "worker": FLAGS.worker_hosts.split(",")})


# start a server for a specific task
server = tf.train.Server(cluster,
                         job_name = FLAGS.job_name,
                         task_index = FLAGS.task_index)

# training configurations
batch_size = 100
learning_rate = 0.001
training_epochs = 50
logs_path = "/tmp/mnist/"

# load training examples
from tensorflow.examples.tutorials.mnist import input_data

# read with one_hot set to true
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

if FLAGS.job_name == "ps":
    server.join()
elif FLAGS.job_name == "worker":

    # <--- Between-graph replication --->
    # Refer slide# 24 in https://www.slideshare.net/ManiGoswami/into-to-tensorflow-architecture-v2
    with tf.device(tf.train.replica_device_setter(
            worker_device = "/job:worker/task:%d" % FLAGS.task_index,
            cluster = cluster)):

        # counting the number of updates so far
        global_step = tf.get_variable('global_step', [],
                                      initializer = tf.constant_initializer(0),
                                      trainable = False)

        # input images
        with tf.name_scope('input'):

            # None = tuple size can be any thing,
            # 784 = flattened mnist image
            x = tf.placeholder(tf.float32, shape = [None, 784], name = "x-input")

            # we place target of 10 here
            y_ = tf.placeholder(tf.float32, shape = [None, 10], name = "y-input")


        tf.set_random_seed(2)
        with tf.name_scope("weights"):
            W1 = tf.Variable(tf.random_normal([784, 100]))
            W2 = tf.Variable(tf.random_normal([100, 10]))

        # biases here
        with tf.name_scope("biases"):
            b1 = tf.Variable(tf.zeros([100]))
            b2 = tf.Variable(tf.zeros([10]))

        # finally the model here...
        with tf.name_scope("softmax"):

            z2 = tf.add(tf.matmul(x, W1), b1)
            a2 = tf.nn.sigmoid(z2)
            z3 = tf.add(tf.matmul(a2, W2), b2)
            y = tf.nn.softmax(z3)

        with tf.name_scope('cross_entropy'):
            cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis = [1]))

        # we will use GradientDescentOptimizer
        with tf.name_scope('train'):

            grad_op = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = grad_op.minimize(cross_entropy, global_step = global_step)

        '''
        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()
        '''

        with tf.name_scope('Accuracy'):

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # creating a summary
        tf.summary.scalar("cost", cross_entropy)
        tf.summary.scalar("accuracy", accuracy)

        # here we merge all summaries into a single one to be executed in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    sv = tf.train.Supervisor(is_chief = (FLAGS.task_index == 0),
                             global_step = global_step,
                             init_op = init_op)

    begin_time = time.time()
    frequency = 100
    with sv.prepare_or_wait_for_session(server.target) as sess:

        # this will log on every node of our cluster
        writer = tf.summary.FileWriter(logs_path, graph = tf.get_default_graph())

        start_time = time.time()
        for epoch in range(training_epochs):
            batch_count = int(mnist.train.num_examples / batch_size)

            count = 0
            for i in range(batch_count):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                _, cost, summary, step = sess.run(
                    [train_op, cross_entropy, summary_op, global_step],
                    feed_dict = {x: batch_x, y_: batch_y})
                writer.add_summary(summary, step)

                count += 1
                if count % frequency == 0 or i + 1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step so far: %d," % (step + 1),
                          " Epoch so far: %2d," % (epoch + 1),
                          " Batch used: %3d of %3d," % (i + 1, batch_count),
                          " Cost now: %.4f," % cost,
                          " Time spent (delta): %3.1fms" % float(elapsed_time * 1000 / frequency))
                    count = 0

        print("Acc: %2.2f" % sess.run(accuracy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
        print("Time Taken: 2fs" % float(time.time() - begin_time))
        print("Final Computed Cost: %.2f" % cost)

    sv.stop()
    print("done with training")
