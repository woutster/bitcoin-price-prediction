import os
from datetime import datetime

import numpy as np
import tensorflow as tf

from data.data_generator import *
from bluevelvet.utils.utils import *

################################################################################
################################################################################
################################################################################

seq_length = 150
data_dim   = 2
min_count  = 2
max_count  = 10
noise      = 0.01

learning_rate_base  = 0.01
learning_rate_step  = 5000
learning_rate_decay = 0.05
batch_size  = 256
train_steps = 1e5
lstm_units  = 512
max_norm_gradient = 2
forget_gate_bias = 1
weight_decay_coeff = 0.0005
num_classes = max_count-min_count

################################################################################
################################################################################
################################################################################

# Placeholders
pl_feat   = tf.placeholder(tf.float32, [None, seq_length, data_dim], name="features")
pl_labels = tf.placeholder(tf.int32, [None], name="labels")

# Training step
global_step = tf.get_variable("global_step", [], initializer=tf.constant_initializer(0), trainable=False)

learning_rate = tf.train.exponential_decay(
    learning_rate_base,
    global_step,
    learning_rate_step,
    learning_rate_decay,
    staircase=True
)

# Initialize LSTM
with tf.variable_scope("lstm"):
    lstm = tf.nn.rnn_cell.LSTMCell(num_units=lstm_units, forget_bias=forget_gate_bias)
    initial_state = state = lstm.zero_state(batch_size, tf.float32)

# Final softmax layer for predictions
with tf.variable_scope("softmax"):
    softmax_W = tf.get_variable("W", [lstm_units, num_classes], initializer=tf.contrib.layers.variance_scaling_initializer())
    softmax_b = tf.get_variable("b", [num_classes], initializer=tf.zeros_initializer)

# Unrolling the LSTM
for step in range(seq_length):
    with tf.variable_scope("lstm") as scope:
        if step > 0: scope.reuse_variables()
        output, state = lstm(pl_feat[:,step,:], state)

# Compute the predictions
with tf.variable_scope("softmax"):
    output = tf.reshape(tf.concat(1, output), [-1, lstm_units])
    logits = tf.matmul(output, softmax_W) + softmax_b
    proba  = tf.nn.softmax(logits, name="probabilities")
    predictions = tf.argmax(proba, dimension=1)

# Compute the loss
with tf.variable_scope("loss"):
    labels_one_hot =tf.one_hot(tf.to_int32(pl_labels), num_classes)
    classification_loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels_one_hot)
    classification_loss = tf.reduce_mean(classification_loss, name="loss")
    accuracy = tf.contrib.metrics.accuracy(predictions, tf.to_int64(pl_labels))

    # Weight decay
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('/b:' in v.name)])
    l2_loss *= weight_decay_coeff

    # Total loss
    loss = classification_loss + l2_loss
    classification_loss_frac = tf.div(classification_loss, loss)

# RMSProp optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate)

# Compute the gradients
grads_and_vars = optimizer.compute_gradients(loss)
grads, variables = zip(*grads_and_vars)

for g, v in grads_and_vars:
    if g is not None:
        tf.histogram_summary("gradients/{}/hist".format(v.name), g)
        tf.scalar_summary("gradients/{}/max".format(v.name), tf.reduce_max(g))
        tf.scalar_summary("gradients/{}/mean".format(v.name), tf.reduce_mean(g))
        tf.scalar_summary("gradients/{}/sparsity".format(v.name), tf.nn.zero_fraction(g))

# Optionally perform gradient clipping
if max_norm_gradient > 0:
    grads_clipped, _ = tf.clip_by_global_norm(grads, clip_norm=max_norm_gradient)
    grads_and_vars   = zip(grads_clipped, variables)

# Apply the gradients to adjust the shared variables.
train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

# Summaries
tf.scalar_summary("train/learning_rate", learning_rate)
tf.scalar_summary("train/classification_loss", classification_loss)
tf.scalar_summary("train/classification_loss_frac", classification_loss_frac)
tf.scalar_summary("train/weight_decay_loss", classification_loss)
tf.scalar_summary("train/loss", loss)
tf.scalar_summary("train/accuracy", accuracy)

# Retain the summaries from the final tower.
summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)

# Initialize the TensorFlow session
gpu_options = tf.GPUOptions(
    per_process_gpu_memory_fraction=0.9,
)
sess = tf.Session(config=tf.ConfigProto(
    gpu_options=gpu_options,
    log_device_placement=False,
    allow_soft_placement=False
))

# Initialize the session
init = tf.initialize_all_variables()
sess.run(init)

# Initialize summary writers
summary_path = os.path.join("./output/%s_seq_length_%i" % (datetime.now().strftime("%Y%m%d_%H%M"), seq_length))
mkdirs(summary_path)

summary_writer = tf.train.SummaryWriter(summary_path, sess.graph)
summary_op = tf.merge_summary(summaries)

for step in range(int(train_steps)):

    # Generate a new batch ==> Replace with your own code...
    X, y = generate_batch(batch_size, seq_length, data_dim, min_count, max_count, noise)

    feed_dict = {
        pl_feat: X,
        pl_labels: y
    }

    if step % 10 == 0:
        # Run with summaries
        _, np_loss, np_predictions, np_accuracy, np_class_loss_frac, np_learn_rate, summary_str = \
            sess.run([train_op, loss, predictions, accuracy, \
                      classification_loss_frac, learning_rate, summary_op], feed_dict=feed_dict)
    else:
        # Run without summaries
        _, np_loss, np_predictions, np_accuracy, np_class_loss_frac, np_learn_rate = \
            sess.run([train_op, loss, predictions, accuracy,
                      classification_loss_frac,
                      learning_rate], feed_dict=feed_dict)

    if step % 10 == 0:
        summary_writer.add_summary(summary_str, step)
        print("[%s] Step %05i/%05i, LR = %.2e, ClassLossFrac = %.2f, Accuracy = %.2f, Loss = %.3f" %
              (datetime.now().strftime("%Y-%m-%d %H:%M"), step, train_steps,
               np_learn_rate, np_class_loss_frac, np_accuracy, np_loss))

    if step % 100 == 0:
        print(y[0:20])
        print(np_predictions[0:20])