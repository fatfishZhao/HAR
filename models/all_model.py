import tensorflow as tf
import torch.nn as nn

class model_config(object):
    def __init__(self, n_channels, learning_rate, seq_len, n_classes, lstm_size, lstm_layers, batch_size):
        self.n_channels = n_channels
        self.learning_rate = learning_rate
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.lstm_size = lstm_size
        self.lstm_layers = lstm_layers
        self.batch_size = batch_size


def CNN_inference(inputs_, keep_prob_, config):
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=config.n_channels * 2, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)

    max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

    # (batch, 64, 18) --> (batch, 32, 36)
    conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=config.n_channels * 4, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    flat = tf.reshape(max_pool_2, (-1, config.seq_len * config.n_channels))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)

    # Predictions
    logits = tf.layers.dense(flat, config.n_classes)
    return logits

###2018-11-21  tf-pytorch
class CNN_inference(nn.Module):
    def __init__(self):
        super(CNN_inference, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(inputs=40, pool_size=2, strides=2,padding=2)
                    )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(inputs=64, pool_size=4, strides=4,padding=2)
                    )
        self.dp = nn.Dropout(0.5)
        self.out = nn.Linear(2000,6)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0),-1)
        x = self.dp(x)
        output = self.out(x)
        return output


def LSTM_inference(inputs_, keep_prob_, config):
    lstm_in = tf.transpose(inputs_, [1, 0, 2])  # reshape into (seq_len, N, channels)
    lstm_in = tf.reshape(lstm_in, [-1, config.n_channels])  # Now (seq_len*N, n_channels)

    # To cells
    lstm_in = tf.layers.dense(lstm_in, config.lstm_size, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?
    # print "++++"
    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, config.seq_len, 0)
    cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(config.lstm_size), output_keep_prob=keep_prob_) for
         _ in
         range(2)], state_is_tuple=True)
    initial_state = cell.zero_state(config.batch_size, tf.float32)
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                     initial_state=initial_state)

    # We only need the last output tensor to pass into a classifier
    logits = tf.layers.dense(outputs[-1], config.n_classes, name='logits')
    return logits


def SerCNN_inference(inputs_, keep_prob_, config):
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=config.n_channels * 2, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    conv2 = tf.layers.conv1d(inputs=conv1, filters=config.n_channels * 2, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)

    max_pool_1 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')
    conv3 = tf.layers.conv1d(inputs=max_pool_1, filters=config.n_channels * 4, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    conv4 = tf.layers.conv1d(inputs=conv3, filters=config.n_channels * 4, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
    conv5 = tf.layers.conv1d(inputs=max_pool_2, filters=config.n_channels * 8, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_3 = tf.layers.max_pooling1d(inputs=conv5, pool_size=2, strides=2, padding='same')
    flat = tf.reshape(max_pool_3, (-1, config.seq_len * config.n_channels))
    flat = tf.nn.dropout(flat, keep_prob=keep_prob_)
    logits = tf.layers.dense(flat, config.n_classes)
    return logits


def CNN_LSTM_inference(inputs_, keep_prob_, config):
    conv1 = tf.layers.conv1d(inputs=inputs_, filters=config.n_channels * 2, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)

    conv2 = tf.layers.conv1d(inputs=conv1, filters=config.n_channels * 2, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_1 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

    # (batch, 64, 18) --> (batch, 32, 36)
    conv3 = tf.layers.conv1d(inputs=max_pool_1, filters=config.n_channels * 4, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    conv4 = tf.layers.conv1d(inputs=conv3, filters=config.n_channels * 4, kernel_size=3, strides=1,
                             padding='same', activation=tf.nn.relu)
    max_pool_2 = tf.layers.max_pooling1d(inputs=conv4, pool_size=2, strides=2, padding='same')
    rnn_channel = int(max_pool_2.shape[2])
    split = int(max_pool_2.shape[1])
    lstm_in = tf.transpose(max_pool_2, [1, 0, 2])  # reshape into (seq_len, batch, channels)
    lstm_in = tf.reshape(lstm_in, [-1, rnn_channel])  # Now (seq_len*N, n_channels)

    # To cells
    lstm_in = tf.layers.dense(lstm_in, config.lstm_size, activation=None)  # or tf.nn.relu, tf.nn.sigmoid, tf.nn.tanh?

    # Open up the tensor into a list of seq_len pieces
    lstm_in = tf.split(lstm_in, split, 0)

    # Add LSTM layers
    cell = tf.contrib.rnn.MultiRNNCell(
        [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(config.lstm_size), output_keep_prob=keep_prob_) for
         _ in
         range(2)], state_is_tuple=True)
    initial_state = cell.zero_state(config.batch_size, tf.float32)
    outputs, final_state = tf.contrib.rnn.static_rnn(cell, lstm_in, dtype=tf.float32,
                                                     initial_state=initial_state)

    # We only need the last output tensor to pass into a classifier
    logits = tf.layers.dense(outputs[-1], config.n_classes, name='logits')
    return logits