import numpy as np
import tensorflow as tf

def normal_rnn(X, num_units, if_bidrec, cell_type, if_peephole, num_classes, reuse = False):
    """
    @author Zhihan Xiong

    X : data in shape [batch, features, time, channel]
    if_bidrec: If use bidirectional RNN
    num_units: number of units in each rnn cell
    cell_type: 'LSTM' or 'GRU'
    """

    num_layers = len(num_units)
    data = tf.squeeze(X, axis = 3)
    outputs = tf.transpose(data, [2, 0, 1])

    if if_bidrec:
        for i in range(num_layers):
            with tf.variable_scope("bd_rnn_{}".format(i+1), reuse = reuse):
                if cell_type == 'LSTM':
                    rnn_fw_cell = tf.contrib.rnn.LSTMCell(num_units[i], if_peephole)
                    rnn_bw_cell = tf.contrib.rnn.LSTMCell(num_units[i], if_peephole)
                elif cell_type == 'GRU':
                    rnn_fw_cell = tf.contrib.rnn.GRUCell(num_units[i])
                    rnn_bw_cell = tf.contrib.rnn.GRUCell(num_units[i])
                else:
                    raise Exception("Unvalid cell type.")
                raw_out, states = tf.nn.bidirectional_dynamic_rnn(rnn_fw_cell, rnn_bw_cell, outputs, dtype = tf.float32, time_major = True)
                outputs = tf.concat(raw_out, 2)
    else:
        for i in range(num_layers):
            with tf.variable_scope("rnn_{}".format(i+1), reuse = reuse):
                if cell_type == 'LSTM':
                    rnn_cell = tf.contrib.rnn.LSTMCell(num_units[i], if_peephole)
                elif cell_type == 'GRU':
                    rnn_cell = tf.contrib.rnn.GRUCell(num_units[i])
                else:
                    raise Exception("Unvalid cell type.")
                outputs, states = tf.nn.dynamic_rnn(rnn_cell, outputs, dtype = tf.float32, time_major = True)

    logits = tf.layers.dense(outputs[-1], num_classes, tf.nn.relu)
    return logits

def test_rnn():
    trial_input = tf.placeholder(tf.float32, [None, 30, 1000, 1])
    fake_X = np.zeros([20, 30, 1000, 1])
    output = normal_rnn(trial_input, [10, 20], False, 'GRU', True, 7)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    result = sess.run(output, feed_dict = {trial_input: fake_X})
    print(result.shape)

if __name__ == '__main__':
    test_rnn()
