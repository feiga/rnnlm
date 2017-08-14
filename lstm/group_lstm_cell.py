import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple

class MyLSTMCell(RNNCell):
    '''My LSTM cell for experiments'''
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return LSTMStateTuple(self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, input, states, scope=None):
        ''' Run one step of My LSTM
        '''
        with tf.variable_scope(scope or "my_lstm_cell"):
            c, h = states

            x_size = input.get_shape().as_list()[1]


            W_h = tf.get_variable('lstm_cell_weight', [x_size + self.num_units, 4*self.num_units])
            bias_h = tf.get_variable('lstm_cell_bias', [4 * self.num_units])

            cell_input = tf.concat([input, h], axis=1)

            internal = tf.matmul(cell_input, W_h) + bias_h

            i, j, f, o = tf.split(internal, 4, axis=1)

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)


            return new_h, LSTMStateTuple(new_c, new_h)

class GroupLSTMCell(RNNCell):
    '''Group LSTM cell for experiments'''
    def __init__(self, num_units, num_groups, is_output_shuffle=False):
        self.num_units = num_units
        self.num_groups = num_groups
        self.num_units_per_group = self.num_units / self.num_groups
        self.is_output_shuffle = is_output_shuffle

    @property
    def state_size(self):
        return LSTMStateTuple(self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, input, states, scope=None):
        ''' Run one step of My LSTM
        '''
        with tf.variable_scope(scope or "my_lstm_cell"):
            c, h = states

            x_size = input.get_shape().as_list()[1]
            x_size_per_group = x_size / self.num_groups

            x_split = tf.reshape(input, [-1, self.num_groups, x_size_per_group])
            h_split = tf.reshape(h, [-1, self.num_groups, self.num_units_per_group])
            c_split = tf.reshape(c, [-1, self.num_groups, self.num_units_per_group])

            W_h = tf.get_variable('lstm_cell_weight', 
                [self.num_groups, 
                 x_size_per_group + self.num_units_per_group, 
                 4*self.num_units_per_group])
            bias_h = tf.get_variable('lstm_cell_bias', 
                [self.num_groups, 4 * self.num_units_per_group])

            cell_input = tf.concat([x_split, h_split], axis=2)
            cell_input = tf.transpose(cell_input, perm=[1,0,2])

            # Do batch matrix multiplication
            # cell_input: group_num x batch_size x input_size
            # W_h       : group_num x input_size x internal_hidden_size
            # internal  : group_num x batch_size x internal_hidden_size
            internal = tf.matmul(cell_input, W_h)

            # internal : batch_size x group_num x internal_hidden_size
            internal = tf.transpose(internal, perm=[1,0,2]) 
            internal = internal + bias_h

            # i, j, f, o: batch_size x group_num x hidden_size
            i, j, f, o = tf.split(internal, 4, axis=2)

            new_c_split = c_split * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_h_split = tf.tanh(new_c_split) * tf.sigmoid(o)

            # Restore the tensor shape by concat each group
            new_c = tf.reshape(new_c_split, [-1, self.num_units])
            new_h = tf.reshape(new_h_split, [-1, self.num_units])
            output = new_h
            if self.is_output_shuffle:
                # batch_size x hidden_size x group_size
                output_split = tf.transpose(new_h_split, perm=[0, 2, 1])
                output = tf.reshape(output_split, [-1, self.num_units])

            return output, LSTMStateTuple(new_c, new_h)
