import tensorflow as tf


INTERACT_FUNC_SET = {'sum', 'sub', 'mean', 'concat'}


def interact_encoder(user_vec, item_vec, hidden1_dim, hidden2_dim,
                     interact_type='sum', activation=tf.nn.relu):
    '''
    Calculate the interaction between the user and the item.
    '''

    interact_type = interact_type.lower()
    assert interact_type in INTERACT_FUNC_SET

    # interaction
    _user = tf.tile(user_vec, [1, tf.shape(item_vec)[1], 1])
    if interact_type == 'sum':
        merge_vec = _user + item_vec
    elif interact_type == 'sub':
        merge_vec = _user - item_vec
    elif interact_type == 'mean':
        merge_vec = (_user + item_vec) / 2
    elif interact_type == 'concat':
        merge_vec = tf.concat([_user, item_vec], axis=-1)

    encoder = tf.layers.dense(merge_vec, hidden1_dim, activation=activation, name='encoder_hidden1',
                              reuse=tf.AUTO_REUSE)
    encoder = tf.layers.batch_normalization(encoder, name='encoder_bn1', reuse=tf.AUTO_REUSE)
    encoder = tf.layers.dense(encoder, hidden2_dim, name='encoder_hidden2', reuse=tf.AUTO_REUSE)
    encoder = tf.layers.batch_normalization(encoder, name='encoder_bn2', reuse=tf.AUTO_REUSE)
    return encoder


def not_modules(input, hidden1_dim, hidden2_dim, activation=tf.nn.relu):
    '''
    An module to calculate the logical operation NOT(*).
    '''
    not_encoder = tf.layers.dense(input, hidden1_dim, activation=activation, name='not_hidden1',
                                  reuse=tf.AUTO_REUSE)
    not_encoder = tf.layers.batch_normalization(not_encoder, name='not_bn1', reuse=tf.AUTO_REUSE)
    not_encoder = tf.layers.dense(not_encoder, hidden2_dim, name='not_hidden2', reuse=tf.AUTO_REUSE)
    not_encoder = tf.layers.batch_normalization(not_encoder, name='not_bn2', reuse=tf.AUTO_REUSE)
    return not_encoder


def cosine_probability(vec_a, vec_b):
    '''
    Calculate the cosine similarity between {vec_a} and {vec_b}.
    '''
    a_norm = tf.sqrt(tf.reduce_sum(tf.square(vec_a), axis=-1))
    b_norm = tf.sqrt(tf.reduce_sum(tf.square(vec_b), axis=-1))
    _prod = tf.multiply(vec_a, vec_b)
    inner_prod = tf.reduce_sum(_prod, axis=-1)
    prob = inner_prod / (a_norm * b_norm)
    return prob


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''
    Noam scheme learning rate decay.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)


class OrMoudleCell(tf.nn.rnn_cell.RNNCell):
    '''
    An module to calculate the logical operation OR(*).
    This is a rnn cell, each time can just operate between 2 vector.

    `input` is a matrix without step 0,
    and `state` is initialized to the vector at step 0.
    '''

    def __init__(self, num_units_1, num_units_2, activation=None, reuse=tf.AUTO_REUSE, name=None):
        super(OrMoudleCell, self).__init__(_reuse=reuse, name=name)
        self._num_units_1 = num_units_1
        self._num_units_2 = num_units_2
        self._activation = activation or tf.nn.tanh

    @property
    def state_size(self):
        return self._num_units_2

    @property
    def output_size(self):
        return self._num_units_2

    def build(self, inputs_shape):
        self.layer_1 = tf.layers.Dense(self._num_units_1, activation=self._activation,
                                       name="or_hidden1")
        self.layer_2 = tf.layers.Dense(self._num_units_2, name="or_hidden2")
        self.built = True

    def call(self, inputs, state):
        hidden = inputs + state
        hidden = self.layer_1(hidden)
        output = self.layer_2(hidden)
        return output, output
