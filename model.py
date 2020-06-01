import tensorflow as tf
from modules import interact_encoder, not_modules, cosine_probability, noam_scheme, OrMoudleCell


class NLR_model:

    def __init__(self, user_embedding_dim=256, item_embedding_dim=256,
                 hidden1_dim=512, hidden2_dim=256, num_users=100,
                 num_items=1000, learning_rate=1e-3, warmup_steps=4000.,
                 l2_weight=1e-4, logical_weight=0.1, interact_type='sum'):
        # hyper param
        self.user_embedding_dim = user_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.hidden1_dim = hidden1_dim
        self.hidden2_dim = hidden2_dim
        self.num_users = num_users
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.l2_weight = l2_weight
        self.logical_weight = logical_weight
        self.warmup_steps = warmup_steps
        self.interact_type = interact_type
        self.activation = tf.nn.relu

        # input
        with tf.name_scope('input_tensor'):
            self.input_user = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input_user')
            self.input_items = tf.placeholder(dtype=tf.int32, shape=[None, None],
                                              name='input_items')
            self.input_feedback_score = tf.placeholder(dtype=tf.float32, shape=[None, None],
                                                       name='input_feedback_score')
            self.input_target = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='input_target')
            self.input_negative_sample = tf.placeholder(dtype=tf.int32, shape=[None, 1],
                                                        name='input_negative_sample')

        # model structure
        self.build()

    def build(self):
        # truth vector
        self.T = tf.get_variable('truth_vector', shape=[1, 1, self.item_embedding_dim],
                                 dtype=tf.float32, trainable=False)
        # embedding matrix
        self.user_embedding_layer = tf.get_variable(name='user_embedding_layer',
                                                    shape=[self.num_users, self.user_embedding_dim],
                                                    dtype=tf.float32)
        self.item_embedding_layer = tf.get_variable(name='item_embedding_layer',
                                                    shape=[self.num_items, self.item_embedding_dim],
                                                    dtype=tf.float32)
        # embedding
        self.user_emb_vec = tf.nn.embedding_lookup(self.user_embedding_layer, self.input_user)
        self.item_emb_vec = tf.nn.embedding_lookup(self.item_embedding_layer, self.input_items)
        self.target_emb_vec = tf.nn.embedding_lookup(self.item_embedding_layer,
                                                     self.input_target)
        self.negative_sample_emb_vec = tf.nn.embedding_lookup(self.item_embedding_layer,
                                                              self.input_negative_sample)

        # interaction
        self.encoder = interact_encoder(self.user_emb_vec, self.item_emb_vec, self.hidden1_dim,
                                        self.hidden2_dim, activation=self.activation,
                                        interact_type=self.interact_type)
        self.encoder_pos = interact_encoder(self.user_emb_vec, self.target_emb_vec,
                                            self.hidden1_dim, self.hidden2_dim,
                                            activation=self.activation,
                                            interact_type=self.interact_type)
        self.encoder_neg = interact_encoder(self.user_emb_vec, self.negative_sample_emb_vec,
                                            self.hidden1_dim, self.hidden2_dim,
                                            activation=self.activation,
                                            interact_type=self.interact_type)

        # NOT(*) operation
        feedback_to_oper = self.input_feedback_score[:, :, tf.newaxis] * tf.ones_like(self.encoder)
        applicable = tf.equal(feedback_to_oper, 1)
        encoder_to_oper = tf.where(applicable, self.encoder, tf.zeros_like(self.encoder))
        not_encoder = not_modules(encoder_to_oper, self.hidden1_dim, self.hidden2_dim,
                                  activation=self.activation)
        self.not_encoder = tf.where(applicable, not_encoder, self.encoder)

        # OR(*) operation
        self.or_cell = OrMoudleCell(self.hidden1_dim, self.hidden2_dim)
        self.or_encoder, _ = tf.nn.dynamic_rnn(self.or_cell, self.not_encoder[:, 1:, :],
                                               initial_state=self.not_encoder[:, 0, :],
                                               dtype=tf.float32)
        self.or_encoder_last = self.or_encoder[:, -1, :]

        self.or_encoder_pos, _ = tf.nn.dynamic_rnn(self.or_cell, self.encoder_pos,
                                                   initial_state=self.or_encoder_last,
                                                   dtype=tf.float32)
        self.or_encoder_neg, _ = tf.nn.dynamic_rnn(self.or_cell, self.encoder_neg,
                                                   initial_state=self.or_encoder_last,
                                                   dtype=tf.float32)

        # cosine similarity
        self.probability_pos = cosine_probability(self.or_encoder_pos, self.T)
        self.probability_neg = cosine_probability(self.or_encoder_neg, self.T)

        # pair-wise loss
        self.traget_loss = -tf.reduce_sum(
            tf.log_sigmoid(self.probability_pos - self.probability_neg))

        # L2 loss
        trainable_variables = tf.trainable_variables()
        self.l2_loss = tf.reduce_sum([tf.nn.l2_loss(var) for var in trainable_variables])

        # model loss
        self.lnn_loss = self.traget_loss + self.l2_weight * self.l2_loss

        # logical regularizer loss
        event_space_vectors = [self.encoder, self.encoder_pos, self.encoder_neg,
                               self.not_encoder,
                               self.or_encoder, self.or_encoder_pos, self.or_encoder_neg]
        event_space_vectors = tf.concat(event_space_vectors, axis=1)
        self.logical_loss = self.logical_regularizer(event_space_vectors)

        # sum
        self.loss = self.lnn_loss + self.logical_weight * self.logical_loss

        # Adam
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.learning_rate, global_step, self.warmup_steps)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=lr)
        # train
        self.train_op = self.optimizer.minimize(self.loss, global_step=global_step)

        # tensorboard scalar
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('traget_loss', self.traget_loss)
        tf.summary.scalar('l2_loss', self.l2_loss)
        tf.summary.scalar('logical_loss', self.logical_loss)
        tf.summary.scalar('lr', lr)
        tf.summary.scalar('global_step', global_step)

        self.summaries = tf.summary.merge_all()

    def logical_regularizer(self, event_space):
        '''
        Build logical regularizers.
        The regularizers make NOT([T])=[F], and so on.
        '''
        F_vec = not_modules(self.T, self.hidden1_dim, self.hidden2_dim, activation=self.activation)
        not_event = not_modules(event_space, self.hidden1_dim, self.hidden2_dim,
                                activation=self.activation)
        double_not_event = not_modules(not_event, self.hidden1_dim, self.hidden2_dim,
                                       activation=self.activation)
        reg_1 = tf.reduce_mean(1 + cosine_probability(not_event, event_space))
        reg_2 = tf.reduce_mean(1 - cosine_probability(double_not_event, event_space))

        event_or_F = self.or_cell(event_space, F_vec)
        reg_7 = tf.reduce_mean(1 - cosine_probability(event_or_F, event_space))

        event_or_T = self.or_cell(event_space, self.T)
        reg_8 = tf.reduce_mean(1 - cosine_probability(event_or_T, self.T))

        event_or_event = self.or_cell(event_space, event_space)
        reg_9 = tf.reduce_mean(1 - cosine_probability(event_or_event, event_or_event))

        event_or_not_event = self.or_cell(event_space, not_event)
        reg_10 = tf.reduce_mean(1 - cosine_probability(event_or_not_event, self.T))

        return reg_1 + reg_2 + reg_7 + reg_8 + reg_9 + reg_10

    def get_hyper_parameter(self):
        '''
        Return all hyper-parameters.
        '''
        params = {
            'user_embedding_dim': self.user_embedding_dim,
            'item_embedding_dim': self.item_embedding_dim,
            'hidden1_dim': self.hidden1_dim,
            'hidden2_dim': self.hidden2_dim,
            'num_users': self.num_users,
            'num_items': self.num_items,
            'learning_rate': self.learning_rate,
            'l2_weight': self.l2_weight,
            'logical_weight': self.logical_weight,
            'warmup_steps': self.warmup_steps,
            'interact_type': self.interact_type
        }
        return params
