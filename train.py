import os
import logging
import numpy as np
import tensorflow as tf

import utils
import data_loader
from model import NLR_model
from hyper_params import HyperParams


def evaluate(sess, model, users, hist_items, scores, labels, user_2_id, item_2_id, hist_len=5,
             batch_size=128):
    batch_iter = data_loader.batch_iterator(users, hist_items, scores, labels, user_2_id, item_2_id,
                                            history_len=hist_len, batch_size=batch_size,
                                            shuffle=False)
    eval_target_loss = 0
    eval_l2_loss = 0
    eval_logical_loss = 0
    eval_loss = 0
    eval_pos_prob = 0
    eval_neg_prob = 0
    for i, batch in enumerate(batch_iter):
        user_batch, items_batch, feedback_batch, label_batch, neg_batch = batch

        _pos_prob, _neg_prob, _target_loss, _l2_loss, _logical_loss, _loss = sess.run(
            [model.probability_pos, model.probability_neg, model.traget_loss,
             model.l2_loss, model.logical_loss, model.loss],
            feed_dict={model.input_user: user_batch, model.input_items: items_batch,
                       model.input_feedback_score: feedback_batch,
                       model.input_negative_sample: neg_batch,
                       model.input_target: label_batch})

        eval_loss += _loss
        eval_target_loss += _target_loss
        eval_l2_loss += _l2_loss
        eval_logical_loss += _logical_loss
        eval_pos_prob += np.sum(_pos_prob)
        eval_neg_prob += np.sum(_neg_prob)

    eval_target_loss /= (i + 1)
    eval_l2_loss /= (i + 1)
    eval_logical_loss /= (i + 1)
    eval_loss /= (i + 1)
    eval_pos_prob /= len(labels)
    eval_neg_prob /= len(labels)

    return eval_target_loss, eval_l2_loss, eval_logical_loss, \
           eval_loss, eval_pos_prob, eval_neg_prob


def train():
    hparams = HyperParams()

    parser = hparams.parser
    hp = parser.parse_args()

    train_users, train_hist_items, train_scores, \
    train_labels, user_2_id, item_2_id = data_loader.load_train_datas(hp.train_datas,
                                                                      hp.is_with_feedback)

    eval_users, eval_hist_items, eval_scores, eval_labels, _, _ = data_loader.load_train_datas(
        hp.eval_datas, hp.is_with_feedback)

    model = NLR_model(user_embedding_dim=hp.user_emb_dim, item_embedding_dim=hp.item_emb_dim,
                      hidden1_dim=hp.hidden1_dim, hidden2_dim=hp.hidden2_dim,
                      num_users=len(user_2_id), num_items=len(item_2_id), learning_rate=hp.lr,
                      l2_weight=hp.l2_weight, warmup_steps=hp.warmup_steps)

    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        # initialize / restore
        ckpt = tf.train.latest_checkpoint(hp.checkpoint_dir)
        if ckpt is None:
            logging.info('Initializing from scratch')
            sess.run(tf.global_variables_initializer())
            utils.save_training_info(user_2_id, item_2_id, hp.checkpoint_dir)
        else:
            saver.restore(sess, ckpt)

        summary_writer = tf.summary.FileWriter(hp.tensorboard_dir, sess.graph)

        best_status = 99999
        num_batch = int((len(train_labels) - 1) / hp.batch_size + 1)
        for epoch in range(hp.num_epochs):
            batch_iter = data_loader.batch_iterator(train_users, train_hist_items, train_scores,
                                                    train_labels, user_2_id, item_2_id,
                                                    history_len=hp.history_len,
                                                    batch_size=hp.batch_size)
            for i, batch in enumerate(batch_iter):
                user_batch, items_batch, feedback_batch, label_batch, neg_batch = batch

                current_step = epoch * num_batch + i
                # evaluate
                if current_step % hp.eval_per_steps == 0:
                    # evaluate train dataset
                    _pos_prob, _neg_prob, _target_loss, _l2_loss, \
                    _logical_loss, _loss, _summary = sess.run(
                        [model.probability_pos, model.probability_neg, model.traget_loss,
                         model.l2_loss, model.logical_loss, model.loss, model.summaries],
                        feed_dict={model.input_user: user_batch, model.input_items: items_batch,
                                   model.input_feedback_score: feedback_batch,
                                   model.input_negative_sample: neg_batch,
                                   model.input_target: label_batch})

                    # evaluate validation dataset
                    eval_target_loss, eval_l2_loss, eval_logical_loss, \
                    eval_loss, eval_pos_prob, eval_neg_prob = evaluate(sess, model, eval_users,
                                                                       eval_hist_items, eval_scores,
                                                                       eval_labels, user_2_id,
                                                                       item_2_id, hp.history_len,
                                                                       hp.eval_batch_size)
                    summary_writer.add_summary(_summary, global_step=current_step)

                    # save
                    if eval_loss <= best_status:
                        is_best = '*'
                        best_status = eval_loss
                        saver.save(sess, os.path.join(hp.checkpoint_dir, hp.ckpt_name),
                                   global_step=current_step)
                    else:
                        is_best = ''

                    print('\nepoch: {}, step: {}, train pos prob: {:.4f}, train neg prob: {:.4f}, '
                          'train target loss: {:.4f}, train l2 loss: {:.4f}, '
                          'train logical loss: {:.4f}, train loss: {:.4f}\n'
                          '\t\t\t\t\t\t\teval pos prob: {:.4f}, eval neg prob: {:.4f}, '
                          'eval target loss: {:.4f}, eval l2 loss: {:.4f}, '
                          'eval logical loss: {:.4f}, eval loss: {:.4f} {}'
                          .format(epoch, current_step, np.mean(_pos_prob), np.mean(_neg_prob),
                                  _target_loss, _l2_loss, _logical_loss, _loss,
                                  eval_pos_prob, eval_neg_prob, eval_target_loss, eval_l2_loss,
                                  eval_logical_loss, eval_loss, is_best))

                # train
                _ = sess.run(model.train_op, feed_dict={model.input_user: user_batch,
                                                        model.input_items: items_batch,
                                                        model.input_feedback_score: feedback_batch,
                                                        model.input_negative_sample: neg_batch,
                                                        model.input_target: label_batch})


if __name__ == '__main__':
    train()
