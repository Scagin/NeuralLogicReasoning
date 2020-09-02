import os
import random
import logging
import numpy as np
import tensorflow as tf

import utils
import data_loader
from model import NLR_model
from hyper_params import HyperParams


def evaluate(sess, model, users, hist_items, scores, labels,
             user_2_id, item_2_id, test_ratio=0.5, topk=5):
    count = 0
    hr_total = 0
    ndcg_total = 0

    items_embedding_matrix = sess.run(model.item_embedding_layer)
    items_embedding_matrix = items_embedding_matrix[:, np.newaxis, :]
    for user, hist, feedback, label in zip(users, hist_items, scores, labels):
        if random.random() > test_ratio:
            continue
        user_data, items_data, feedback_data = data_loader.test_batch(user, hist, feedback,
                                                                      user_2_id, item_2_id)

        prob_pos = sess.run(model.probability_pos,
                            feed_dict={model.input_user: user_data,
                                       model.input_items: items_data,
                                       model.input_feedback_score: feedback_data,
                                       model.target_emb_vec: items_embedding_matrix})

        prob_pos = np.squeeze(prob_pos, axis=1)
        pred_item_ids = np.argsort(prob_pos, axis=0)[::-1][:topk]
        label_ids = [item_2_id.get(label, item_2_id[data_loader.UNKNOWN_TAG])]

        ndcg_score = utils.calNDCG(pred_item_ids, label_ids)
        ndcg_total += ndcg_score

        hr_score = len(set(pred_item_ids).intersection(set(label_ids))) / len(label_ids)
        hr_total += hr_score

        count += 1

    return hr_total / count, ndcg_total / count


def train():
    hparams = HyperParams()

    parser = hparams.parser
    hp = parser.parse_args()

    # read datas
    train_users, train_hist_items, train_scores, \
    train_labels, user_2_id, item_2_id = data_loader.load_train_datas(hp.train_datas,
                                                                      hp.is_with_feedback)

    eval_users, eval_hist_items, eval_scores, eval_labels, _, _ = data_loader.load_train_datas(
        hp.eval_datas, hp.is_with_feedback)

    # build model
    model = NLR_model(user_embedding_dim=hp.user_emb_dim, item_embedding_dim=hp.item_emb_dim,
                      hidden1_dim=hp.hidden1_dim, hidden2_dim=hp.hidden2_dim,
                      num_users=len(user_2_id), num_items=len(item_2_id), learning_rate=hp.lr,
                      l2_weight=hp.l2_weight, warmup_steps=hp.warmup_steps,
                      interact_type=hp.interact_type)

    saver = tf.train.Saver(max_to_keep=5)
    with tf.Session() as sess:
        # initialize / restore
        ckpt = tf.train.latest_checkpoint(hp.checkpoint_dir)
        if ckpt is None:
            logging.info('Initializing from scratch')
            sess.run(tf.global_variables_initializer())
            if not os.path.exists(hp.checkpoint_dir):
                os.mkdir(hp.checkpoint_dir)
            utils.save_training_info(user_2_id, item_2_id, model.get_hyper_parameter(),
                                     hp.checkpoint_dir)
        else:
            saver.restore(sess, ckpt)

        summary_writer = tf.summary.FileWriter(hp.tensorboard_dir, sess.graph)

        stop_flag = False
        best_hr = 0
        best_ndcg = 0
        last_update_step = 0
        max_nonupdate_steps = 100000
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
                    hr_k, ndcg_k = evaluate(sess, model, eval_users, eval_hist_items, eval_scores,
                                            eval_labels, user_2_id, item_2_id, hp.history_len,
                                            hp.eval_batch_size)
                    summary_writer.add_summary(_summary, global_step=current_step)

                    # save
                    if ndcg_k >= best_ndcg or hr_k >= best_hr:
                        is_best = '*'
                        best_ndcg = ndcg_k
                        best_hr = hr_k
                        last_update_step = current_step
                        saver.save(sess, os.path.join(hp.checkpoint_dir, hp.ckpt_name),
                                   global_step=current_step)
                    else:
                        is_best = ''

                    print('\nepoch: {}, step: {}, train pos prob: {:.4f}, train neg prob: {:.4f}, '
                          'train target loss: {:.4f}, train l2 loss: {:.4f}, '
                          'train logical loss: {:.4f}, train loss: {:.4f} '
                          'eval hr@k: {:.4f}, eval ndcg@k: {:.4f} {}'
                          .format(epoch, current_step, np.mean(_pos_prob), np.mean(_neg_prob),
                                  _target_loss, _l2_loss, _logical_loss, _loss, hr_k, ndcg_k,
                                  is_best))

                # train
                _ = sess.run(model.train_op, feed_dict={model.input_user: user_batch,
                                                        model.input_items: items_batch,
                                                        model.input_feedback_score: feedback_batch,
                                                        model.input_negative_sample: neg_batch,
                                                        model.input_target: label_batch})

                if current_step - last_update_step >= max_nonupdate_steps:
                    stop_flag = True
                    break

            if stop_flag:
                break


if __name__ == '__main__':
    train()
