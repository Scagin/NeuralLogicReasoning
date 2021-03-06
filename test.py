import tqdm
import random
import numpy as np
import tensorflow as tf

import utils
import data_loader
from model import NLR_model
from hyper_params import HyperParams


def test():
    hparams = HyperParams()

    parser = hparams.parser
    hp = parser.parse_args()

    test_users, test_hist_items, test_scores, test_labels = data_loader.load_test_datas(
        hp.test_datas, hp.is_with_feedback)

    user_2_id, item_2_id, train_hypers = utils.load_training_info(hp.checkpoint_dir)
    id_2_item = {id: item for item, id in item_2_id.items()}

    model = NLR_model(user_embedding_dim=train_hypers.get('user_embedding_dim'),
                      item_embedding_dim=train_hypers.get('item_embedding_dim'),
                      hidden1_dim=train_hypers.get('hidden1_dim'),
                      hidden2_dim=train_hypers.get('hidden2_dim'),
                      num_users=train_hypers.get('num_users'),
                      num_items=train_hypers.get('num_items'),
                      interact_type=train_hypers.get('interact_type'))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # restore
        saver.restore(sess, hp.ckpt)

        items_embedding_matrix = sess.run(model.item_embedding_layer)
        items_embedding_matrix = items_embedding_matrix[:, np.newaxis, :]

        topk = hp.topk
        count = 0
        hr_total = 0
        ndcg_total = 0
        for user, hist, feedback, label in tqdm.tqdm(zip(test_users, test_hist_items, test_scores,
                                                         test_labels)):
            # if random.random() > 0.02:
            #     continue
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

        print('HR@{}: {:.6f}'.format(topk, hr_total / count))
        print('NDCG@{}: {:.6f}'.format(topk, ndcg_total / count))


if __name__ == '__main__':
    test()
