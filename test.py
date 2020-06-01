import tqdm
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

        topk = 10
        num_right_samples = 0
        for user, hist, feedback, label in tqdm.tqdm(zip(test_users, test_hist_items, test_scores,
                                                         test_labels)):
            user_data, items_data, feedback_data = data_loader.test_batch(user, hist, feedback,
                                                                          user_2_id, item_2_id)

            prob = sess.run(model.probability_pos,
                            feed_dict={model.input_user: user_data, model.input_items: items_data,
                                       model.input_feedback_score: feedback_data,
                                       model.target_emb_vec: items_embedding_matrix})

            prob = np.squeeze(prob, axis=1)
            pred_item_ids = np.argsort(prob, axis=0)[:topk]
            pred_items = [id_2_item.get(int(id), '<unk>') for id in pred_item_ids]
            sorted_prob = np.sort(prob, axis=0)[:topk]
            if label in pred_items:
                num_right_samples += 1

        print('top {} accuracy: {}'.format(topk, num_right_samples / len(test_labels)))


if __name__ == '__main__':
    test()
