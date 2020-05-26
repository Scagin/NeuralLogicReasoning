import numpy as np
import tensorflow as tf

UNKNOWN_TAG = '<unk>'
PADDING_TAG = '<pad>'
TAGS = [UNKNOWN_TAG, PADDING_TAG]


def load_train_datas(data_path, feed_back=True):
    with open(data_path, 'r') as f:
        raw_lines = f.readlines()

    users, hist_items, scores, labels = [], [], [], []
    all_items = []
    for line in raw_lines:
        user, history_seq, label = line.strip().split('\t', 2)
        history_seq = history_seq.split('|')

        users.append(user)
        labels.append(label)

        all_items.append(label)

        if feed_back:
            history_seq = [hist.split(',') for hist in history_seq]
            hist_ = [hist[0] for hist in history_seq]
            hist_items.append(hist_)
            all_items.extend(hist_)
            scores.append([int(hist[1]) for hist in history_seq])
        else:
            history_seq = [hist.split(',')[0] for hist in history_seq]
            hist_items.append(history_seq)
            all_items.extend(history_seq)
            scores.append([1] * len(history_seq))

    user_2_id = {user: i + len(TAGS) for i, user in enumerate(set(users))}
    item_2_id = {item: i + len(TAGS) for i, item in enumerate(set(all_items))}
    for i, tag in enumerate(TAGS):
        user_2_id[tag] = i
        item_2_id[tag] = i

    return users, hist_items, scores, labels, user_2_id, item_2_id


def load_test_datas(data_path, feed_back=True):
    with open(data_path, 'r') as f:
        raw_lines = f.readlines()

    users, hist_items, scores, labels = [], [], [], []
    for line in raw_lines:
        user, history_seq, label = line.strip().split('\t', 2)
        history_seq = history_seq.split('|')

        users.append(user)
        labels.append(label)

        if feed_back:
            history_seq = [hist.split(',') for hist in history_seq]
            hist_ = [hist[0] for hist in history_seq]
            hist_items.append(hist_)
            scores.append([int(hist[1]) for hist in history_seq])
        else:
            history_seq = [hist.split(',')[0] for hist in history_seq]
            hist_items.append(history_seq)
            scores.append([1] * len(history_seq))

    return users, hist_items, scores, labels


def batch_iterator(users, hist_items, feedback_scores, labels, user_2_id, item_2_id, history_len=5,
                   batch_size=128, shuffle=True):
    # string to index_id
    user_data = [[user_2_id.get(user, user_2_id[UNKNOWN_TAG])] for user in users]
    label_data = [[item_2_id.get(label, item_2_id[UNKNOWN_TAG])] for label in labels]
    hist_data = []
    for hist in hist_items:
        hist_ = [item_2_id.get(h, item_2_id[UNKNOWN_TAG]) for h in hist]
        hist_data.append(hist_)

    # padding
    hist_data = tf.keras.preprocessing.sequence.pad_sequences(hist_data, maxlen=history_len,
                                                              value=item_2_id[PADDING_TAG])
    feedback_data = tf.keras.preprocessing.sequence.pad_sequences(feedback_scores,
                                                                  maxlen=history_len, value=0)
    user_data = np.array(user_data)
    label_data = np.array(label_data)

    # shuffle datas
    if shuffle:
        indices = np.random.permutation(range(len(labels)))
        hist_data = hist_data[indices]
        feedback_data = feedback_data[indices]
        user_data = user_data[indices]
        label_data = label_data[indices]

    # negative sampling
    negative_samples = []
    for label in label_data:
        neg_sample = np.random.randint(0, len(item_2_id), size=[1])
        while label == neg_sample:
            neg_sample = np.random.randint(0, len(item_2_id), size=[1])
        negative_samples.append(neg_sample)
    negative_samples = np.array(negative_samples)

    num_batch = int((len(labels) - 1) / batch_size + 1)
    for i in range(num_batch):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(labels))

        yield user_data[start: end], hist_data[start: end], feedback_data[start: end], \
              label_data[start: end], negative_samples[start: end]


def test_batch(user, hist_items, feedback_score, user_2_id, item_2_id):
    item_num = len(item_2_id)
    user_data = np.array([[user_2_id.get(user, user_2_id[UNKNOWN_TAG])]] * item_num)
    items_data = np.array(
        [[item_2_id.get(item, item_2_id[UNKNOWN_TAG]) for item in hist_items]] * item_num)
    feedback_data = np.array([feedback_score] * item_num)
    return user_data, items_data, feedback_data
