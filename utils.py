import os
import json
import numpy as np


def save_training_info(user2id, item2id, hypers, path):
    with open(os.path.join(path, 'user2id.txt'), 'w') as f:
        for key, value in user2id.items():
            print('{} {}'.format(key, value), file=f)

    with open(os.path.join(path, 'item2id.txt'), 'w') as f:
        for key, value in item2id.items():
            print('{} {}'.format(key, value), file=f)

    with open(os.path.join(path, 'hypers.json'), 'w') as f:
        json.dump(hypers, f)


def load_training_info(path):
    user2id = {}
    with open(os.path.join(path, 'user2id.txt'), 'r') as lines:
        for line in lines:
            key, value = line.strip().split(' ', 1)
            user2id[key] = int(value)

    item2id = {}
    with open(os.path.join(path, 'item2id.txt'), 'r') as lines:
        for line in lines:
            key, value = line.strip().split(' ', 1)
            item2id[key] = int(value)

    with open(os.path.join(path, 'hypers.json'), 'r') as f:
        hypers = json.load(f)

    return user2id, item2id, hypers


def calDCG(scores):
    return np.sum(
        np.divide(scores,
                  np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)


def calNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = calDCG(relevance)

    dcg = calDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg
