import os
import pandas as pd


current_dir = os.path.split(os.path.realpath(__file__))[0]


def preprocess_movieslens(history_items_num=5):
    r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
    ratings = pd.read_csv('dataset/ml-100k/u.data', sep='\t', names=r_cols, encoding='latin-1')
    ratings = ratings.sort_values('unix_timestamp')
    print(ratings)

    user_items_dict = {}
    for rating in ratings.values:
        user_id, movie_id, rating, timestamp = rating[0], rating[1], rating[2], rating[3]
        if user_id not in user_items_dict:
            user_items_dict[user_id] = []
        feedback = -1 if rating <= 3 else 1
        user_items_dict[user_id].append((movie_id, feedback))

    user_samples = {}
    for user, histories in user_items_dict.items():
        samples = []
        if len(histories) > history_items_num:
            for i in range(history_items_num, len(histories)):
                sample = ['{},{}'.format(hist[0], hist[1]) for hist in histories[i - history_items_num: i]]
                label = histories[i][0]
                samples.append('{}\t{}\t{}'.format(user, '|'.join(sample), label))
        elif len(histories) > 1:
            sample = ['{},{}'.format(hist[0], hist[1]) for hist in histories[:-1]]
            label = histories[-1][0]
            samples.append('{}\t{}\t{}'.format(user, '|'.join(sample), label))
        else:
            samples.append('{}\t{}\t{}'.format(user, '<unk>,1', histories[0][0]))
        user_samples[user] = samples

    train_datas, eval_datas, test_datas = [], [], []
    for user, samples in user_samples.items():
        if len(samples) > 2:
            train_datas.extend(samples[:-2])
            eval_datas.append(samples[-2])
            test_datas.append(samples[-1])
        elif len(samples) == 2:
            train_datas.append(samples[0])
            test_datas.append(samples[1])
        else:
            test_datas.append(samples[0])

    with open('dataset/ml-100k/train.data', 'w') as f:
        _ = [print(sample, file=f) for sample in train_datas]
    with open('dataset/ml-100k/eval.data', 'w') as f:
        _ = [print(sample, file=f) for sample in eval_datas]
    with open('dataset/ml-100k/test.data', 'w') as f:
        _ = [print(sample, file=f) for sample in test_datas]



if __name__ == '__main__':
    preprocess_movieslens()







