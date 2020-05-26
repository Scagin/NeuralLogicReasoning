import os


def save_training_info(user2id, item2id, path):
    with open(os.path.join(path, 'user2id.txt'), 'w') as f:
        for key, value in user2id.items():
            print('{} {}'.format(key, value), file=f)

    with open(os.path.join(path, 'item2id.txt'), 'w') as f:
        for key, value in item2id.items():
            print('{} {}'.format(key, value), file=f)


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

    return user2id, item2id

