import argparse


class HyperParams:
    parser = argparse.ArgumentParser()

    # train
    ## files
    parser.add_argument('--train_datas', default='dataset/ml-100k/train.data',
                        help='training data.')
    parser.add_argument('--eval_datas', default='dataset/ml-100k/eval.data',
                        help='evaluating data.')
    parser.add_argument('--is_with_feedback', default=True,
                        help='the history with explicit feedback or not.')
    parser.add_argument('--checkpoint_dir', default='model_ckpt/test1',
                        help='model save directory.')
    parser.add_argument('--ckpt_name', default='nlr', help='the save model name.')
    parser.add_argument('--tensorboard_dir', default='tensorboard/test1',
                        help='tensorboard log directory.')

    # training scheme
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--l2_weight', default=1e-4, type=int, help='the weight of L2 loss.')
    parser.add_argument('--logical_weight', default=0.1, type=int,
                        help='the weight of logical regularizer loss.')
    parser.add_argument('--history_len', default=5, type=int, help='length of historical items.')
    parser.add_argument('--warmup_steps', default=500, type=int,
                        help='warm up steps for adam optimizer.')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--num_epochs', default=20, type=int)
    parser.add_argument('--eval_per_steps', default=100, type=int, help='evaluate per steps.')

    # model
    parser.add_argument('--user_emb_dim', default=64, type=int,
                        help='embedding dimension of user.')
    parser.add_argument('--item_emb_dim', default=64, type=int,
                        help='embedding dimension of item.')
    parser.add_argument('--hidden1_dim', default=128, type=int,
                        help='fisrt hidden layer in logical modules dimension.')
    parser.add_argument('--hidden2_dim', default=64, type=int,
                        help='second hidden layer in logical modules dimension.')
    parser.add_argument('--interact_type', default='concat', type=str,
                        help='interact type between user embedding and item embedding.')

    # test
    parser.add_argument('--test_datas', default='dataset/ml-100k/test.data', help='test data')
    parser.add_argument('--ckpt', default='model_ckpt/test1/nlr-10100', help='checkpoint file path')
    parser.add_argument('--topk', default=10, help='how many items return from sorted result list.')
