import argparse
from data_processer import load_data, load_data_wikikg
from train import train, infer
import os


def print_setting(args):
    print()
    print('=============================================')
    print('dataset: ' + args.dataset)
    print('steps: ' + str(args.steps))
    print('batch_size: ' + str(args.batch_size))
    print('dim: ' + str(args.dim))
    print('l2: ' + str(args.l2))
    print('lr: ' + str(args.lr))
    print('feature_type: ' + args.feature_type)
    print('add_reverse: ' + str(args.add_reverse))
    print('use_bce: ' + str(args.use_bce))
    print('use_ranking_loss: ' + str(args.use_ranking_loss))
    print('ranking_loss_margin: ' + str(args.margin))
    print('ranking_loss_gamma: ' + str(args.gamma))

    print('context_hops: ' + str(args.context_hops))
    print('neighbor_samples: ' + str(args.neighbor_samples))
    print('neighbor_agg: ' + args.neighbor_agg)

    print('=============================================')
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, help='use gpu', action='store_true')

    parser.add_argument('--dataset', type=str, default='YAGO3-10', help='dataset name')
    parser.add_argument('--steps', type=int, default=10000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--cpu_num', type=int, default=16, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--l2', type=float, default=1e-7, help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=5e-3, help='learning rate')
    parser.add_argument('--feature_type', type=str, default='id',
                        help='type of relation features: id, bow, bert')

    # settings for model
    parser.add_argument('--add_reverse', type=bool, default=True,
                        help='whether add reverse triples')
    parser.add_argument('--context_hops', type=int, default=2, help='number of context hops')
    parser.add_argument('--neighbor_samples', type=int, default=8,
                        help='number of sampled neighbors for one hop')
    parser.add_argument('--neighbor_agg', type=str, default='pna',
                        help='neighbor aggregator: mean, sum, pna')
    parser.add_argument('--neg_sample_num', type=int, default=10,
                        help='number of sampled neighbors for one hop')
    parser.add_argument('--uni_weight', default=False, help='sample weight', action='store_true')
    parser.add_argument('--use_bce', default=False, help='loss type', action='store_true')
    parser.add_argument('--use_ranking_loss', type=bool, default=True, help='Ranking loss')
    parser.add_argument('--gamma', type=float,
                        default=1.0, help='max length of a path')
    parser.add_argument('--margin', type=float,
                        default=3.0, help='max length of a path')

    # settings for wikikg90mv2 dataset
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--nentity', type=int, default=91230610, help='max length of a path')
    parser.add_argument('--nrelation', type=int, default=1387, help='max length of a path')

    args = parser.parse_args()
    args.save_path = os.path.join('./models/', args.dataset)
    print(args.save_path)
    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print_setting(args)

    if args.dataset == 'WikiKG90Mv2':
        data = load_data_wikikg(args, args.nentity, args.nrelation, args.data_path)
    else:
        data = load_data(args)

    train(args, data)
    infer(args, data)


if __name__ == '__main__':
    main()
