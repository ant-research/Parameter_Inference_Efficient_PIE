import os
import gc
import time
import sys
import math
import torch
import numpy as np
from tqdm import tqdm
from KGDataset import get_dataset
from sampler import ConstructGraph, EvalDataset
from utils import CommonArgParser, thread_wrapped_func
import torch.multiprocessing as mp


class ArgParser(CommonArgParser):
    def __init__(self):
        super(ArgParser, self).__init__()
        self.add_argument('--has_edge_importance', action='store_true',
                          help='Allow providing edge importance score for each edge during training.'
                          'The positive score will be adjusted '
                          'as pos_score = pos_score * edge_importance')
        self.add_argument('--valid', action='store_true',
                          help='Evaluate the model on the validation set in the training.')
        self.add_argument('--num_hops', type=int, default=2, help='.')
        self.add_argument('--expand_factors', type=int, default=1000000, help='.')
        self.add_argument('--num_workers', type=int, default=16, help='.')
        self.add_argument('--print_on_screen', action='store_true', help='')
        self.add_argument('--num_candidates', type=int, default=20000, help='')
        self.add_argument('--save_file', type=str, default="test_tail_candidate", help='')


def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)


def infer(args, samplers, save_paths):
    candidates, spos = [], []
    find, total = 0, 0
    for sampler in samplers:
        for candidate, is_find, spo in tqdm(sampler, disable=not args.print_on_screen, total=sampler.num_edges):
            candidates.append(candidate.unsqueeze(0))
            spos.append(spo)
            if is_find == 1:
                find += 1
            total += 1
            if total % 100 == 0:
                print("%d/%d=%f" % (find, total, float(find)/float(total)))
    candidates = torch.cat(candidates, axis=0)
    spos = torch.cat(spos, axis=0)
    ret = torch.cat([spos, candidates], axis=1).numpy()
    return np.save(save_paths[0], ret)


@thread_wrapped_func
def infer_mp(args, samplers, save_paths, rank=0, mode='Test'):
    if args.num_proc > 1:
        torch.set_num_threads(args.num_thread)
    infer(args, samplers, save_paths)


def main():
    args = ArgParser().parse_args()
    prepare_save_path(args)
    dataset = get_dataset(args.data_path, args.dataset, args.format,
                          args.delimiter, args.data_files, False)
    g, in_degree, out_degree = ConstructGraph(dataset, args)
    if args.valid or args.test:
        eval_dataset = EvalDataset(g, dataset, args)
    if args.num_proc > 1:
        if args.valid:
            valid_samplers, save_paths = [], []
            for i in range(args.num_proc):
                valid_sampler = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                            args.num_hops,
                                                            args.expand_factors,
                                                            'tail-batch',
                                                            in_degree,
                                                            num_workers=args.num_workers,
                                                            num_candidates=args.num_candidates,
                                                            rank=i, ranks=args.num_proc)
                save_file = args.save_file + '_' + \
                    str(args.num_candidates) + '_' + str(i) + '.npy'
                if args.dataset == 'wikikg90M':
                    save_path = os.path.join(args.data_path, 'wikikg90m-v2/processed/', save_file)
                else:
                    save_path = os.path.join(args.data_path, args.dataset, save_file)
                save_paths.append(save_path)
                valid_samplers.append(valid_sampler)
            procs = []
            for i in range(args.num_proc):
                proc = mp.Process(target=infer_mp, args=(
                    args, [valid_samplers[i]], [save_paths[i]]))
                procs.append(proc)
                proc.start()
            for proc in procs:
                proc.join()
    else:
        if args.valid:
            valid_sampler = eval_dataset.create_sampler('valid', args.batch_size_eval,
                                                        args.num_hops,
                                                        args.expand_factors,
                                                        'tail-batch',
                                                        in_degree,
                                                        num_workers=args.num_workers,
                                                        num_candidates=args.num_candidates)
            save_file = args.save_file + '_' + str(args.num_candidates) + '.npy'
            if args.dataset == 'wikikg90M':
                save_path = os.path.join(args.data_path, 'wikikg90m-v2/processed/', save_file)
            else:
                save_path = os.path.join(args.data_path, args.dataset, save_file)
            candidates, spos = infer(args, [valid_sampler], [save_path])
            np.save(save_path, candidates.numpy())


if __name__ == '__main__':
    main()
