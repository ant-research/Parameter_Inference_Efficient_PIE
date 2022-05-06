# -*- coding: utf-8 -*-
#
# sampler.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import math
import numpy as np
import scipy as sp
import dgl.backend as F
import dgl
import os
import sys
import pickle
import time
import torch
import scipy
from dgl.base import NID, EID


def ConstructGraph(dataset, args):
    """Construct Graph for training
    Parameters
    ----------
    dataset :
        the dataset
    args :
        Global configs.
    """
    src = [dataset.train[0]]
    etype_id = [dataset.train[1]]
    dst = [dataset.train[2]]
    num_train = len(dataset.train[0])
    # get node degree from train data
    train_src = np.concatenate(src)
    train_dst = np.concatenate(dst)
    coo = sp.sparse.coo_matrix((np.ones(len(train_src)), (train_src, train_dst)), shape=[
                               dataset.n_entities, dataset.n_entities])
    g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
    in_degree = g.in_degrees()
    out_degree = g.out_degrees()
    del g, coo
    # add reverse triples
    src.append(dataset.train[2])
    dst.append(dataset.train[0])
    etype_id.append(dataset.train[1]+dataset.n_relations)
    if args.dataset == "wikikg90M":
        valid_dict = dataset.valid_dict
        num_valid = len(valid_dict['h,r->t']['hr'])
    elif hasattr(dataset, 'valid') and dataset.valid is not None:
        # src.append(dataset.valid[0])
        # etype_id.append(dataset.valid[1])
        # dst.append(dataset.valid[2])
        num_valid = len(dataset.valid[0])
    else:
        num_valid = 0
    if args.dataset == "wikikg90M":
        test_dict = dataset.test_dict
        num_test = len(test_dict['h,r->t']['hr'])
    elif hasattr(dataset, 'test') and dataset.test is not None:
        # src.append(dataset.test[0])
        # etype_id.append(dataset.test[1])
        # dst.append(dataset.test[2])
        num_test = len(dataset.test[0])
    else:
        num_test = 0
    src = np.concatenate(src)
    etype_id = np.concatenate(etype_id)
    dst = np.concatenate(dst)
    n_entities = dataset.n_entities
    coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
    g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
    g.edata['tid'] = F.tensor(etype_id, F.int64)
    if args.has_edge_importance:
        e_impts = F.tensor(dataset.train[3], F.float32)
        e_impts_vt = F.zeros((num_valid + num_test,), dtype=F.float32, ctx=F.cpu())
        g.edata['impts'] = F.cat([e_impts, e_impts_vt], dim=0)
    return g, in_degree, out_degree


class EvalSampler(object):
    """Sampler for validation and testing
    Parameters
    ----------
    g : DGLGraph
        Graph containing KG graph
    edges : tensor
        Seed edges
    batch_size : int
        Batch size of each mini batch.
    neg_sample_size : int
        How many negative edges sampled for each node.
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    mode : str
        Sampling mode.
    number_workers: int
        Number of workers used in parallel for this sampler
    """

    def __init__(self, g, edges, mode, batch_size, num_hops, expand_factors, num_candidates, node_degree, num_workers=32, e2r=None, dataset='wikikg90M'):
        NodeSampler = getattr(dgl.contrib.sampling, 'NeighborSampler')
        assert(batch_size == 1)
        self.g = g
        self.mode = mode
        if dataset != 'wikikg90M':
            if mode == 'head-batch':
                seed_nodes = edges[:, 2]
                target_nodes = edges[:, 0]
            else:
                seed_nodes = edges[:, 0]
                target_nodes = edges[:, 2]
            relations = edges[:, 1]
        else:
            assert(self.mode == 'tail-batch')
            seed_nodes = edges[0][:, 0]
            relations = edges[0][:, 1]
            target_nodes = edges[1]
            edges = edges[0]
        neighbor_type = 'out'
        if self.mode == 'head-batch':
            neighbor_type = 'in'
        self.sampler = NodeSampler(g,
                                   batch_size=batch_size,
                                   seed_nodes=seed_nodes,
                                   expand_factor=expand_factors,
                                   num_hops=num_hops,
                                   neighbor_type=neighbor_type,
                                   num_workers=num_workers)
        self.node_degree = node_degree
        self.num_edges = len(edges)
        self.batch_size = batch_size
        self.num_hops = num_hops
        self.num_candidates = num_candidates
        self.e2r = e2r
        if not type(e2r) is np.ndarray:
            self.sparse_prob = True
        else:
            self.sparse_prob = False
        """
        if e2r is not None:
            if type(e2r) is np.ndarray:
                self.e2r = torch.tensor(e2r)
            else:
                self.e2r = torch.sparse_csr_tensor(e2r)
        else:
            self.e2r = None
        """
        self.sampler_iter = iter(self.sampler)
        self.sampler_paired_relation_iter = iter(relations)
        if target_nodes is not None:
            self.sampler_paired_target_iter = iter(target_nodes)
        else:
            self.sampler_paired_target_iter = None

    def __iter__(self):
        return self

    def __next__(self):
        g = next(self.sampler_iter)
        r = next(self.sampler_paired_relation_iter)
        center_id = g.map_to_parent_nid(g.layer_nid(self.num_hops))
        ids = []
        for layer in range(self.num_hops):
            ids.append(g.map_to_parent_nid(g.layer_nid(layer)))
        unique_ids = torch.unique(torch.cat(ids))
        #target = next(self.sampler_paired_target_iter)
        #find = len(torch.nonzero(unique_ids == target))
        if len(unique_ids) == 0:
            print(g.layer_nid(self.num_hops))
            print(g.map_to_parent_nid(g.layer_nid(self.num_hops)))
        # 1, delete seed node
        unique_ids = unique_ids[unique_ids != center_id]
        # 2, filter true answers existed in train
        eids_one_hop = g.map_to_parent_eid(g.block_eid(self.num_hops-1))
        edges_one_hop = self.g.find_edges(eids_one_hop)
        edges_relations = self.g.edata['tid'][eids_one_hop]
        if self.mode == 'tail-batch':
            true_ans = edges_one_hop[1][edges_relations == r]
        else:
            true_ans = edges_one_hop[0][edges_relations == r]
        if self.sampler_paired_target_iter is not None:
            target = next(self.sampler_paired_target_iter)
            true_ans = true_ans[true_ans != target]
        if len(true_ans) > 0:
            mask = torch.ones(unique_ids.numel(), dtype=torch.bool)
            index = torch.nonzero((unique_ids.unsqueeze(1) == true_ans.unsqueeze(0)))[:, 0]
            mask[index] = False
            unique_ids = unique_ids[mask]
        if len(unique_ids) > self.num_candidates:
            if self.sparse_prob:
                # e2r_prob = np.array(self.e2r[unique_ids][:, r].todense()).squeeze()
                # ent_degree = self.node_degree[unique_ids].squeeze()
                # _, ids_degree_topk_indices = torch.topk(ent_degree*e2r_prob, self.num_candidates)
                # fast computation
                e2r_matrix = self.e2r[unique_ids][:, r]
                unique_ids = unique_ids[e2r_matrix.nonzero()[0]]
                e2r_prob = e2r_matrix.data
                if len(unique_ids) > self.num_candidates:
                    e2r_prob = e2r_matrix.data
                    ent_degree = self.node_degree[unique_ids]
                    _, ids_degree_topk_indices = torch.topk(
                        ent_degree*e2r_prob, self.num_candidates)
                    unique_ids = unique_ids[ids_degree_topk_indices]
            else:
                e2r_prob = self.e2r[unique_ids][:, r]
                ent_degree = self.node_degree[unique_ids]
                _, ids_degree_topk_indices = torch.topk(ent_degree*e2r_prob, self.num_candidates)
                unique_ids = unique_ids[ids_degree_topk_indices]
        if len(unique_ids) < self.num_candidates:
            unique_ids = torch.cat([unique_ids, torch.tensor(
                [-1]*(self.num_candidates - len(unique_ids)))])
        find = len(torch.nonzero(unique_ids == target))
        return unique_ids, find, torch.stack((center_id[0], torch.tensor(r), torch.tensor(target))).unsqueeze(0)


class EvalDataset(object):
    """Dataset for validation or testing
    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    """

    def __init__(self, g, dataset, args):
        # reverse triples are added
        self.num_train = len(dataset.train[0])*2
        self.g = g
        self.e2r = dataset.e2r
        self.dataset = args.dataset
        if args.dataset == "wikikg90M":
            self.valid_dict = dataset.valid_dict
            self.num_valid = len(self.valid_dict['h,r->t']['hr'])
        elif dataset.valid is not None:
            self.num_valid = len(dataset.valid[0])
            self.valid = dataset.valid
        else:
            self.num_valid = 0
        if args.dataset == "wikikg90M":
            self.test_dict = dataset.test_dict
            self.num_test = len(self.test_dict['h,r->t']['hr'])
        elif dataset.test is not None:
            self.num_test = len(dataset.test[0])
            self.test = dataset.test
        else:
            self.num_test = 0
        print('|train|:', self.num_train)
        print('|valid|:', self.num_valid)
        print('|test|:', self.num_test)

    def get_edges(self, eval_type):
        """ Get all edges in this dataset
        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        Returns
        -------
        np.array
            Edges
        """
        if eval_type == 'valid':
            h = np.expand_dims(self.valid[0], axis=1)
            r = np.expand_dims(self.valid[1], axis=1)
            t = np.expand_dims(self.valid[2], axis=1)
            return np.concatenate([h, r, t], axis=1)
        elif eval_type == 'test':
            h = np.expand_dims(self.test[0], axis=1)
            r = np.expand_dims(self.test[1], axis=1)
            t = np.expand_dims(self.test[2], axis=1)
            return np.concatenate([h, r, t], axis=1)
        else:
            raise Exception('get invalid type: ' + eval_type)

    def get_dicts(self, eval_type):
        """ Get all edges dict in this dataset
        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        Returns
        -------
        dict
            all edges
        """
        if eval_type == 'valid':
            return self.valid_dict
        elif eval_type == 'test':
            return self.test_dict
        else:
            raise Exception('get invalid type: ' + eval_type)

    def create_sampler(self, eval_type, batch_size, num_hops, expand_factors, mode, node_degree, num_candidates, num_workers=32, rank=0, ranks=1):
        """Create sampler for validation or testing
        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        filter_false_neg : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.
        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        if self.dataset == 'wikikg90M':
            edges = self.get_dicts(eval_type)['h,r->t']['hr']
            targets = self.get_dicts(eval_type)['h,r->t']['t']
            beg = edges.shape[0] * rank // ranks
            end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
            edges = (edges[beg:end], targets[beg:end])
        else:
            edges = self.get_edges(eval_type)
            beg = edges.shape[0] * rank // ranks
            end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
            edges = edges[beg: end]
        return EvalSampler(self.g, edges, mode, batch_size, num_hops, expand_factors, num_candidates, node_degree, num_workers, self.e2r, self.dataset)
