#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
import torch

from torch.utils.data import Dataset
from scipy.sparse import coo_matrix


class TrainTestDataset(Dataset):
    def __init__(self, triples, nrelation, entity2edges, edge2entities, edge2relation, entity2relation, context_hops):
        self.len = len(triples)
        self.triples = triples
        self.nrelation = nrelation
        self.entity2edges = entity2edges
        self.edge2entities = edge2entities
        self.edge2relation = edge2relation
        self.entity2relation = entity2relation
        self.context_hops = context_hops

        self.__getitem__(0)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        label = torch.tensor(relation)
        label_set = np.zeros((self.nrelation,))
        for i in self.entity2relation[tail]:
            label_set[i] = 1
        label_set = torch.tensor(label_set)

        # subgraph sampling
        tail_edges_list, tail_masks, tail_edge2relation_list = self._get_neighbors_and_masks(
            relation, tail, idx)
        entity_pair = torch.LongTensor([head, tail])

        return entity_pair, label, label_set, tail_edges_list, tail_masks, tail_edge2relation_list

    @staticmethod
    def collate_fn(data):
        entity_pairs = torch.stack([_[0] for _ in data], dim=0)
        labels = torch.stack([_[1] for _ in data], dim=0)
        label_sets = torch.stack([_[2] for _ in data], dim=0)

        tail_edges_lists = []
        for i in range(len(data[0][3])):
            tail_edges_lists.append(torch.stack([torch.tensor(_[3][i]) for _ in data], dim=0))

        tail_masks = []
        for i in range(1, len(data[0][4])):
            tail_masks.append(torch.stack([torch.tensor(_[4][i]) for _ in data], dim=0))

        tail2relation = []
        for i in range(1, len(data[0][5])):
            tail2relation.append(torch.stack([torch.tensor(_[5][i]) for _ in data], dim=0))

        return entity_pairs, labels, label_sets, tail_edges_lists, tail_masks, tail2relation

    def _get_neighbors_and_masks(self, relation, entity, edge):
        edges_list = [relation]
        masks = [1]
        edge2relation_list = [1]
        for i in range(self.context_hops):
            if i == 0:
                neighbor_entities = entity
            else:
                neighbor_entities = np.take(self.edge2entities, edges_list[-1], 0)
            neighbor_edges = np.take(self.entity2edges, neighbor_entities, 0)

            edges_list.append(neighbor_edges)

            mask = neighbor_edges - edge
            mask = (mask != 0)

            relations = np.take(self.edge2relation, edges_list[-1], 0)
            # remove null relation
            mask1 = relations - self.nrelation
            mask1 = (mask1 != 0)
            mask = mask * mask1

            masks.append(mask)
            edge2relation_list.append(relations)
        return np.array(edges_list), np.array(masks), np.array(edge2relation_list)


class OneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)

    def __next__(self):
        data = next(self.iterator)
        return data

    @ staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
