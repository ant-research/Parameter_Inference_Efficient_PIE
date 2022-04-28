import os
import re
import pickle
import random
import pylab
import os.path as osp
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
from utils import count_all_paths_with_mp, count_paths, get_path_dict_and_length, one_hot_path_id, sample_paths


e2re = defaultdict(set)  # entity index -> set of pair (relation, entity) connecting to this entity


def read_entities(file_name):
    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)
    file.close()
    return d


def read_relations(file_name, add_reverse):
    d = {}
    file = open(file_name)
    for line in file:
        index, name = line.strip().split('\t')
        d[name] = int(index)
    file.close()

    rel_num = len(d)
    if add_reverse:
        file = open(file_name)
        for line in file:
            index, name = line.strip().split('\t')
            d[name+'rev'] = int(index) + rel_num
        file.close()

    return d


def read_triplets(file_name, entity_dict, relation_dict, add_reverse):
    data = []

    file = open(file_name)
    for line in file:
        head, relation, tail = line.strip().split('\t')

        head_idx = entity_dict[head]
        relation_idx = relation_dict[relation]
        tail_idx = entity_dict[tail]

        data.append((head_idx, relation_idx, tail_idx))
        if add_reverse:
            data.append((tail_idx, relation_dict[relation+'rev'], head_idx))
    file.close()

    return np.array(data)


def build_kg(train_data, add_reverse, directory):

    if os.path.exists(os.path.join(directory, "entity2relation.pkl")):
        print('loading the cached data ...')
        entity2relation = pickle.load(open(os.path.join(directory, "entity2relation.pkl"), 'rb'))
        entity2edges = np.load(os.path.join(directory, "entity2edges.npy"))
        edge2entities = np.load(os.path.join(directory, "edge2entities.npy"))
        edge2relation = np.load(os.path.join(directory, "edge2relation.npy"))
        train_triples_new = np.load(os.path.join(directory, "train_triples_new.npy"))
    else:
        entity2edge_set = defaultdict(set)
        entity2relation_set = defaultdict(set)
        entity2relation = defaultdict(set)
        entity2edges = []  # each row in entity2edges is the sampled edges connecting to this entity
        edge2entities = []  # each row in edge2entities is the two entities connected by this edge
        edge2relation = []  # each row in edge2relation is the relation type of this edge

        edge_idx_total = np.arange(train_data.shape[0])
        np.random.shuffle(edge_idx_total)

        train_triples_new = []
        edge_idx_new = 0
        for idx, edge_idx in enumerate(edge_idx_total):
            if idx % 10000000 == 0:
                print("%d/%d=%f" %
                      (idx, train_data.shape[0], float(idx)/float(train_data.shape[0])))
            head_idx, relation_idx, tail_idx = train_data[edge_idx]
            # single dirction
            # direction: relation --> entity
            if (len(entity2edge_set[tail_idx]) > args.neighbor_samples) and (relation_idx in entity2relation_set[tail_idx]):
                continue
            else:
                entity2relation_set[tail_idx].add(relation_idx)
                # the index for triple in train_triples_new
                entity2edge_set[tail_idx].add(edge_idx_new)
                edge2entities.append([head_idx])
                edge2relation.append(relation_idx)

                train_triples_new.append(train_data[edge_idx])
                edge_idx_new += 1

        null_entity = nentity
        null_relation = nrelation
        null_edge = len(edge2entities)
        edge2entities.append([null_entity])
        edge2relation.append(null_relation)

        train_triples_new = np.stack(train_triples_new)

        print('sampling neighbors ...')
        for i in range(nentity + 1):
            if i % 10000000 == 0:
                print("%d/%d=%f" % (i, nentity, float(i)/float(nentity)))

            if i not in entity2edge_set:
                entity2edge_set[i] = {null_edge}

            if len(entity2edge_set[i]) < args.neighbor_samples:
                sampled_neighbors = list(entity2edge_set[i]) + [null_edge] * \
                    (args.neighbor_samples - len(entity2edge_set[i]))
            else:
                rels_dict = {}
                edges = list(entity2edge_set[i])
                num_edges = len(edges)

                for edge in edges:
                    p = train_triples_new[edge][1]
                    if p not in rels_dict:
                        rels_dict[p] = 1.0
                    else:
                        rels_dict[p] += 1.0
                ps = []
                for edge in edges:
                    p = train_triples_new[edge][1]
                    ps.append(1.0 / (float(len(rels_dict)) * rels_dict[p]))
                sampled_neighbors = np.random.choice(edges, size=args.neighbor_samples,
                                                     replace=False, p=ps)
            entity2edges.append(sampled_neighbors)
            entity2relation[i] = list(entity2relation_set[i])

        del entity2edge_set
        del entity2relation_set

        print('saving the processed data ...')
        with open(os.path.join(directory, "entity2relation.pkl"), 'wb') as f:
            pickle.dump(entity2relation, f)
        entity2edges = np.array(entity2edges)
        edge2entities = np.array(edge2entities)
        edge2relation = np.array(edge2relation)
        np.save(os.path.join(directory, "entity2edges.npy"), entity2edges)
        np.save(os.path.join(directory, "edge2entities.npy"), edge2entities)
        np.save(os.path.join(directory, "edge2relation.npy"), edge2relation)
        np.save(os.path.join(directory, "train_triples_new.npy"), train_triples_new)
    return entity2relation, entity2edges, edge2entities, edge2relation, train_triples_new


def load_data(model_args):
    global args, entity_dict, relation_dict, nentity, nrelation
    args = model_args
    directory = '../data/' + args.dataset + '/'

    print('reading entity dict and relation dict ...')
    entity_dict = read_entities(directory + 'entities.dict')
    relation_dict = read_relations(directory + 'relations.dict', args.add_reverse)
    nentity = len(entity_dict)
    nrelation = len(relation_dict)

    print('reading train, validation, and test data ...')
    train_triplets = read_triplets(directory + 'train.txt', entity_dict,
                                   relation_dict, args.add_reverse)
    valid_triplets = read_triplets(directory + 'valid.txt', entity_dict,
                                   relation_dict, args.add_reverse)
    test_triplets = read_triplets(directory + 'test.txt', entity_dict,
                                  relation_dict, args.add_reverse)

    print('processing the knowledge graph ...')
    entity2relation, entity2edges, edge2entities, edge2relation, train_triplets = build_kg(
        train_triplets, args.add_reverse, directory)

    infer_triplets = np.array([np.arange(nentity), [0]*nentity, np.arange(nentity)]).T

    triplets = [train_triplets, valid_triplets, test_triplets, infer_triplets]

    neighbor_data = [entity2edges, edge2entities, edge2relation, entity2relation]
    return triplets, nrelation, neighbor_data


def load_data_wikikg(model_args, num_entity, num_relation, directory):
    global args, entity_dict, relation_dict, nentity, nrelation
    args = model_args
    nentity = num_entity
    nrelation = num_relation

    print('reading train, validation, and test data ...')
    train_triplets = np.load(directory + 'train_hrt.npy')
    # add_reverse
    train_triplets_reverse = np.stack(
        (train_triplets[:, 2], train_triplets[:, 1] + nrelation, train_triplets[:, 0]), axis=1)
    train_triplets = np.concatenate([train_triplets, train_triplets_reverse])
    nrelation = nrelation*2

    valid_triplets = np.load(directory + 'val_hr.npy')
    valid_triplets = np.stack((valid_triplets[:, 0], valid_triplets[:, 1], np.load(
        directory + 'val_t.npy')), axis=1)
    test_triplets = None

    print('processing the knowledge graph ...')
    entity2relation, entity2edges, edge2entities, edge2relation, train_triplets = build_kg(
        train_triplets, True, directory)
    print('processing the knowledge graph done')

    infer_triplets = np.array([np.arange(nentity), [0]*nentity, np.arange(nentity)]).T
    triplets = [train_triplets, valid_triplets, test_triplets, infer_triplets]

    neighbor_data = [entity2edges, edge2entities, edge2relation, entity2relation]

    return triplets, nrelation, neighbor_data
