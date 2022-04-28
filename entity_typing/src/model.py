import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from aggregators import MeanAggregator
from ranking_loss import RankingLoss


class PathCon(nn.Module):
    def __init__(self, args, n_relations):
        super(PathCon, self).__init__()
        self._parse_args(args, n_relations)
        self._build_model()

    def _parse_args(self, args, n_relations):
        self.n_relations = n_relations
        self.use_gpu = args.cuda

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.hidden_dim = args.dim
        self.feature_type = args.feature_type
        self.use_bce = args.use_bce
        self.use_ranking_loss = args.use_ranking_loss

        self.neighbor_samples = args.neighbor_samples
        self.context_hops = args.context_hops
        self.neighbor_agg = MeanAggregator
        self.agg_func = args.neighbor_agg

        if self.use_bce:
            self.loss = nn.BCEWithLogitsLoss(reduction='mean')
        elif self.use_ranking_loss:
            self.loss = RankingLoss(args.margin, args.gamma)
        else:
            self.loss = nn.CrossEntropyLoss()

    def _build_model(self):
        # define initial relation features
        self._build_relation_feature()
        # define aggregators for each layer
        self.aggregators = nn.ModuleList(self._get_neighbor_aggregators())
        self.output_layer = nn.Linear(self.hidden_dim*self.context_hops, self.n_relations)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, batch, mode='all'):
        self.labels = batch['labels']
        logits_tail = self._aggregate_neighbors(
            batch['tail_edges_lists'], batch['tail_masks'], batch["tail2relation"])
        logits_tail = self.output_layer(logits_tail)
        return logits_tail

    def _build_relation_feature(self):
        if self.feature_type == 'id':
            self.relation_dim = self.n_relations
            self.relation_features = torch.eye(self.n_relations).cuda(
            ) if self.use_gpu else torch.eye(self.n_relations)
        elif self.feature_type == 'bow':
            bow = np.load('../data/' + self.dataset + '/bow.npy')
            self.relation_dim = bow.shape[1]
            self.relation_features = torch.tensor(bow).cuda() if self.use_gpu else torch.tensor(bow)
        elif self.feature_type == 'bert':
            bert = np.load('../data/' + self.dataset + '/' + self.feature_type + '.npy')
            self.relation_dim = bert.shape[1]
            self.relation_features = torch.tensor(
                bert).cuda() if self.use_gpu else torch.tensor(bert)

        # the feature of the last relation (the null relation) is a zero vector
        self.relation_features = torch.cat([self.relation_features,
                                            torch.zeros([1, self.relation_dim]).cuda() if self.use_gpu
                                            else torch.zeros([1, self.relation_dim])], dim=0)

    def _get_neighbor_aggregators(self):
        aggregators = []  # store all aggregators
        if self.context_hops == 1:
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.n_relations))
        else:
            # the first layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.relation_dim,
                                                 output_dim=self.hidden_dim,
                                                 act=F.relu))
            # middle layers
            for i in range(self.context_hops - 2):
                aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                     input_dim=self.hidden_dim,
                                                     output_dim=self.hidden_dim,
                                                     act=F.relu))
            # the last layer
            aggregators.append(self.neighbor_agg(batch_size=self.batch_size,
                                                 input_dim=self.hidden_dim,
                                                 output_dim=self.hidden_dim))
        return aggregators

    def _aggregate_neighbors(self, edge_list, mask_list, edge2relation):
        batch_size = mask_list[0].shape[0]
        # translate edges IDs to relations IDs, then to features
        edge_vectors = [torch.index_select(
            self.relation_features, 0, edge_list[0]).unsqueeze(1)]
        idx = 0
        for edges in edge_list[1:]:
            relations = edge2relation[idx]
            edge_vectors.append(torch.index_select(self.relation_features, 0,
                                                   relations.view(-1)).view(list(relations.shape)+[-1]))
            idx += 1

        # shape of edge vectors:
        # [[batch_size, 1, edge_vectors],
        #  [batch_size, neighbor_samples, relation_dim],
        #  [batch_size, (neighbor_samples) * neighbor_samples , relation_dim],

        #  [batch_size, (neighbor_samples)^2 * neighbor_samples, relation_dim],
        #  ...]
        res = []
        for i in range(self.context_hops):
            aggregator = self.aggregators[i]
            edge_vectors_next_iter = []

            for hop in range(self.context_hops - i):
                neighbors_shape = [batch_size, -1, 1,
                                   self.neighbor_samples, aggregator.input_dim]
                masks_shape = [batch_size, -1, 1, self.neighbor_samples, 1]

                if (hop == 0 and i == 0) or (i == self.context_hops-1):
                    self_included = False
                else:
                    self_included = True
                vector = aggregator(self_vectors=edge_vectors[hop],
                                    neighbor_vectors=edge_vectors[hop +
                                                                  1].view(neighbors_shape),
                                    masks=mask_list[hop].view(masks_shape),
                                    agg_func=self.agg_func,
                                    self_included=self_included)

                edge_vectors_next_iter.append(vector)
            edge_vectors = edge_vectors_next_iter
            res.append(edge_vectors[0].squeeze(1))

        res = torch.cat(res, axis=1)
        return res


def train_step(model, optimizer, batch, uni_weight):
    model.train()
    optimizer.zero_grad()
    logits_tail = model(batch)

    if model.use_bce or model.use_ranking_loss:
        labels = batch['label_sets']
        loss = model.loss(logits_tail, labels)
    else:
        loss = model.loss(logits_tail, batch['labels'])

    loss.backward()
    optimizer.step()

    return loss.item()


def test_step_loss(model, batch):
    model.eval()
    with torch.no_grad():
        logits_tail = model(batch)
        if model.use_bce or model.use_ranking_loss:
            labels = batch['label_sets']
            loss = model.loss(logits_tail, labels)
        else:
            loss = model.loss(logits_tail, batch['labels'])
    return loss.item()


def test_step_score(model, batch):
    model.eval()
    with torch.no_grad():
        if model.use_bce and model.use_ranking_loss:
            logits_tail = F.sigmoid(model(batch))
        else:
            logits_tail = F.softmax(model(batch), dim=-1)
    return logits_tail.cpu().detach()
