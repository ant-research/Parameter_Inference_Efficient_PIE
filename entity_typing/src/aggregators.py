import torch
import torch.nn as nn
from abc import abstractmethod


class Aggregator(nn.Module):
    def __init__(self, batch_size, input_dim, output_dim, act):
        super(Aggregator, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.fd_layer = nn.Linear(self.input_dim*5, self.input_dim)
        nn.init.xavier_uniform_(self.fd_layer.weight)

    def forward(self, self_vectors, neighbor_vectors, masks, agg_func='mean', self_included=True):
        # self_vectors: [batch_size, -1, input_dim]
        # neighbor_vectors: [batch_size, -1, 2, n_neighbor, input_dim]
        # masks: [batch_size, -1, 2, n_neighbor, 1]
        if agg_func == 'mean':
            entity_vectors = torch.mean(neighbor_vectors * masks, dim=-
                                        2)  # [batch_size, -1, 1, input_dim]
        elif agg_func == 'sum':
            entity_vectors = torch.sum(neighbor_vectors * masks, dim=-
                                       2)  # [batch_size, -1, 1, input_dim]
        elif agg_func == 'pna':
            mean_v = torch.mean(neighbor_vectors * masks, dim=-2)  # [batch_size, -1, 1, input_dim]
            max_v = torch.max(neighbor_vectors * masks, dim=-2)[0]  # [batch_size, -1, 1, input_dim]
            sum_v = torch.sum(neighbor_vectors * masks, dim=-2)  # [batch_size, -1, 1, input_dim]
            min_v = torch.min(neighbor_vectors * masks, dim=-2)[0]  # [batch_size, -1, 1, input_dim]
            seq_mean = torch.mean(neighbor_vectors ** 2 * masks, dim=-
                                  2)  # [batch_size, -1, 1, input_dim]
            std_v = (seq_mean - mean_v**2).clamp(min=1e-6).sqrt()
            features = torch.cat([mean_v, max_v, min_v, sum_v, std_v], dim=-
                                 1)  # [batch_size, -1, 1, input_dim*4]
            shape = features.shape

            entity_vectors = self.fd_layer(features.squeeze()).view(
                [shape[0], shape[1], shape[2], self.input_dim])
        else:
            raise ValueError("Unknown aggregation function `%s`" % agg_func)

        outputs = self._call(self_vectors, entity_vectors, self_included)
        return outputs

    @ abstractmethod
    def _call(self, self_vectors, entity_vectors, self_included):
        pass


class MeanAggregator(Aggregator):
    def __init__(self, batch_size, input_dim, output_dim, act=lambda x: x):
        super(MeanAggregator, self).__init__(batch_size, input_dim, output_dim, act)

        self.layer = nn.Linear(self.input_dim, self.output_dim)
        nn.init.xavier_uniform_(self.layer.weight)

    def _call(self, self_vectors, entity_vectors, self_included):
        # self_vectors: [batch_size, -1, input_dim]
        # entity_vectors: [batch_size, -1, 2, input_dim] or [batch_size, -1, 1, input_dim]
        batch_size = self_vectors.shape[0]
        output = torch.mean(entity_vectors, dim=-2)  # [batch_size, -1, input_dim]
        if self_included:
            output += self_vectors.view([output.shape[0], -1, output.shape[2]])
        output = output.view([-1, self.input_dim])  # [-1, input_dim]
        output = self.layer(output)  # [-1, output_dim]
        output = output.view([batch_size, -1, self.output_dim])  # [batch_size, -1, output_dim]

        return self.act(output)
