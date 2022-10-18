import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gcn import _init_weights
from models.layers import build_mlp

class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, input_dim, output_dim, dropout, alpha, hidden_dim=None, concat=True,
                 pooling='avg', mlp_normalization='none'):
        super(GraphAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        if hidden_dim is None:
            hidden_dim = input_dim
        self.hidden_dim = hidden_dim

        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*output_dim, 1)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling
        self.pooling = pooling
        net1_layers = [input_dim + 2 * output_dim, hidden_dim, 2 * hidden_dim + output_dim]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, 2 * hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
        self.net2.apply(_init_weights)

    def forward(self, obj_vecs, pred_vecs, edges):
        dtype, device = obj_vecs.dtype, obj_vecs.device
        O, T = obj_vecs.size(0), pred_vecs.size(0)
        H, Din, Dout = self.hidden_dim, self.input_dim, self.output_dim

        # Break apart indices for subjects and objects; these have shape (T,)
        s_idx = edges[:, 0].contiguous().long()
        o_idx = edges[:, 1].contiguous().long()

        # add attention mechanism
        Wh = torch.mm(obj_vecs, self.W)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e, device=device)  # will be 0 after softmax opration
        b = edges.size(0)
        obj_counts = int((1 + (1 + 4 * b) ** 0.5) / 2)
        adj = torch.zeros(obj_counts, obj_counts, device=device)
        for i, j in edges:
            adj[i][j] = 1
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = F.relu(torch.matmul(attention, Wh))

        # Get current vectors for subjects and objects; these have shape (T, Din)
        cur_s_vecs = h_prime[s_idx]
        cur_o_vecs = h_prime[o_idx]

        # Get current vectors for triples; shape is (T, 3 * Din)
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs
        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H + Dout)]
        new_o_vecs = new_t_vecs[:, (H + Dout):(2 * H + Dout)]

        # Allocate space for pooled object vectors of shape (O, H)
        pooled_obj_vecs = torch.zeros(O, H, dtype=dtype, device=device)
        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
            # Figure out how many times each object has appeared
            obj_counts = torch.zeros(O, dtype=dtype, device=device)
            ones = torch.ones(T, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they appeared
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        # Send pooled object vectors through net2 to get output object vectors,
        new_obj_vecs = self.net2(pooled_obj_vecs)
        return new_obj_vecs, new_p_vecs

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)

        return all_combinations_matrix.view(N, N, 2*self.output_dim)

class GAT(nn.Module):
    def __init__(self, nfeat, nhiden=None, dropout=0.2, alpha=0.2, nheads=4):
        """Dense version of GAT"""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.input_dim = nfeat
        self.output_dim = nfeat
        if nhiden is None:
            nhiden = nfeat
        self.hidden_dim = nhiden

        net1_layers = [3 * nfeat, nhiden, 2 * nhiden + nfeat]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm='batch')
        self.net1.apply(_init_weights)

        self.attentions = [GraphAttentionLayer(nfeat, nfeat, dropout, alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nfeat*nheads, nfeat, dropout, alpha, concat=False)

    # avoid vanishing gradient
    def forward(self, obj_vecs, pred_vecs, edges):
        obj_vecs = F.dropout(obj_vecs, self.dropout, training=self.training)
        pred_vecs = F.dropout(pred_vecs, self.dropout, training=self.training)
        obj_cat_vecs = torch.tensor([0])
        pred_cat_vecs = torch.tensor([0])
        for att in self.attentions:
            obj_vecs, pred_vecs = att(obj_vecs, pred_vecs, edges)
            if obj_cat_vecs.shape[0] == 1:
                obj_cat_vecs = obj_vecs
                pred_cat_vecs = pred_vecs
            else:
                obj_cat_vecs = torch.cat([obj_cat_vecs, obj_vecs], dim=1)
                pred_cat_vecs = torch.cat([pred_cat_vecs, pred_vecs], dim=1)
        obj_cat_vecs = F.dropout(obj_cat_vecs, self.dropout, training=self.training)
        pred_cat_vecs = F.dropout(pred_cat_vecs,self.dropout, training=self.training)

        obj_cat_vecs, pred_cat_vecs = self.out_att(obj_cat_vecs, pred_cat_vecs, edges)
        obj_cat_vecs = F.relu(obj_cat_vecs)
        pred_cat_vecs = F.relu(pred_cat_vecs)
        # for att in self.attentions:
        #     obj_vecs = obj_vecs + att(obj_vecs, edges)
        # obj_vecs /= len(self.attentions)
        return F.log_softmax(obj_cat_vecs, dim=1), F.log_softmax(pred_cat_vecs, dim=1)