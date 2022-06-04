import torch.nn as nn
import torch
import torch.nn.functional as f
from layers import MixHopConv


class MixHop(nn.Module):
    def __init__(self,
                 g,
                 d_sim_dim,
                 m_sim_dim,
                 disease_number,
                 mirna_number,
                 hid_dim,
                 out_dim,
                 num_layers,
                 p,
                 input_dropout=0.0,
                 layer_dropout=0.0,
                 activation=None,
                 batchnorm=False):
        super(MixHop, self).__init__()
        self.g = g
        self.d_sim_dim = d_sim_dim
        self.m_sim_dim = m_sim_dim
        self.disease_number = disease_number
        self.mirna_number = mirna_number
        self.num_layers = num_layers
        self.p = p
        self.input_dropout = input_dropout
        self.layer_dropout = layer_dropout
        self.activation = activation
        self.batchnorm = batchnorm

        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(self.input_dropout)
        self.m_fc = nn.Linear(m_sim_dim, hid_dim)
        self.d_fc = nn.Linear(d_sim_dim, hid_dim)
        self.m_fc1 = nn.Linear(out_dim + m_sim_dim, out_dim)
        self.d_fc1 = nn.Linear(out_dim+ d_sim_dim, out_dim)
        self.predict = nn.Sequential(nn.Linear(out_dim * 2, out_dim),
                                     nn.ReLU(),
                                     nn.Linear(out_dim, 1),
                                     nn.Sigmoid())
        # Input layer
        self.layers.append(MixHopConv(hid_dim,
                                      out_dim,
                                      p=self.p,
                                      dropout=self.input_dropout,
                                      activation=self.activation,
                                      batchnorm=self.batchnorm))

        # Hidden layers with n - 1 MixHopConv layers
        for i in range(self.num_layers - 2):
            self.layers.append(MixHopConv(hid_dim * len(p),
                                          hid_dim,
                                          p=self.p,
                                          dropout=self.layer_dropout,
                                          activation=self.activation,
                                          batchnorm=self.batchnorm))

        self.fc_layers = nn.Linear(hid_dim * len(p), out_dim, bias=False)

    def forward(self, graph, src, dst):
        self.g.apply_nodes(lambda nodes: {'z': self.dropout(self.d_fc(nodes.data['d_sim']))}, 383)
        self.g.apply_nodes(lambda nodes: {'z': self.dropout(self.m_fc(nodes.data['m_sim']))}, 495)

        feats = self.g.ndata.pop('z')
        for layer in self.layers:
            feats = layer(graph, feats)

        feats = self.fc_layers(feats)

        # h_d = feats[:383]
        # h_m = feats[383:]

        h_d = torch.cat((feats[:self.disease_number], self.g.ndata['d_sim'][:self.disease_number]), dim=1)
        h_m = torch.cat((feats[self.disease_number:], self.g.ndata['m_sim'][self.disease_number:]), dim=1)

        # h_m = self.dropout(F.elu(h_m))
        # h_d = self.dropout(F.elu(h_d))
        h_m = self.dropout(f.elu(self.m_fc1(h_m)))
        h_d = self.dropout(f.elu(self.d_fc1(h_d)))

        h = torch.cat((h_d, h_m), dim=0)

        h_diseases = h[src]
        h_mirnas = h[dst]

        h_concat = torch.cat((h_diseases, h_mirnas), 1)
        predict_score = self.predict(h_concat)

        return predict_score

