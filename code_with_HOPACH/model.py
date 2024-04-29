import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Features, gather_nodes, cat_neighbors_nodes

class information_mix(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(information_mix, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h


class EncLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.1, scale=16):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden*3, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden*3, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.information_mix = information_mix(num_hidden, num_hidden * 2)

    def forward(self, h_V, h_E, E_idx):

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.information_mix(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E


class DecLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.1, scale=16):
        super(DecLayer, self).__init__()
        self.num_hidden = num_hidden
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden*3, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.information_mix = information_mix(num_hidden, num_hidden * 2)

    def forward(self, h_V, h_EV):
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))
        dh = self.information_mix(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        return h_V 


class scMPNN(nn.Module):
    def __init__(self, num_states, num_genes, gene_vecdim=200, node_feature_dim=128, edge_feature_dim=64,
        hidden_dim=128, num_encoder_layers=2, num_decoder_layers=1,
        k_neighbors_pos=8, k_neighbors_neg=8, augment_eps=0.05, dropout=0.1):
        super(scMPNN, self).__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim

        self.features = Features(num_genes, edge_feature_dim=edge_feature_dim, node_feature_dim=node_feature_dim,
                                 top_k_pos=k_neighbors_pos, top_k_neg=k_neighbors_neg, augment_eps=augment_eps)

        self.W_e = nn.Linear(edge_feature_dim, hidden_dim, bias=True)
        self.W_v = nn.Linear(node_feature_dim, hidden_dim, bias=True)

        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, dropout=dropout, scale=k_neighbors_pos+k_neighbors_neg)
            for _ in range(num_encoder_layers)
        ])

        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, dropout=dropout, scale=k_neighbors_pos+k_neighbors_neg)
            for _ in range(num_decoder_layers)
        ])
        self.W_recover = nn.Linear(hidden_dim, gene_vecdim, bias=True)
        
        self.W_cell1 = nn.Linear(hidden_dim*3, 1, bias=True)
        self.W_cell2 = nn.Linear(k_neighbors_pos+k_neighbors_neg, 1, bias=True)
        self.W_cell3 = nn.Linear(num_genes, num_states-1, bias=True)
        self.act = torch.nn.GELU()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, genes, X, mask):
        device=X.device
        V, E, E_idx = self.features(genes, X, mask)
        h_V = self.W_v(V) ###128
        h_E = self.W_e(E) ###128
        
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx)

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_EV = self.act(self.W_cell1(h_EV))
        
        h_EV=h_EV.view(h_EV.shape[0],h_EV.shape[1],h_EV.shape[2])
        h_EV = self.act(self.W_cell2(h_EV))
        
        h_EV=h_EV.view(h_EV.shape[0],h_EV.shape[1])
        cell_state_logits=self.W_cell3(h_EV)
        
        for layer in self.decoder_layers:
            h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
            h_V = layer(h_V, h_EV)

        recover = self.W_recover(h_V) 
        ## h-V:[batch,ngenes,200] 
        
        return recover, cell_state_logits