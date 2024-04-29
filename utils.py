import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np


class Data_using(data.Dataset):
    def __init__(self, gene_vecs, input_data, label_data, label_dict, training = False,
                 Domain_data=None, Domain_dict=None):
        self.gene_vecs=gene_vecs
        self.input_data = input_data
        self.label_data = label_data
        self.label_dict = label_dict
        self.training = training
        self.Domain_data = Domain_data
        self.Domain_dict = Domain_dict
        self.n_data = input_data.shape[0]

    def __getitem__(self, index):
        
        cur_cell=np.array(self.input_data[index])
        mask=np.ones(cur_cell.shape[-1])
        if self.training:
            num_0 = int(0.15 * cur_cell.shape[-1])
            random_indices = np.random.choice(cur_cell.shape[-1] , num_0, replace=False)
            mask[random_indices] = 0
        cur_gene=np.array(self.gene_vecs)
        classlabel=int(self.label_dict[self.label_data[index]])
           
        return cur_gene, cur_cell, mask, classlabel
        
    def __len__(self):
        
        return self.n_data



class Features(nn.Module):
    def __init__(self, num_genes, edge_feature_dim=128, node_feature_dim=128,
        num_rbf=8, top_k_pos=8, top_k_neg=8, augment_eps=0.01):
        
        super(Features, self).__init__()
        self.edge_feature_dim = edge_feature_dim
        self.node_feature_dim = node_feature_dim
        self.top_k_pos = top_k_pos
        self.top_k_neg = top_k_neg
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf

        node_in_dim = 200 ## gene2vec would generate vector of dim 200
        edge_in_dim = num_rbf
        
        self.node_embedding = nn.Linear(node_in_dim, node_feature_dim, bias=False)
        self.norm_node = nn.LayerNorm(node_feature_dim)
        self.edge_embedding = nn.Linear(edge_in_dim, edge_feature_dim, bias=False)
        self.norm_edge = nn.LayerNorm(edge_feature_dim)
        self.mix_info1 = nn.Linear(num_genes, num_genes)
        self.mix_info2 = nn.Linear(num_genes, num_genes)
        self.norm_raw = nn.LayerNorm(num_genes)
        self.act = torch.nn.GELU()

    def _get_logratio(self, X):
        X[X==0] += (0.5+self.augment_eps * torch.randn_like(X))[X==0]
        ##log-ratio matrix
        logratio_matrix = torch.log (X.unsqueeze(-2) / X.unsqueeze(-1))
        logratio_matrix = self.norm_raw(self.mix_info2(self.act(self.mix_info1(logratio_matrix))))
        LogR_neigh_pos, E_idx_pos = torch.topk(logratio_matrix, np.minimum(self.top_k_pos, X.shape[-1]), dim=-1, largest=True)
        LogR_neigh_neg, E_idx_neg = torch.topk(logratio_matrix, np.minimum(self.top_k_neg, X.shape[-1]), dim=-1, largest=False)
        LogR_neighbors = torch.cat((LogR_neigh_pos, LogR_neigh_neg), dim=-1)
        E_idx = torch.cat((E_idx_pos, E_idx_neg), dim=-1)
        return LogR_neighbors, E_idx

    def _rbf(self, LogR_neighbors):
        device = LogR_neighbors.device
        Ratio_mu = torch.tensor([-3.2,-1.6,-0.8,-0.2,0.2,0.8,1.6,3.2], device=device)
        Ratio_sigma = torch.tensor([2.8,1.4,1.0,0.4,0.4,1.0,1.4,2.8], device=device)
        Ratio_mu = Ratio_mu.view([1,1,1,-1]) ##[1,1,1,8]
        Ratio_sigma = Ratio_sigma.view([1,1,1,-1])
        Ratio_expand = torch.unsqueeze(LogR_neighbors, -1) ##[b,ngenes,2k,1]
        result = torch.exp(-((Ratio_expand - Ratio_mu) / Ratio_sigma)**2)
        return result

    def forward(self, genes, X, mask):
        LogR_neighbors, E_idx = self._get_logratio(X)
        Ratio_values = self._rbf(LogR_neighbors) 
        E = self.edge_embedding(Ratio_values)
        E = self.norm_edge(E)
        V = genes * mask.unsqueeze(-1)
        V = self.node_embedding(V)
        V = self.norm_node(V)
        ## [b,ngenes,128]
        
        return V, E, E_idx 

def gather_nodes(nodes, neighbor_idx):
    # Features [batch,ngenes,node_dim] at Neighbor indices [batch,ngenes,2k] => [batch,ngenes,2k,node_dim]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_edges, E_idx):
    h_neighbor_nodes = gather_nodes(h_nodes, E_idx)
    #### [batch,ngenes,2k,node_dim]
    h_nn = torch.cat([h_edges, h_neighbor_nodes], -1)
    #### [batch,ngenes,2k,node_dim+edge_dim]
    return h_nn

def get_prediction(logits):
    
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels