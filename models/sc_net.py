import torch
import math
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU,Parameter

from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops,to_dense_batch,to_dense_adj,mask_select
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool,DenseSAGEConv, dense_diff_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.pool.topk_pool import TopKPooling

num_atom_type = 119

def gcn_norm(edge_index, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class GNNConv(MessagePassing):
    def __init__(self,in_channels,out_channels):
        super().__init__(aggr='add')
        self.edge_weight1 = Parameter(torch.Tensor(2,300))

        # self.reset_parameters()

        nn.init.xavier_uniform_(self.edge_weight1)

        self.mlp = Seq(
            Linear(in_channels,out_channels),
            ReLU()
        )
        self.mlp.apply(self.reset_parameters)

    def reset_parameters(self,m):
        if type(m) == Linear:
            nn.init.xavier_uniform_(m.weight)

    def forward(self,x,edge_index,edge_attr):
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,1] = 4 # bond-order
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_fea = edge_attr @ self.edge_weight1

        edge_index,_ = gcn_norm(edge_index)

        out = self.propagate(edge_index, x=x, edge_attr=edge_fea, size=None)

        return out

    def message(self, x_j: Tensor,edge_attr) -> Tensor:

        return self.mlp( x_j + edge_attr )

class GNN(nn.Module):
    def __init__(self,padden_len,num_layer=5, emb_dim=300,drop_ratio=0,pre_train=False,pool='mean',metal_offset=False) -> None:
        super(GNN,self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.x_embedding = nn.Embedding(num_atom_type,emb_dim)
        self.x_weight = Parameter(torch.Tensor(padden_len,emb_dim))
        self.pre_train = pre_train
        self.metal_offset = metal_offset
        if metal_offset:
            self.offset = Seq(
                nn.Linear(2*emb_dim,2*emb_dim),
                nn.ReLU(),
                nn.Linear(2*emb_dim,emb_dim)
            )
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')

        nn.init.xavier_uniform_(self.x_embedding.weight.data)
        nn.init.xavier_uniform_(self.x_weight.data)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GNNConv(emb_dim,emb_dim))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
    def forward(self,data,metal_feature):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        h = self.x_embedding(x[:,0].int()) + x[:,1:] @ self.x_weight

        for layer in range(self.num_layer):
            if layer == math.floor(self.num_layer/2)+1:
                temp,mask = to_dense_batch(h,data.batch)
                metals = metal_feature.repeat(1,temp.size(1),1)
                comb_fea = torch.cat((temp,metals),dim=-1)
                comb_fea = temp + metals
                comb_fea = comb_fea.reshape(-1,comb_fea.size(-1))
                mask = mask.reshape(-1)
                comb_fea = mask_select(comb_fea,0,mask)
            h = self.gnns[layer](h,edge_index,edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        if self.pre_train:
            h = self.pool(h, data.batch)
            h = self.feat_lin(h)
            out = self.out_lin(h)

            return out
        else:
            return h
        
        
class CoordiNet(nn.Module):
    def __init__(self,d_feature,d_k,pool,apool,name) -> None:
        super().__init__()
        self.W_q = nn.Linear(d_feature,d_k,bias=False)
        self.W_k = nn.Linear(d_feature,d_k,bias=False)
        self.W_v = nn.Linear(d_feature,d_k,bias=False)
        self.dk = d_k

        self.apool = apool
        if apool['name'] == 'diff_pool':
            self.SAGE = DenseSAGEConv(d_feature,self.apool['num'])
            self.pool_bn = nn.BatchNorm1d(self.apool['num'])
        elif apool['name'] == 'topk':
            self.topk = TopKPooling(d_feature,apool['ratio'])

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            self.pool = pool
        
        self.metal_layer = Seq(
            nn.Linear(d_feature,d_k),
            nn.ReLU()
        )

        self.ligand_layer = Seq(
            nn.Linear(d_feature,d_k),
            nn.ReLU()
        )

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self,h,metal,batch,edge_index):    
        if self.apool['name'] == 'diff_pool':
            dense_h,mask = to_dense_batch(h,batch)
            adj = to_dense_adj(edge_index,batch)
            s = self.SAGE(dense_h,adj,mask).permute(0,2,1)
            s = self.pool_bn(s).relu().permute(0,2,1)
            x, _, _, _ = dense_diff_pool(dense_h, adj, s, mask)
        elif self.apool['name'] == 'topk':
            x,_,_,bc,_,_ = self.topk(h,edge_index,batch=batch)
            x,mask = to_dense_batch(x,bc)
        elif self.apool['name'] == 'none':
            x,mask = to_dense_batch(h,batch)
        
        Q = self.W_q(metal)
        K = self.W_k(x)
        V = self.W_v(x)
        if self.apool['name'] == 'topk':
            mask = mask.int().unsqueeze(-2)
            attention = torch.bmm(torch.nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))/math.sqrt(self.dk)-(1-mask)*1e10),V)
        elif self.apool['name'] == 'diff_pool':
            attention = torch.bmm(torch.nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))/math.sqrt(self.dk)),V)
        elif self.apool['name'] == 'none':
            attention = torch.bmm(torch.nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))/math.sqrt(self.dk)),V)

        metal = self.metal_layer(metal)
        if self.pool != False:
            graph_fea = self.pool(h,batch)
            if len(graph_fea.size()) < 3:
                graph_fea = graph_fea.unsqueeze(1)
            graph_fea = self.ligand_layer(graph_fea)
            feature = attention.relu() + metal + graph_fea
        else:
            feature = attention.relu() + metal

        return feature 
    
class NonAttention(nn.Module):
    def __init__(self,d_feature,d_k,pool,apool,name) -> None:
        super().__init__()
        self.ligand_layer = Seq(
            nn.Linear(d_feature,d_k),
            nn.ReLU()
        )
        self.metal_layer = Seq(
            nn.Linear(d_feature,d_k),
            nn.ReLU()
        )

        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')

    def forward(self,h,metal,batch,edge_index):

        h = self.pool(h,batch)
        li_featrue = self.ligand_layer(h)
        metal_feature = self.metal_layer(metal)
        if len(metal_feature.size()) >2:
            metal_feature = metal_feature.squeeze(1)

        feature = li_featrue + metal_feature

        return feature


                  
class ControlNet(nn.Module):
    # Equivalent parameter quantity
    def __init__(self,d_feature,d_k,pool='mean') -> None:
        super().__init__()
        self.mlp = Seq(
            Linear(d_feature,d_feature),
            ReLU(),
            Linear(d_feature,d_feature),
            ReLU(),
            Linear(d_feature,d_k),
        )
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'add':
            self.pool = global_add_pool
        elif pool == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError('Not defined pooling!')
        
        self.mlp.apply(self.reset_parameters)

    def reset_parameters(self,m):
        if type(m) == Linear:
            nn.init.xavier_uniform_(m.weight)

    def forward(self,h,batch):
        h = self.pool(h,batch)
        h = self.mlp(h)

        return h

class SCnet(nn.Module):
    def __init__(self,padden_len,GNN_config,Coor_config,out_dimention,separated=False) -> None:
        super(SCnet,self).__init__()
        # self.patch_sampler = 
        self.GNN = GNN(padden_len,**GNN_config)
        if separated:
            if Coor_config['name'] == 'attention':
                self.CoordiNet = CoordiNet(**Coor_config)
            elif Coor_config['name'] == 'NonAttention':
                self.CoordiNet = NonAttention(**Coor_config)
        else:
            self.ControlNet = ControlNet(**Coor_config)
        self.separated = separated
        self.Coor_config = Coor_config
        self.pre_head = Seq(
            Linear(self.Coor_config['d_k'],2*self.Coor_config['d_k']),
            nn.Softplus(),
            Linear(2*self.Coor_config['d_k'],out_dimention)
        )
        self.metal_embedding1 = nn.Linear(17,GNN_config['emb_dim'])
        self.metal_embedding2 = nn.Embedding(119,GNN_config['emb_dim'])

        nn.init.xavier_uniform_(self.metal_embedding1.weight)
        nn.init.xavier_uniform_(self.metal_embedding2.weight.data)

    def forward(self,data):
        if self.separated:
            metals = data.metal.reshape(-1,18).unsqueeze(dim=-2)
            metal_feature = self.metal_embedding1(metals[:,:,0:17].float()) + self.metal_embedding2(metals[:,:,17])
            h = self.GNN(data,metal_feature)
            h = self.CoordiNet(h,metal_feature,data.batch,data.edge_index)
            if len(h.size()) > 2:
                h = h.squeeze(1)
            out = self.pre_head(h)
        else:
            h = self.GNN(data)
            h = self.ControlNet(h,data.batch)
            out = self.pre_head(h)

        return out