import torch
import math
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear,Parameter

from torch_scatter import scatter_add

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch,to_dense_adj,add_self_loops,degree
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool,DenseSAGEConv, dense_diff_pool
from torch_geometric.utils.num_nodes import maybe_num_nodes

from torch_geometric.nn.models import GCN
from torch_geometric.nn.models import GIN
from torch_geometric.nn.models import GAT
from torch_geometric.nn.models import PNA
from torch_geometric.nn.models import GraphSAGE

import sys
sys.path.append('./models')
from SchNet import tmqm_SchNet
from AttentiveFP import tmqm_AttentiveFP

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
    def __init__(self,in_channels,out_channels,edge_type):
        super().__init__(aggr='add')

        self.mlp = Seq(
            Linear(2*in_channels,out_channels),
            nn.ReLU(),
        )

        # self.lin = Linear(2*in_channels,in_channels)

        self.edge_weight1 = Parameter(torch.Tensor(3,out_channels))
        nn.init.xavier_uniform_(self.edge_weight1)
        self.edge_weight2 = Parameter(torch.Tensor(1,out_channels))
        nn.init.xavier_uniform_(self.edge_weight2)

        self.mlp.apply(self.reset_parameters)
        self.edge_type = edge_type

    def reset_parameters(self,m):
        if type(m) == Linear:
            nn.init.xavier_uniform_(m.weight)

    def forward(self,x,edge_index,edge_attr):
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # self_loop_attr = torch.zeros(x.size(0), 4)
        # self_loop_attr[:,3] = 4 # bond-order
        # self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        # edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        # edge_index,_ = gcn_norm(edge_index)

        if edge_attr.shape[1] != 1:
            if self.edge_type == '2d':
                edge_fea = edge_attr @ self.edge_weight2
            elif self.edge_type == '3d':
                edge_fea = edge_attr @ self.edge_weight1
            elif self.edge_type == '2-3d':
                edge_fea = edge_attr[:,0:3] @ self.edge_weight1 + edge_attr[:,3:] @ self.edge_weight2
            elif self.edge_type == 'off':
                edge_fea = -1
        else:
            edge_fea = edge_attr @ self.edge_weight2

        edge_index,_ = gcn_norm(edge_index)
        row, col = edge_index[0],edge_index[1]
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        out = self.propagate(edge_index, x=x, edge_attr=edge_fea,norm=norm)
        out = self.mlp(torch.cat([x,out],dim=1))

        return out

    def message(self, x_j: Tensor,edge_attr,norm) -> Tensor:
        if self.edge_type == 'off':
            return norm.view(-1,1)*(x_j)
        else:
            return norm.view(-1,1)*(x_j + edge_attr)
        # return norm.view(-1,1)*(x_j)

class diff_pool(nn.Module):
    def __init__(self, d_feature=300,class_num=16) -> None:
        super().__init__()

        self.SAGE = DenseSAGEConv(d_feature,class_num)
        self.pool_ln = nn.BatchNorm1d(class_num)

    def forward(self,h,batch,edge_index):
        dense_h,mask = to_dense_batch(h,batch)
        adj = to_dense_adj(edge_index,batch)
        s = self.SAGE(dense_h,adj,mask).permute(0,2,1)
        s = self.pool_ln(s).relu().permute(0,2,1)
        x, _, _, _ = dense_diff_pool(dense_h, adj, s, mask)

        s = s.unsqueeze(0) if s.dim() == 2 else s
        s = torch.softmax(s, dim=-1)

        return x,adj,s,mask
    
class metal_attention(nn.Module):
    def __init__(self,d_feature=300,d_k=256,kernel='exp',interprate=False) -> None:
        super().__init__()

        self.W_q = nn.Linear(d_feature,d_k,bias=False)
        self.W_k = nn.Linear(d_feature,d_k,bias=False)
        self.W_v = nn.Linear(d_feature,d_k,bias=False)
        self.lynorm = nn.LayerNorm(d_k)
        self.dk = d_k
        self.d_feature = d_feature
        self.kernel = kernel
        self.interprate = interprate

        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)

    def forward(self,metal_feature,x):
        Q = self.W_q(metal_feature)
        K = self.W_k(x)
        V = self.W_v(x)
        if self.kernel == 'exp':
            attention = torch.bmm(torch.nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))/math.sqrt(self.dk)),V)
        elif self.kernel == 'RBF':
            attention = torch.bmm(torch.nn.Softmax(dim=-1)(-torch.norm(torch.sub(Q,K),p=2,dim=-1).unsqueeze(1)/math.sqrt(self.dk)),V)

        if self.interprate:
            weight = torch.nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))/math.sqrt(self.dk))
            return attention.relu().squeeze(1),weight
        else:
            return attention.relu().squeeze(1)

class MPNN(nn.Module):
    def __init__(self,num_layer=5,emb_dim=300,edge_type='2-3d',drop_ratio=0) -> None:
        super(MPNN,self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio

        # nn.init.xavier_uniform_(self.x_embedding.weight.data)
        # nn.init.xavier_uniform_(self.x_weight.data)

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GNNConv(emb_dim,emb_dim,edge_type))

        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
    def forward(self,x,edge_index,edge_attr):
        h=x
        for layer in range(self.num_layer):
            h = self.gnns[layer](h,edge_index,edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        return h
               
        
class CoordiNet(nn.Module):
    def __init__(self,d_feature=300,d_k=256,apool=16,drop_ratio=0,kernel='exp',interprate=False) -> None:
        super().__init__()
        if apool != -1:
            self.apool_layer = diff_pool(d_feature,apool)
        else:
            pass
        self.attention = metal_attention(d_feature,d_k,kernel,interprate)
        self.kernel = kernel
        self.drop_ratio = drop_ratio
        self.apool = apool
        self.interprate = interprate

    def forward(self,x,metal_feature,batch,edge_index):
        metal_feature = metal_feature.unsqueeze(dim=-2)
        if self.apool != -1:
            h,_,s,_ = self.apool_layer(x,batch,edge_index)
        else:
            h = x

        if self.interprate:
            h,weight = self.attention(metal_feature,h)
            return h,weight,s
        else:
            h = self.attention(metal_feature,h)
            return h

class SCnet(nn.Module):
    def __init__(self,GNN_config,Coor_config,out_dimention,separated=False,attention=False,gnn='MPNN',deg=None) -> None:
        super(SCnet,self).__init__()
        if gnn == 'MPNN':
            self.GNN = MPNN(num_layer=GNN_config['num_layer'],
                            emb_dim=GNN_config['emb_dim'],
                            edge_type=GNN_config['edge_type'],
                            drop_ratio=GNN_config['drop_ratio'])
        elif gnn == 'GCNs':
            self.GNN = GCN(
                in_channels=GNN_config['emb_dim'],
                hidden_channels=GNN_config['emb_dim'],
                edge_dim=1,
                num_layers=GNN_config['num_layer'],
                dropout=GNN_config['drop_ratio'],
                act='relu',
                norm="BatchNorm"
            )
        elif gnn == 'GIN':
            self.GNN = GIN(in_channels=GNN_config['emb_dim'],
                           hidden_channels=GNN_config['emb_dim'],
                           edge_dim=1,
                           num_layers=GNN_config['num_layer'],
                           dropout=GNN_config['drop_ratio'],
                           act='relu',
                           norm="BatchNorm")
        elif gnn == 'GAT':
            self.GNN = GAT(in_channels=GNN_config['emb_dim'],
                           hidden_channels=GNN_config['emb_dim'],
                           edge_dim=1,
                           num_layers=GNN_config['num_layer'],
                           dropout=GNN_config['drop_ratio'],
                           act='relu',
                           norm="BatchNorm")
        elif gnn == 'GAT-v2':
            self.GNN = GAT(in_channels=GNN_config['emb_dim'],
                           hidden_channels=GNN_config['emb_dim'],
                           num_layers=GNN_config['num_layer'],
                           edge_dim=1,
                           dropout=GNN_config['drop_ratio'],
                           act='relu',
                           norm="BatchNorm",
                           v2=True)
        elif gnn == 'PNA':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            self.GNN = PNA(in_channels=GNN_config['emb_dim'],
                           hidden_channels=GNN_config['emb_dim'],
                           num_layers=GNN_config['num_layer'],
                           edge_dim=1,
                           dropout=GNN_config['drop_ratio'],
                           act='relu',
                           norm="BatchNorm",
                           aggregators=aggregators,
                           scalers=scalers,
                           deg=deg)
        elif gnn == 'GraphSAGE':
            self.GNN = GraphSAGE(
                in_channels=GNN_config['emb_dim'],
                hidden_channels=GNN_config['emb_dim'],
                num_layers=GNN_config['num_layer'],
                edge_dim=1,
                dropout=GNN_config['drop_ratio'],
                act='relu',
                norm="BatchNorm",
            )
        elif gnn == 'SchNet':
            self.GNN = tmqm_SchNet(
                hidden_channels=GNN_config['emb_dim'],
                num_filters=GNN_config['emb_dim'],
                num_interactions=6,
                cutoff=10
            )
        elif gnn == 'AttentiveFP':
            self.GNN = tmqm_AttentiveFP(
                in_channels=GNN_config['emb_dim'],
                hidden_channels=Coor_config['d_feature'],
                out_channels=8,     
                num_layers=GNN_config['num_layer'],
                dropout=GNN_config['drop_ratio'],
                edge_dim=1,
                num_timesteps=2      
            )
        
        self.separated = separated
        self.attention = attention
        self.Coor_config = Coor_config
        self.GNN_config = GNN_config
        self.metal_length = Coor_config['metal_length']
        self.interprate = Coor_config['interprate']
        self.gnn = gnn

        if separated:
            self.CoordiNet = CoordiNet(Coor_config['d_feature'],
                                        Coor_config['d_k'],
                                        Coor_config['apool'],
                                        Coor_config['drop_ratio'],
                                        Coor_config['kernel'],
                                        Coor_config['interprate']
                                        )
            
        self.ligand_layer = Seq(
            nn.Linear(Coor_config['d_feature'],Coor_config['d_k']),
            nn.Softplus()
        )

        self.metal_layer = Seq(
            nn.Linear(Coor_config['d_feature'],Coor_config['d_k']),
            nn.Softplus()
        )
        
        self.pre_head = Seq(
            Linear(self.Coor_config['d_k'],2*self.Coor_config['d_k']),
            nn.Softplus(),
            Linear(2*self.Coor_config['d_k'],out_dimention)
        )

        self.x_embedding = nn.Embedding(num_atom_type,GNN_config['emb_dim'])
        self.metal_embedding = nn.Embedding(num_atom_type,GNN_config['emb_dim'])
        self.x_weight = Parameter(torch.Tensor(3,GNN_config['emb_dim']))

        self.metal_encoding = nn.Linear(self.metal_length-1,GNN_config['emb_dim'])

        nn.init.xavier_uniform_(self.x_embedding.weight.data)
        nn.init.xavier_uniform_(self.x_weight)

        nn.init.xavier_uniform_(self.metal_embedding.weight.data)
        nn.init.xavier_uniform_(self.metal_encoding.weight)

        if not self.attention:
            if Coor_config['pool'] == 'mean':
                self.pool = global_mean_pool
            elif Coor_config['pool'] == 'add':
                self.pool = global_add_pool
            elif Coor_config['pool'] == 'max':
                self.pool = global_max_pool
            else:
                raise ValueError('Not defined pooling!')

    def forward(self,data):
        if self.GNN_config['node_type']=='2d':
            h = self.x_embedding(data.x[:,0].int())
        elif self.GNN_config['node_type']=='3d':
            h = self.x_embedding(data.x[:,0].int()) + data.x[:,1:] @ self.x_weight
        batch = data.batch
        edge_index = data.edge_index
        if self.GNN_config['edge_type'] == '2d':
            edge_attr = data.edge_attr[:,3:]
        elif self.GNN_config['edge_type'] == '3d':
            edge_attr = data.edge_attr[:,0:3]
        else:
            edge_attr = data.edge_attr

        if self.separated:
            metals = data.metal.reshape(-1,self.metal_length)
            metal_feature = self.metal_encoding(metals[:,0:self.metal_length-1].float()) + self.metal_embedding(metals[:,self.metal_length-1].int())

            if self.gnn == 'SchNet':
                h = self.GNN(data,separated=True)
            elif self.gnn == 'AttentiveFP':
                pass
            else:
                h = self.GNN(x=h,edge_index=edge_index,edge_attr=edge_attr)
            metal = self.metal_layer(metal_feature)

            if self.attention:
                if self.interprate:
                    h,weight,s = self.CoordiNet(h,metal_feature,batch,edge_index)
                    feature = h + self.Coor_config['mp']*metal
                else:
                    if self.gnn == 'AttentiveFP':
                        h = self.GNN(x=h,edge_index=edge_index,edge_attr=edge_attr,batch=batch,attention=self.attention,separated=self.separated)
                    h = self.CoordiNet(h,metal_feature,batch,edge_index)
                    feature = h + self.Coor_config['mp']*metal
            else:
                if self.gnn == 'AttentiveFP':
                    h = self.GNN(x=h,edge_index=edge_index,edge_attr=edge_attr,batch=batch,attention=self.attention,separated=self.separated)
                else:
                    h = self.pool(h,batch)
                li_featrue = self.ligand_layer(h)
                feature = li_featrue + self.Coor_config['mp']*metal
            out = self.pre_head(feature)
            if self.interprate:
                return out,weight,s
        else:
            if self.gnn == 'SchNet':
                out = self.GNN(data,separated=False)
            elif self.gnn == 'AttentiveFP':
                out = self.GNN(x=h,edge_index=edge_index,edge_attr=edge_attr,batch=batch,attention=False,separated=False)
            else:
                h = self.GNN(x=h,edge_index=edge_index,edge_attr=edge_attr)
                h = self.pool(h,batch)
                li_featrue = self.ligand_layer(h)
                out = self.pre_head(li_featrue)

        return out