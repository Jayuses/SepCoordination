import torch
import sys
sys.path.append('./models')
import torch.nn as nn
from torch.nn import Sequential as Seq, Linear, Parameter
import torch.nn.functional as F
from einops import repeat

from sat import GraphTransformer
from sc_net import CoordiNet

num_atom_type = 119

class tmqm_SAT(GraphTransformer):
    def __init__(self,d_feature,d_k,out_dimention,apool='apool',mp=1,separated=False,attention=False):
        super(tmqm_SAT,self).__init__(
            in_size=4,
            num_class=out_dimention,
            d_model=d_feature,
            use_edge_attr=True,
            global_pool='mean',
            num_layers=2,
            dim_feedforward=256
        )

        self.separated = separated
        self.attention = attention
        self.mp = mp

        self.x_embedding = nn.Embedding(num_atom_type,d_feature)
        self.x_weight = Parameter(torch.Tensor(3,d_feature))

        self.edge_weight1 = Parameter(torch.Tensor(3,d_feature))
        self.edge_weight2 = Parameter(torch.Tensor(1,d_feature))

        nn.init.xavier_uniform_(self.x_embedding.weight.data)
        nn.init.xavier_uniform_(self.x_weight.data)

        nn.init.xavier_uniform_(self.edge_weight1)
        nn.init.xavier_uniform_(self.edge_weight2)

        self.metal_embedding1 = nn.Linear(17,d_feature)
        self.metal_embedding2 = nn.Embedding(119,d_feature)

        nn.init.xavier_uniform_(self.metal_embedding1.weight)
        nn.init.xavier_uniform_(self.metal_embedding2.weight.data)

        self.pre_head = Seq(
            Linear(d_k,2*d_k),
            nn.Softplus(),
            Linear(2*d_k,out_dimention)
        )

        self.CoordiNet = CoordiLayer(d_feature,d_k,apool)

        self.metal_layer = Seq(
            nn.Linear(d_feature,d_k),
            nn.Softplus()
        )

        self.ligand_layer = Seq(
            nn.Linear(d_feature,d_k),
            nn.Softplus()
        )

    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # node_depth = data.node_depth if hasattr(data, "node_depth") else None
        
        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator 
            subgraph_edge_attr = data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") \
                                    else None
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None
        # output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))
        output = self.x_embedding(x[:,0].int()) + x[:,1:] @ self.x_weight
            
        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            # edge_attr = self.embedding_edge(edge_attr)
            edge_attr = edge_attr[:,0:3] @ self.edge_weight1 + edge_attr[:,3:] @ self.edge_weight2
            if subgraph_edge_attr is not None:
                # subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
                subgraph_edge_attr = subgraph_edge_attr[:,0:3] @ self.edge_weight1 + subgraph_edge_attr[:,3:] @ self.edge_weight2
        else:
            edge_attr = None
            subgraph_edge_attr = None

        if self.global_pool == 'cls' and self.use_global_pool:
            bsz = len(data.ptr) - 1
            if complete_edge_index is not None:
                new_index = torch.vstack((torch.arange(data.num_nodes).to(data.batch), data.batch + data.num_nodes))
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat((
                    complete_edge_index, new_index, new_index2, new_index3), dim=-1)
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(data.batch)
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack((subgraph_indicator_index, idx_tmp))
            degree = None
            cls_tokens = repeat(self.cls_token, '() d -> b d', b=bsz)
            output = torch.cat((output, cls_tokens))

        h = self.encoder(
            output, 
            edge_index, 
            complete_edge_index,
            edge_attr=edge_attr, 
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index, 
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )

        if self.separated:
            metals = data.metal.reshape(-1,18)
            metal = self.metal_embedding1(metals[:,0:17].float()) + self.metal_embedding2(metals[:,17])
            metal_feature = self.metal_layer(metal)
            if self.attention:
                h = self.CoordiNet(h,metal,data.batch,data.edge_index)
                h = h + self.mp*metal_feature
            else:
                if self.use_global_pool:
                    if self.global_pool == 'cls':
                        h = h[-bsz:]
                    else:
                        h = self.pooling(h, data.batch)
                li_featrue = self.ligand_layer(h)
                h = li_featrue + self.mp*metal_feature

            out = self.pre_head(h)
        else:
            if self.use_global_pool:
                if self.global_pool == 'cls':
                    h = h[-bsz:]
                else:
                    h = self.pooling(h, data.batch)
            li_featrue = self.ligand_layer(h)
            out = self.pre_head(li_featrue)
                
        return out