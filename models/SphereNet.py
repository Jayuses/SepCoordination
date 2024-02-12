import torch
import copy
import sys
from torch_scatter import scatter
sys.path.append('./models')
import torch.nn as nn
from torch import Tensor
from torch.nn import Sequential as Seq, Linear
import torch.nn.functional as F
from torch_geometric.nn import radius_graph
from torch_geometric.nn.models import SchNet
from sc_net import CoordiNet
from threedgraph.utils import xyz_to_dat
from threedgraph.method.spherenet.spherenet import SphereNet

def split_batches(batch,node_split,subgraph_num):
    sub_num  = copy.deepcopy(subgraph_num)
    split_batch = copy.deepcopy(batch)
    for i in range(1,len(subgraph_num)):
        sub_num[i] += sub_num[i-1]

    for ind in range(len(sub_num)):
        split_batch[torch.where(batch==ind)] = sub_num[ind]-1
    
    split_batch = split_batch-node_split.squeeze()

    return split_batch.to(batch.device)

class tmqm_SphereNet(SphereNet):
    def __init__(self,out_dimention,apool,mp,separated=False,attention=False) -> None:
        super(tmqm_SphereNet,self).__init__(
            out_channels=out_dimention,
            num_layers=5
        )
        self.separated = separated
        self.attention = attention
        self.apool = apool
        self.mp = mp

    def forward(self, batch_data):
        z, pos, batch = batch_data.x[:,0].int(), batch_data.x[:,1:], batch_data.batch
        if self.energy_and_force:
            pos.requires_grad_()

        if self.separated:
            split_batch = split_batches(batch,batch_data.node_split,batch_data.subgraph_num)
        else:
            split_batch = batch

        edge_index = radius_graph(pos, r=self.cutoff, batch=split_batch)
        num_nodes=z.size(0)
        dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(pos, edge_index, num_nodes, use_torsion=True)

        emb = self.emb(dist, angle, torsion, idx_kj)

        #Initialize edge, node, graph features
        e = self.init_e(z, emb, i, j)
        v = self.init_v(e, i)
        u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch) #scatter(v, batch, dim=0)

        for update_e, update_v, update_u in zip(self.update_es, self.update_vs, self.update_us):
            e = update_e(e, emb, idx_kj, idx_ji)
            v = update_v(e, i)
            u = update_u(u, v, batch) #u += scatter(v, batch, dim=0)

        return u

