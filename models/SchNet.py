import torch
import torch.nn as nn
import pickle
import copy
from torch_geometric.nn.models import SchNet

def split_batches(batch,node_split,subgraph_num):
    sub_num  = copy.deepcopy(subgraph_num)
    split_batch = copy.deepcopy(batch)
    for i in range(1,len(subgraph_num)):
        sub_num[i] += sub_num[i-1]

    for ind in range(len(sub_num)):
        split_batch[torch.where(batch==ind)] = sub_num[ind]-1
    
    split_batch = split_batch-node_split.squeeze()

    return split_batch.to(batch.device)

class tmqm_SchNet(SchNet):
    def __init__(self,hidden_channels,num_filters,num_interactions,cutoff,num_gaussians=50,
                 interaction_graph=None,max_num_neighbors=32,readout='add',dipole=False,mean=None,std=None,atomref=None):
        super().__init__(hidden_channels,num_filters,num_interactions,num_gaussians,
                 cutoff,interaction_graph,max_num_neighbors,readout,dipole,mean,std,atomref)
        self.lin2 = nn.Linear(self.hidden_channels // 2, 8)
    
    def forward(self,data,separated):
        z = data.x[:,0].int()
        pos = data.x[:,1:]
        batch = data.batch
        h = self.embedding(z)
        if separated:
            if len(data.subgraph_num.shape) == 0:
                split_batch = data.node_split.squeeze()
            else:
                split_batch = split_batches(batch,data.node_split,data.subgraph_num)
            edge_index, edge_weight = self.interaction_graph(pos,split_batch)
        else:
            edge_index, edge_weight = self.interaction_graph(pos,batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, edge_index, edge_weight, edge_attr)

        if separated:
            return h
        else:
            h = self.lin1(h)
            h = self.act(h)
            h = self.lin2(h)
            out = self.readout(h, batch, dim=0)
            return out
            
        # h = self.lin1(h)
        # h = self.act(h)
        # h = self.lin2(h)

        # if self.dipole:
        #     # Get center of mass.
        #     mass = self.atomic_mass[z].view(-1, 1)
        #     M = self.sum_aggr(mass, batch, dim=0)
        #     c = self.sum_aggr(mass * pos, batch, dim=0) / M
        #     h = h * (pos - c.index_select(0, batch))

        # if not self.dipole and self.mean is not None and self.std is not None:
        #     h = h * self.std + self.mean

        # if not self.dipole and self.atomref is not None:
        #     h = h + self.atomref(z)
            


if __name__ == '__main__':
    with open('../../dataset/data/complex.pkl','rb') as f:
        complexes = pickle.load(f)
    complexes