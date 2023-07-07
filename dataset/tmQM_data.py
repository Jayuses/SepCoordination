import numpy as np
import torch
from rdkit import Chem
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pandas as pd
import pickle

periodic_table = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                  'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 
                  'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 
                  'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 
                  'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 
                  'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm','Md', 'No', 'Lr',
                  'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og', 'Uue')  

energy_level = ('3d','4s','4p','4d','4f','5s','5p','5d','5f','5g','6s','6p','6d','6f','6g','6h','7s')

def re_xyz(xyz,adj):

    ref = [len(np.where(adj[i,:]!=0)[0]) for i in range(adj.shape[0])]
    ref = ref.index(max(ref))

    return [[xyz[j][i]-xyz[ref][i] for i in range(3)] for j in range(len(xyz))]

def query_level(metal,metal_level):
    level_dic = []
    level = metal_level[np.where(metal_level[:,1]==metal)[0],2][0]
    for item in level.split(' '):
        if item[0] == '[' and len(item) == 4:
            continue
        elif item[0] == '[' and len(item) > 4:
            level_dic.append({
                'level':item[4:6],
                'ele_num':int(item[6])
            })
        else:
            level_dic.append({
                'level':item[0:2],
                'ele_num':int(item[2])
            })
    level_emb = torch.zeros(len(energy_level)+1,dtype=int)
    for item in level_dic:
        level_emb[energy_level.index(item['level'])] = item['ele_num']
    return level_emb

def combine_graph(graph_list):
    combined_graph = Data()
    combined_graph.x = torch.cat([data.x for data in graph_list], dim=0)
    combined_edge_indices = []
    combined_edge_attrs = []
    node_split = []
    node_offset = 0
    for data in graph_list:
        num_nodes = data.num_nodes
        node_split.append(num_nodes)
        # Adjust edge indices based on node offset
        edge_index = data.edge_index.clone()
        edge_index[0] += node_offset
        edge_index[1] += node_offset
        combined_edge_indices.append(edge_index)

        # Concatenate edge attributes
        combined_edge_attrs.append(data.edge_attr)

        node_offset += num_nodes

    combined_graph.edge_index = torch.cat(combined_edge_indices, dim=1)
    combined_graph.edge_attr = torch.cat(combined_edge_attrs, dim=0)
    combined_graph.node_split = torch.tensor(node_split,dtype=int)

    return combined_graph

class tmQM_dataset(Dataset):
    def __init__(self,path,separated=False):        
        super().__init__()
        with open(path['complex'],'rb') as f:
            self.complexes = pickle.load(f)
        self.separated = separated
        if separated:
            with open(path['adjs'],'rb') as f:
                self.adjs = pickle.load(f)
            with open(path['nodes'],'rb') as f:
                self.nodes = pickle.load(f)
            self.metal_level = pd.read_csv(path['metal_level'],delimiter=',',
                                           usecols=['AtomicNumber','Symbol','ElectronConfiguration']).to_numpy()
        else:
            with open(path['entirety'],'rb') as f:
                self.adjs = pickle.load(f)

        self.data_list = self.process()

    def process_step(self,index):
        if self.separated:
            graphs = []
            atoms = [self.complexes[index]['atoms'][i] 
                    for i in range(len(self.complexes[index]['atoms'])) 
                    if i != self.complexes[index]['metal_pos']]
            xyz = [self.complexes[index]['xyz'][i] 
                    for i in range(len(self.complexes[index]['xyz'])) 
                    if i != self.complexes[index]['metal_pos']]
            adj = self.adjs[index]
            y = torch.tensor(self.complexes[index]['y'],dtype=torch.float32)
            metal = query_level(self.complexes[index]['metal'],self.metal_level)
            metal[-1] = torch.tensor(self.metal_level[np.squeeze(
                np.where(self.metal_level[:,1] == self.complexes[index]['metal'])),0])
            
            for ind,node in enumerate(self.nodes[index]):
                atomic_number = [periodic_table.index(i)+1 for i in [atoms[j] for j in node]]
                sub_xyz = re_xyz([xyz[j] for j in node],adj[ind])

                x1 = torch.tensor(atomic_number,dtype=torch.float32).view(-1,1)
                x2 = torch.tensor(sub_xyz,dtype=torch.float32).view(-1,3)
                x = torch.cat([x1,x2],dim=1)

                edge_attr = []
                bond_pos = np.argwhere(adj[ind]!=0)
                edge_index = torch.tensor(bond_pos.T,dtype=torch.int64)
                for pos in bond_pos:
                    edge_xyz = [sub_xyz[pos[1]][i] - sub_xyz[pos[0]][i] for i in range(3)]
                    edge_xyz.append(adj[ind][pos[0],pos[1]])
                    edge_attr.append(edge_xyz)
                edge_attr = torch.tensor(edge_attr,dtype=torch.float32)
                data = Data(x,edge_index,edge_attr)

                graphs.append(data)  # The graph set will be returned.
            
            if len(graphs) == 0:
                print(F"ERROR SAMPLE:{self.complexes[index]['CSD_code']}")
                idx = np.random.randint(0,len(self.complexes)-1)
                return self.process_step(idx)
            else:
                combined_graph = combine_graph(graphs)
                combined_graph.metal = metal
                combined_graph.y = y

                return combined_graph
        else:
            atoms = self.complexes[index]['atoms']
            xyz = self.complexes[index]['xyz']
            adj = self.adjs[index]
            atomic_number = [periodic_table.index(i)+1 for i in atoms]
            sub_xyz = re_xyz(xyz,adj)

            x1 = torch.tensor(atomic_number,dtype=torch.float32).view(-1,1)
            x2 = torch.tensor(sub_xyz,dtype=torch.float32).view(-1,3)
            x = torch.cat([x1,x2],dim=-1)
            y = torch.tensor(self.complexes[index]['y'],dtype=torch.float32)

            edge_attr = []
            bond_pos = np.argwhere(adj!=0)
            edge_index = torch.tensor(bond_pos.T,dtype=torch.int64)
            for pos in bond_pos:
                edge_xyz = [sub_xyz[pos[1]][i] - sub_xyz[pos[0]][i] for i in range(3)]
                edge_xyz.append(adj[pos[0],pos[1]])
                edge_attr.append(edge_xyz)
            edge_attr = torch.tensor(edge_attr,dtype=torch.float32)
            graph = Data(x,edge_index,edge_attr)

            graph.y = y

            return graph
        
    def process(self):
        data_list = []
        for index in range(len(self.complexes)):
            data_list.append(self.process_step(index))

        return data_list
            
    def __getitem__(self,index):
        return self.data_list[index]
        
    def __len__(self):
        return len(self.data_list)
                
class tmQM_wrapper(object):
    def __init__(self,path,batch_size,num_workers,valid_size,test_size,separated):
        super(object,self).__init__()
        self.separated = separated
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.path = path

    def get_data_loaders(self):
        dataset = tmQM_dataset(self.path,self.separated)
        num_data = len(dataset)
        indices = list(range(num_data))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_data))
        split2 = int(np.floor(self.test_size * num_data))
        valid_idx, test_idx, train_idx = indices[:split], indices[split:split+split2], indices[split+split2:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        if self.separated:
            train_loader = DataLoader(
                dataset, batch_size=self.batch_size, sampler=train_sampler,
                num_workers=self.num_workers, drop_last=False
            )
            valid_loader = DataLoader(
                dataset, batch_size=self.batch_size, sampler=valid_sampler,
                num_workers=self.num_workers, drop_last=False
            )
            test_loader = DataLoader(
                dataset, batch_size=self.batch_size, sampler=test_sampler,
                num_workers=self.num_workers, drop_last=False
            )
        else:
            train_loader = DataLoader(
                dataset, batch_size=self.batch_size, sampler=train_sampler,
                num_workers=self.num_workers, drop_last=False
            )
            valid_loader = DataLoader(
                dataset, batch_size=self.batch_size, sampler=valid_sampler,
                num_workers=self.num_workers, drop_last=False
            )
            test_loader = DataLoader(
                dataset, batch_size=self.batch_size, sampler=test_sampler,
                num_workers=self.num_workers, drop_last=False
            )

        return train_loader,valid_loader,test_loader
    