import numpy as np
import torch
import copy
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9
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
    coor = [[xyz[j][i]-xyz[ref][i] for i in range(3)] for j in range(len(xyz))]

    return np.array(coor)

def distance(xyz):
    xyz = np.array(xyz)
    dist = []
    for i in range(xyz.shape[0]):
        dist.append(np.linalg.norm(xyz[i,:]-xyz,axis=1))
    return np.array(dist)

def DG(xyz,dim):
    x = copy.deepcopy(xyz)
    dist = distance(x)
    center = np.sum(x,axis=0)/x.shape[0]
    
    x = x-center
    do = np.linalg.norm(x,axis=1)
    do2 = np.tile(do**2,len(do)).reshape(-1,len(do))
    G = (do2+do2.T-dist**2)/2
    val,vec = np.linalg.eig(G)
    val = np.real(val)
    vec = np.real(vec)
    ind = np.argsort(val)[::-1][:dim]
    eigv = np.array([np.sqrt(abs(i)) if i!=0 else 0 for i in val[ind]])
    coor = vec[:,ind] * eigv

    if coor.shape[1] < dim:
        coor = np.pad(coor,((0,0),(0,dim-coor.shape[1])),mode='constant')

    return coor

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
    level_emb = [0 for _ in range(len(energy_level))]
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
    sub_num = len(graph_list)
    for ind,data in enumerate(graph_list):
        num_nodes = data.num_nodes
        node_split.append((sub_num-ind-1)*torch.ones((num_nodes,1),dtype=torch.int))
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
    combined_graph.node_split = torch.cat(node_split, dim=0)
    combined_graph.subgraph_num = torch.tensor(len(graph_list))

    return combined_graph

def pos_encode(coor=None,sigma=1,dim=3,type = 'laplace'):
    if type == 'laplace':
        w = copy.deepcopy(coor)
        non_zero = np.nonzero(coor)
        w[non_zero] = np.exp(-w[non_zero]**2/(2*sigma**2))
        d = np.diag(np.sum(w,axis=1))
        L = d-w
        val,vec = np.linalg.eig(L)
        sel = []
        val_sort = np.argsort(val)
        for i in val_sort:
            if val[i] > 1e-8:
                sel.append(i)
            if len(sel) == dim:
                break
        pe = vec[:,sel]
        if len(sel)<dim:
            pe = np.pad(pe,((0,0),(0,dim-len(sel))),mode='constant')
    elif type == 'DG':
        pe = DG(coor,dim)
    else:
        print('Undefined position code!')

    return pe

class tmQM_dataset(Dataset):
    def __init__(self,path,separated=False,pe='laplace',metal_list=None,index_list=None,qm9=False):        
        super().__init__()
        if qm9:
            self.qm = QM9(root='./dataset/Data/QM9/')
        else:
            with open(path['complex'],'rb') as f:
                temp = pickle.load(f)
            if metal_list != None:
                metal_list = [periodic_table[idx] for idx in metal_list]
                self.complexes = []
                metal_idx = []
                for idx,comp in enumerate(temp):
                    if comp['metal'] in metal_list:
                        metal_idx.append(idx)
                        self.complexes.append(comp)
            else:
                self.complexes = temp
            self.separated = separated
            if separated:
                with open(path['adjs'],'rb') as f:
                    temp = pickle.load(f)
                if metal_list != None:
                    self.adjs = [temp[idx] for idx in metal_idx]
                else:
                    self.adjs = temp

                with open(path['nodes'],'rb') as f:
                    temp = pickle.load(f)
                if metal_list != None:
                    self.nodes = [temp[idx] for idx in metal_idx]
                else:
                    self.nodes = temp

                self.metal_level = pd.read_csv(path['metal_level'],delimiter=',',
                                            usecols=['AtomicNumber','Symbol','ElectronConfiguration']).to_numpy()
            else:
                with open(path['entirety'],'rb') as f:
                    temp = pickle.load(f)
                if metal_list != None:
                    self.adjs = [temp[idx] for idx in metal_idx]
                else:
                    self.adjs = temp

        self.pe = pe
        self.qm9 = qm9
        self.data_list = self.process(index_list)

    def process_step(self,index):
        if self.qm9:
            x1 = self.qm[index].z.unsqueeze(dim=1)
            edge_index = self.qm[index].edge_index
            edge_attr = []
            if self.pe != 'off':
                if self.pe == 'laplace':
                    dist = distance(self.qm[index].pos.numpy())
                    xyz = pos_encode(dist,type=self.pe)
                elif self.pe == 'DG':
                    xyz = self.qm[index].pos.numpy()
                    xyz = pos_encode(xyz,type=self.pe)
                else:
                    xyz = self.qm[index].pos.numpy()
                
                x2 = torch.tensor(xyz,dtype=torch.float32).view(-1,3)
                x = torch.cat([x1,x2],dim=-1)

                for pos in edge_index.T:
                    edge_xyz = [xyz[pos[1]][i] - xyz[pos[0]][i] for i in range(3)]
                    edge_attr.append(edge_xyz)
            else:
                x = x1

            y = self.qm[index].y
            edge_attr = torch.tensor(edge_attr,dtype=torch.float32)
            graph = Data(x,edge_index,edge_attr)

            graph.y = y

            return graph

        else:
            if self.separated:
                graphs = []
                atoms = [self.complexes[index]['atoms'][i] 
                        for i in range(len(self.complexes[index]['atoms'])) 
                        if i != self.complexes[index]['metal_pos']]
                    
                if self.pe == 'laplace':
                    dist = np.delete(self.complexes[index]['dist'],self.complexes[index]['metal_pos'],0)
                    dist = np.delete(dist,self.complexes[index]['metal_pos'],1)
                else:
                    xyz = [self.complexes[index]['xyz'][i] 
                        for i in range(len(self.complexes[index]['xyz'])) 
                        if i != self.complexes[index]['metal_pos']]

                adj = self.adjs[index]
                y = torch.tensor(self.complexes[index]['y'],dtype=torch.float32)
                metal = query_level(self.complexes[index]['metal'],self.metal_level)
                metal.append(self.metal_level[np.squeeze(
                    np.where(self.metal_level[:,1] == self.complexes[index]['metal'])),0])

                metal = torch.tensor(metal)
                
                for ind,node in enumerate(self.nodes[index]):
                    atomic_number = [periodic_table.index(i)+1 for i in [atoms[j] for j in node]]
                    edge_attr = []
                    bond_pos = np.argwhere(adj[ind]!=0)
                    edge_index = torch.tensor(bond_pos.T,dtype=torch.int64)
                    x1 = torch.tensor(atomic_number,dtype=torch.float32).view(-1,1)

                    if self.pe != 'off':
                        if self.pe == 'laplace':
                            sub_dist = dist[node,:]
                            sub_dist = sub_dist[:,node]
                            sub_xyz = pos_encode(sub_dist,type=self.pe)
                        elif self.pe == 'DG':
                            sub_xyz = re_xyz([xyz[j] for j in node],adj[ind])
                            sub_xyz = pos_encode(sub_xyz,type=self.pe)
                        else:
                            sub_xyz = re_xyz([xyz[j] for j in node],adj[ind])
                    
                        # sub_xyz = np.append(sub_xyz,np.ones((sub_xyz.shape[0],1))*ind,axis=1)
                        x2 = torch.tensor(sub_xyz,dtype=torch.float32).view(-1,3)

                        for pos in bond_pos:
                            edge_xyz = [sub_xyz[pos[1]][i] - sub_xyz[pos[0]][i] for i in range(3)]
                            edge_xyz.append(adj[ind][pos[0],pos[1]])
                            edge_attr.append(edge_xyz)
                        x = torch.cat([x1,x2],dim=1)
                    else:
                        # x2 = torch.tensor(np.ones((x1.shape[0],1))*ind)
                        x = x1
                        for pos in bond_pos:
                            edge_attr.append([adj[ind][pos[0],pos[1]]])

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
                    combined_graph.sample_ind = index

                    return combined_graph
            else:
                atoms = self.complexes[index]['atoms']
                adj = self.adjs[index]
                atomic_number = [periodic_table.index(i)+1 for i in atoms]

                x1 = torch.tensor(atomic_number,dtype=torch.float32).view(-1,1)
                edge_attr = []
                bond_pos = np.argwhere(adj!=0)
                edge_index = torch.tensor(bond_pos.T,dtype=torch.int64)

                if self.pe != 'off':
                    if self.pe == 'laplace':
                        dist = self.complexes[index]['dist']
                        sub_xyz = pos_encode(dist,type=self.pe)
                    elif self.pe == 'DG':
                        xyz = self.complexes[index]['xyz']
                        sub_xyz = np.array(xyz)
                        sub_xyz = pos_encode(sub_xyz,type=self.pe)
                    else:
                        sub_xyz = self.complexes[index]['xyz']
                        
                    x2 = torch.tensor(sub_xyz,dtype=torch.float32).view(-1,3)
                    x = torch.cat([x1,x2],dim=-1)

                    for pos in bond_pos:
                        edge_xyz = [sub_xyz[pos[1]][i] - sub_xyz[pos[0]][i] for i in range(3)]
                        edge_xyz.append(adj[pos[0],pos[1]])
                        edge_attr.append(edge_xyz)
                else:
                    x = x1
                    for pos in bond_pos:
                        edge_attr.append([adj[pos[0],pos[1]]])

                y = torch.tensor(self.complexes[index]['y'],dtype=torch.float32)
                edge_attr = torch.tensor(edge_attr,dtype=torch.float32)
                graph = Data(x,edge_index,edge_attr)

                graph.y = y
                graph.sample_ind = index

                return graph
        
    def process(self,index_list):
        data_list = []
        if self.qm9:
            sample_num = len(self.qm)
            for index in range(sample_num):
                data_list.append(self.process_step(index))
        else:
            if index_list != None:
                for index in index_list:
                    data_list.append(self.process_step(index))
            else:
                sample_num = len(self.complexes)
                for index in range(sample_num):
                    data_list.append(self.process_step(index))

        return data_list
            
    def __getitem__(self,index):
        return self.data_list[index]
        
    def __len__(self):
        return len(self.data_list)
                
class tmQM_wrapper(object):
    def __init__(self,path,batch_size=32,num_workers=0,valid_size=0.1,test_size=0.1,separated=True,pe='laplace',predata=None,qm9=False):
        super(object,self).__init__()
        self.separated = separated
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.test_size = test_size
        self.path = path
        self.predata = predata
        self.pe = pe
        self.qm9 = qm9

    def get_data_loaders(self):
        if self.predata != None:
            dataset = self.predata
        else:
            dataset = tmQM_dataset(self.path,self.separated,self.pe,qm9=self.qm9)
            
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