import os
import math
import pickle
import shutil
import yaml
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_absolute_error

from models.sc_net import SCnet
# from models.SCnet import SCnet
from models.SchNet import tmqm_SchNet
from models.AttentiveFP import tmqm_AttentiveFP
from models.SphereNet import tmqm_SphereNet
from models.SAT import tmqm_SAT

from dataset.tmQM_data import tmQM_wrapper
from dataset.tmQM_data import tmQM_dataset
from models.sat.data import GraphDataset

from torch_geometric.utils import degree

LABEL_NAME = ['Electronic_E','Dispersion_E','Dipole_M','Metal_q','HL_Gap','HOMO_Energy','LUMO_Energy','Polarizability']

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor,device):
        """tensor is taken as a sample to calculate the mean and std"""
        if len(tensor.size()) > 1:
            self.mean = torch.mean(tensor,dim=0).to(device)
            self.std = torch.std(tensor,dim=0).to(device)
        else:
            self.mean = torch.mean(tensor).to(device)
            self.std = torch.std(tensor).to(device)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']

class SepCoordination(object):
    def __init__(self,dataset,config) -> None:
        self.config = config
        self.device = self._get_device()

        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        dir_name = current_time + '_' + config['experiment']['name']
        log_dir = os.path.join(config['experiment']['path'], dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.dataset = dataset

        self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device
    
    def _step(self,model,data):
        if self.config['Coor_config']['interprate']:
            pred,_,_ = model(data)
        else:
            pred = model(data)
        if self.config['label_normalize']:
            loss = self.criterion(pred, self.normalizer.norm(data.y.reshape(-1,self.config['out_dimention'])))
        else:
            loss = self.criterion(pred, data.y.reshape(-1,self.config['out_dimention']))

        return loss
        
    
    def train(self):
        print(time.ctime())
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        with open(self.config['experiment']['path']+'/testset.pkl','wb') as f:
            pickle.dump(test_loader.sampler.indices,f)

        if self.config['model'] == 'SphereNet':
            model = tmqm_SphereNet(
                out_dimention=self.config['out_dimention'],
                apool=self.config['Coor_config']['apool'],
                mp = self.config['Coor_config']['mp'],
                separated=self.config['separated'],
                attention=self.config['attention']
            )
        elif self.config['model'] == 'SAT':
            model = tmqm_SAT(
                d_feature=self.config['Coor_config']['d_feature'],
                d_k=self.config['Coor_config']['d_k'],
                out_dimention=self.config['out_dimention'],
                apool=self.config['Coor_config']['apool'],
                mp = self.config['Coor_config']['mp'],
                separated=self.config['separated'],
                attention=self.config['attention']
            )
        else:
            model = SCnet(
                GNN_config=self.config['GNN'],
                Coor_config=self.config['Coor_config'],
                out_dimention=self.config['out_dimention'],
                separated=self.config['separated'],
                attention=self.config['attention'],
                gnn=self.config['model']
            )

        model.to(self.device)

        if self.config['label_normalize']:
            labels = []
            for d in train_loader:
                labels.append(d.y.reshape(-1,self.config['out_dimention']))
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels,self.device)
            print('Average:',self.normalizer.mean)
            print('std:',self.normalizer.std)
            print('size:',labels.shape)

        optimizer = torch.optim.Adam(
            model.parameters(),
            weight_decay=self.config['weight_decay'],
            lr=self.config['init_lr']
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)
        # _,data = next(enumerate(train_loader))
        # data = data.to(self.device)
        # temp_data = {
        #         'x' : data.x,
        #         'edge_index' : data.edge_index,
        #         'edge_attr' : data.edge_attr,
        #         'y' : data.y,
        #         'batch' : data.batch
        #         }
        # if self.config['separated'] == True:
        #     temp_data['metal'] = data.metal
        # self.writer.add_graph(model,temp_data)

        best_valid_loss = np.inf
        start = time.time()

        for epoch in range(self.config['epochs']):
            n_iter = 0
            for bn,data in enumerate(train_loader):
                optimizer.zero_grad()
                
                data = data.to(self.device)
                loss = self._step(model,data)
                if n_iter % self.config['log_every_n_steps'] == 0:     
                    print(epoch, bn, loss.item())              
                loss.backward()
                optimizer.step()
                n_iter += 1
                
            self.writer.add_scalar('train_loss', loss, global_step=epoch)
            
            scheduler.step()

            # validate the model if requested
            if epoch % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=epoch)
            end = time.time()
            print(f'Running Time: {end-start} seconds')
            start = end

        self._test(model, test_loader)
        if self.config['Coor_config']['interprate']:
            nodes = train_loader.dataset.nodes
            with open('./dataset/Data/start.pkl','rb') as f:
                start = pickle.load(f)
            self._interprate(model,[train_loader,valid_loader,test_loader],nodes,start)

        print(time.ctime())

    def _validate(self,model,valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):           
                data = data.to(self.device)
                if self.config['Coor_config']['interprate']:
                    pred,_,_ = model(data)
                else:
                    pred = model(data)
                loss = self._step(model,data)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.config['label_normalize']:
                    pred = self.normalizer.denorm(pred)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().numpy())

            valid_loss /= num_data
        
        model.train()

        predictions = np.array(predictions)
        labels = np.array(labels).reshape(-1,predictions.shape[1])
        mae = mean_absolute_error(labels, predictions,multioutput='raw_values')
        print('Validation loss:', valid_loss)
        print('MAE:',mae)
        return valid_loss
    
    def _test(self,model,test_loader):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                if self.config['Coor_config']['interprate']:
                    pred,_,_ = model(data)
                else:
                    pred = model(data)
                
                loss = self._step(model,data)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.config['label_normalize']:
                    pred = self.normalizer.denorm(pred)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        predictions = np.array(predictions)
        labels = np.array(labels).reshape(-1,predictions.shape[1])
        self.mae = mean_absolute_error(labels, predictions,multioutput='raw_values')
        print('Test loss:', test_loss)
        print(f'MAE:{self.mae}\n')

    def _interprate(self,model,loaders,nodes,start):
        acc_list = []
        print('INTERPRATE\n')
        for loader in loaders:
            with torch.no_grad():
                model.eval()
            for _,data in enumerate(loader):
                data = data.to(self.device)
                _,weight,s = model(data)
                imp = torch.bmm(s,weight.permute(0,2,1))
                sample_ind = data.sample_ind

                for i,ind in enumerate(sample_ind):
                    acc = False
                    temp = []
                    imp_sort = torch.argsort(imp[i,:,:])
                    for node in nodes[ind]:
                        temp += node
                    m = 0
                    n = len(temp)
                    for s in start[ind]:
                        if s in temp:
                            m += 1
                    for i in imp_sort[-3:]:
                        if temp[i] in start[ind]:
                            acc = True
                            break
                    acc_list.append(acc)

        self.ACC = {sum(acc_list)/len(acc_list)}
        print(f'ACC:{self.ACC}')

def main(config,predata=None):
    dataset = tmQM_wrapper(config['path'],config['batch_size'],separated=config['separated'],predata=predata,**config['data'])

    sepcoor = SepCoordination(dataset,config)
    sepcoor.train()

    if config['Coor_config']['interprate']:
        return sepcoor.mae,sepcoor.ACC
    return sepcoor.mae

if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    task_list1 = ['MPNN_se_no_xyz_2d_2d_1']
    task_list2 = ['MPNN_se_no_xyz_2d_2d_1_interprate',]
    task_list3 = ['AttentiveFP_se_no_xyz_2d_2d_1_3l','AttentiveFP_se_no_xyz_2d_2d_0_3l']
    task_list = task_list1
    # print('Generate datasets......')
    # predata = tmQM_dataset(config['path'],config['separated'])
    
    if config['experiment']['name'] == 'test':
            config['experiment']['path'] = 'experiment/test'
            config['experiment']['suffix'] = 'test'
            if not os.path.exists(config['experiment']['path']):
                os.mkdir(config['experiment']['path'])

            print(f"Separated:{config['separated']}")
            print(f"Experiment:{config['experiment']['name']}\n")

            results = main(config)
    else:
        for task in task_list:
            task_config = task.split(sep='_')
            config['experiment']['name'] = task
            config['experiment']['path'] = 'experiment/'+task
            config['model'] = task_config[0]

            if task_config[1][:2] == 'en':
                config['separated'] = False                      
            else:
                config['separated'] = True
            
            if task_config[2][:3] == 'att':
                config['attention'] = True
                config['Coor_config']['apool'] = int(task_config[2][3:])
            else:
                config['attention'] = False

            config['data']['pe'] = task_config[3]

            config['GNN']['node_type'] = task_config[4]

            config['GNN']['edge_type'] = task_config[5]

            config['Coor_config']['mp'] = float(task_config[6])
            
            if len(task_config) == 8 and task_config[7] == 'interprate':
                config['Coor_config']['interprate'] = True

            if len(task_config) == 8 and task_config[7] == 'QM9':
                config['data']['qm9'] = True

            config['Coor_config']['mp'] = float(task_config[6])

            if task_config[0] == 'SAT':
                predata = GraphDataset(predata)
                config['Coor_config']['d_feature'] = 256
                config['init_lr'] = 0.001

            if task_config[0] == 'SchNet':
                config['init_lr'] = 0.0001

            if task_config[0] == 'SCnet-GIN':
                config['Coor_config']['pool'] = 'mean'

            if not os.path.exists(config['experiment']['path']):
                os.mkdir(config['experiment']['path'])

            print(f"Model:{config['model']}")
            print(f"Separated:{config['separated']}")
            print(f"Attention:{config['attention']}")
            print(f"Interprate:{config['Coor_config']['interprate']}")
            print(f"QM9:{config['data']['qm9']}")
            print(f"Train set:{1-(config['data']['valid_size']+config['data']['test_size'])}")
            print(f"Epoch:{config['epochs']}")
            print(f"Experiment:{config['experiment']['name']}\n")

            result_list = []
            acc_list = []

            print('Generate datasets......\n')
            if len(task_config) == 8 and task_config[7] == 'metal':
                predata = tmQM_dataset(config['path'],config['separated'],pe=config['data']['pe'],metal_list=config['metal_list'])
            elif len(task_config) == 8 and task_config[7] == 'QM9':
                predata = tmQM_dataset(config['path'],config['separated'],pe=config['data']['pe'],qm9=True)
            else:
                predata = tmQM_dataset(config['path'],config['separated'],pe=config['data']['pe'])

            for i in range(config['repeat']):
                print(f'The {i}th experiment......\n')
                if config['Coor_config']['interprate']:
                    results,acc = main(config,predata=predata)
                    acc_list.append(acc)
                else:
                    results = main(config,predata=predata)
                result_list.append(results)

                df  = pd.DataFrame(result_list)
                df.to_csv(
                    config['experiment']['path']+'/result.csv',
                    mode='a',index=False, header=False
                )

                if len(acc_list) != 0:
                    df  = pd.DataFrame(acc_list)
                    df.to_csv(
                        config['experiment']['path']+'/acc.csv',
                        mode='a',index=False, header=False
                    )