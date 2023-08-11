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
from dataset.tmQM_data import tmQM_wrapper
from dataset.tmQM_data import tmQM_dataset

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
        dir_name = current_time + '_' + config['experiment']['suffix']
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
        pred = model(data)
        if self.config['label_normalize']:
            loss = self.criterion(pred, self.normalizer.norm(data.y.reshape(-1,pred.size(-1))))
        else:
            loss = self.criterion(pred, data.y.reshape(-1,pred.size(-1)))

        return loss
        
    
    def train(self):
        print(time.ctime())
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        with open(self.config['experiment']['path']+'/testset.pkl','wb') as f:
            pickle.dump(test_loader.sampler.indices,f)

        model = SCnet(
            padden_len=self.config['data']['padden_len'],
            GNN_config=self.config['GNN'],
            Coor_config=self.config['Coor_config'],
            out_dimention=self.config['out_dimention'],
            separated=self.config['separated'],
        )

        model.to(self.device)

        if self.config['label_normalize']:
            labels = []
            for d in train_loader:
                labels.append(d.y.reshape(-1,8))
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

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0
        start = time.time()

        for epoch in range(self.config['epochs']):
            n_iter = 0
            for bn,data in enumerate(train_loader):
                optimizer.zero_grad()
                
                data = data.to(self.device)
                loss = self._step(model,data)
                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch, bn, loss.item())
                
                loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            end = time.time()
            print(f'Running Time: {end-start} seconds')
            start = end

        self._test(model, test_loader)
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

def main(config,predata):
    dataset = tmQM_wrapper(config['path'],config['batch_size'],separated=config['separated'],predata=predata,**config['data'])

    sepcoor = SepCoordination(dataset,config)
    sepcoor.train()

    return sepcoor.mae

if __name__ == "__main__":
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    task_list = ['diffpool_32']

    print('Generate datasets......')
    predata = tmQM_dataset(config['path'],config['separated'])
    
    if config['experiment']['name'] == 'test':
            config['experiment']['path'] = 'experiment/test'
            config['experiment']['suffix'] = 'test'
            if not os.path.exists(config['experiment']['path']):
                os.mkdir(config['experiment']['path'])

            print(f"Separated:{config['separated']}")
            print(f"Experiment:{config['experiment']['name']}\n")

            results = main(config,predata=predata)
    else:
        for task in task_list:
            if task[:8] == 'diffpool':
                config['experiment']['name'] = task
                config['experiment']['path'] = 'experiment/'+task
                config['experiment']['suffix'] = 'df'+task[-2:]
                config['Coor_config']['name'] = 'attention'
                config['Coor_config']['apool'] = {'name':'diff_pool','num':int(task[-2:])}
                config['Coor_config']['pool'] = False

            elif task == 'NoAttention':
                config['experiment']['name'] = 'NoAttention'
                config['experiment']['path'] = 'experiment/NoAttention'
                config['experiment']['suffix'] = 'na'
                config['Coor_config']['name'] = 'NonAttention'
                config['Coor_config']['pool'] = 'mean'

            elif task[:8] == 'trainAtt':
                config['experiment']['name'] = task
                config['experiment']['path'] = 'experiment/'+task
                config['experiment']['suffix'] = 'tA'+task[-3:]
                config['data']['valid_size'] = (1-float(task[-3:]))/2
                config['data']['test_size'] = (1-float(task[-3:]))/2
                config['epochs'] = math.floor(0.8/float(task[-3:])*50)
                if config['epochs'] > 100:
                    config['epochs'] = 100
                config['Coor_config']['name'] = 'attention'
                config['Coor_config']['apool'] = {'name':'diff_pool','num':32}
                config['Coor_config']['pool'] = False
            
            elif task[:8] == 'trainNoA':
                config['experiment']['name'] = task
                config['experiment']['path'] = 'experiment/'+task
                config['experiment']['suffix'] = 'tN'+task[-3:]
                config['data']['valid_size'] = (1-float(task[-3:]))/2
                config['data']['test_size'] = (1-float(task[-3:]))/2
                config['epochs'] = math.floor(0.8/float(task[-3:])*50)
                if config['epochs'] > 100:
                    config['epochs'] = 100
                config['Coor_config']['name'] = 'NonAttention'
                config['Coor_config']['pool'] = 'mean'
                config['Coor_config']['apool'] = False


            if not os.path.exists(config['experiment']['path']):
                os.mkdir(config['experiment']['path'])

            print(f"Separated:{config['separated']}")
            print(f"Train set:{1-config['data']['valid_size']*2}")
            print(f"Epoch:{config['epochs']}")
            print(f"Experiment:{config['experiment']['name']}\n")

            result_list = []
            for i in range(config['repeat']):
                results = main(config,predata=predata)
                result_list.append(results)
            df  = pd.DataFrame(result_list)
            df.loc[len(df)] = df.mean()
            df.loc[len(df)] = df.std()
            df = df.rename(index={df.index[-2]:'mean',df.index[-1]:'std'})
            df.to_csv(
                config['experiment']['path']+'/result.csv',
                mode='a', index=True, header=LABEL_NAME
            )
