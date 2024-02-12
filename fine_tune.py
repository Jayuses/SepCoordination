import torch
import os
import yaml
from tqdm import tqdm
import shutil
import numpy as np
import pandas as pd
import torch.nn as nn
from models.sc_net import SCnet
from datetime import datetime
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from transfer.target_dataset import target_dataset,target_wrapper
from sklearn.metrics import mean_absolute_error
import sklearn.metrics as sm
import time
import pickle

def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

def plot_regress(test_p=[],test_t=[],train_p=[],train_t=[],title=None):
    if any(test_p):
        plt.plot(test_t,test_p,'go',label='test')
        y_pre = np.array(test_p)
        y_tre = np.array(test_t)
        R2 = sm.r2_score(y_tre,y_pre)
        rmse = np.sqrt(sm.mean_squared_error(y_tre,y_pre))
        plt.text(np.min(y_tre),np.max(y_tre),'$R^2$(test) = {}'.format(np.around(R2,4)))
        print('RMSE(test)={}'.format(np.around(rmse,4)))
    
    if any(train_p):
        plt.plot(train_t,train_p,'r*',label='train')
        y_pre = np.array(train_p)
        y_tre = np.array(train_t)
        R2 = sm.r2_score(y_tre,y_pre)
        rmse = np.sqrt(sm.mean_squared_error(y_tre,y_pre))
        plt.text(np.max(y_tre),np.min(y_tre),'$R^2$(train) = {}'.format(np.around(R2,4)))
        print('RMSE(train)={}'.format(np.around(rmse,4)))
    
    if title != None:
        plt.title(title)

    if any(test_p) and any(train_p):
        ref = [min(train_t),max(train_t)]
        plt.plot(ref,ref,'b-')
    elif any(test_p) and not any(train_p):
        ref = [min(test_t),max(test_t)]
        plt.plot(ref,ref,'b-')
    elif not any(test_p) and any(train_p):
        ref = [min(train_t),max(train_t)]
        plt.plot(ref,ref,'b-')
    else:
        print('NO Data!')

    plt.xlabel('true')
    plt.ylabel('predicted')
    plt.legend(loc='lower right')

class Normalizer(object):
    """Normalize a Tensor and restore it later. """

    def __init__(self, tensor=None,device='cpu'):
        """tensor is taken as a sample to calculate the mean and std"""
        if tensor == None:
            pass
        else:
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
        self.mean = state_dict['mean'].item()
        self.std = state_dict['std'].item()

class FineTune(object):
    def __init__(self,config,dataset=None,model_path=None):
        self.config = config
        self.device = self._get_device()

        if isinstance(model_path,str):
            if os.path.exists(os.path.join(model_path,'normal.pt')):
                self.model = self._model_init(os.path.join(model_path,'model.pth'),screen=True)
                print('normalizer!')
                state_dict = torch.load(os.path.join(model_path,'normal.pt'))
                self.normalizer = Normalizer()
                self.normalizer.load_state_dict(state_dict)
            else:
                self.model = self._model_init(os.path.join(model_path,'model.pth'))
                current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                dir_name = current_time + '_' + config['experiment']['name']
                log_dir = os.path.join(config['experiment']['path'], dir_name)
                self.writer = SummaryWriter(log_dir=log_dir)
                self.dataset = dataset
        else:
            self.model = None
        self.criterion = nn.MSELoss()
    
    def _model_init(self,pre_train_path,screen=False):
        model = SCnet(
                GNN_config=self.config['GNN'],
                Coor_config=self.config['Coor_config'],
                out_dimention=self.config['out_dimention'],
                separated=self.config['separated'],
                attention=self.config['attention']
            )
        net = torch.load(pre_train_path,map_location=torch.device(self.device))
        if screen:
            pass
        else:
            pre_head = [key for key in list(net.keys()) if key[:3]=='pre']
            for k in pre_head:
                del net[k]
        model_dict = model.state_dict()
        model_dict.update(net)
        model.load_state_dict(model_dict)
        
        return model

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
        else:
            device = 'cpu'
        # print(f'Running on {device}')
        return device

    def _step(self,model,data):
        pred = model(data)
        if self.config['label_normalize']:
            loss = self.criterion(pred, self.normalizer.norm(data.y.reshape(-1,pred.size(-1))))
        else:
            loss = self.criterion(pred, data.y.reshape(-1,pred.size(-1)))

        return loss
    
    def train_result(self,plot=True):
        model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)

        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        train_index = train_loader.sampler.indices
        test_index = test_loader.sampler.indices
        data = train_loader.dataset.data['y']

        dataset = train_loader.dataset
        self.model.eval()
        
        test_p,test_t,train_p,train_t = [],[],[],[]
        for data in train_loader:
            pred = self.model(data.to(self.device))
            pred = self.normalizer.denorm(pred)
            train_p.extend(pred.cpu().detach().numpy())
            train_t.extend(data.y.cpu().flatten().numpy())
        for data in test_loader:
            pred = self.model(data.to(self.device))
            pred = self.normalizer.denorm(pred)
            test_p.extend(pred.cpu().detach().numpy())
            test_t.extend(data.y.cpu().flatten().numpy())

        if plot:
            plt.figure(figsize=(8,8))
            plt.plot(data[train_index],data[train_index],'go',label='train:{}'.format(len(train_index)))
            plt.plot(data[test_index],data[test_index],'r*',label='test:{}'.format(len(test_index)))
            plt.legend(fontsize='large')
            plt.figure(figsize=(8,8))
            plot_regress(test_p,test_t,train_p,train_t)

        return sm.r2_score(train_t,train_p),sm.r2_score(test_t,test_p),self.writer.log_dir


    def train(self,show_loss=True):
        layer_list = []
        # layer_list = []
        for name, _ in self.model.named_parameters():
            # if name[0:8] == 'pre_head' or name[0:9] == 'CoordiNet':
            #     layer_list.append(name)
            if name[0:8] == 'pre_head':
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, self.model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, self.model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],self.config['init_lr'],
            weight_decay=self.config['weight_decay']
        )

        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()
        with open(self.writer.log_dir+'/validset.pkl','wb') as f:
            pickle.dump(valid_loader.sampler.indices,f)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)

        self.model.to(self.device)
        if self.config['label_normalize']:
            labels = []
            for d in train_loader:
                labels.append(d.y.reshape(-1,self.config['out_dimention']))
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels,self.device)
            torch.save(self.normalizer.state_dict(),os.path.join(model_checkpoints_folder, 'normal.pt'))
            if show_loss:
                print('Average:',self.normalizer.mean)
                print('std:',self.normalizer.std)
                print('size:',labels.shape)

        valid_n_iter = 0
        best_valid_loss = np.inf
        best_test_loss = np.inf
        # start = time.time()

        for epoch in range(self.config['epochs']):
            for bn,data in enumerate(train_loader):
                optimizer.zero_grad()
                
                data = data.to(self.device)
                loss = self._step(self.model,data)
                loss.backward()

                optimizer.step()
            
            self.writer.add_scalar('train_loss', loss, global_step=epoch)
            if show_loss:
                if epoch % 1 == 0:
                    print(epoch, bn, loss.item())
            scheduler.step()

            if epoch % self.config['eval_every_n_epochs'] == 0:
                test_loss = self._test(self.model, test_loader,show_loss=show_loss)
                if test_loss < best_test_loss:
                    # save the model weights
                    best_test_loss = test_loss
                    torch.save(self.model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('test_loss', test_loss, global_step=epoch)

            # validate the model if requested
            if epoch % self.config['eval_every_n_epochs'] == 0 and self.config['data']['valid_size'] > 0:
                valid_loss = self._validate(self.model, valid_loader)
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            # end = time.time()
            # print(f'Running Time: {end-start} seconds')
            # start = end
        # torch.save(self.model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
        # if self.config['data']['test_size'] > 0:
        #     self._test(self.model, test_loader)
        # print(time.ctime())

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
    
    def _test(self,model,test_loader,show_loss=True):
        # model_path = os.path.join(self.writer.log_dir, 'checkpoints', 'model.pth')
        # state_dict = torch.load(model_path, map_location=self.device)
        # model.load_state_dict(state_dict)
        # print("Loaded trained model with success.")

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
        if show_loss:
            print('Test loss:', test_loss)
            print(f'MAE:{self.mae}\n')
        return test_loss

    def predict(self,data):
        self.model.eval()
        prediction = self.model(data)
        prediction = prediction.detach().numpy().squeeze()
        if self.config['label_normalize']:
            prediction = self.normalizer.denorm(prediction)

        return prediction
        
# def integrate(path,data_list,config,bs=64,aug='off'):
#     paths = os.walk(path)
#     path_list = []
#     result = []
#     dataset = target_dataset(screen=data_list,aug=aug)
#     datas = DataLoader(dataset,batch_size=bs)
#     for _, dir_lst, file_lst in paths:
#         for dir_name in dir_lst:
#             path_list.append(os.path.join(path, dir_name))
#     for p in path_list[:int(len(path_list)/2)]:
#         Cu_model = FineTune(config,dataset,model_path=p+'/checkpoints')
#         result.append(Cu_model.predict(datas,dataset,aug=aug))
#     result = np.array(result)
    
#     return result,dataset.error_list,dataset


if __name__ == '__main__': 
    pre_train_path = './experiment/SCnet-GCN_se_att_off/Nov13_07-37-47_SCnet-GCN_se_att_off/checkpoints'
    config = yaml.load(open("config_ft.yaml", "r"), Loader=yaml.FullLoader)
    metal_list = ['Cu']
    result_list = []
    predata = target_dataset(metal_list=metal_list,aug='off',conformer_num=50)
    for i in range(config['repeat']):
        dataset = target_wrapper(config['batch_size'],separated=config['separated'],predata=predata,**config['data'])
        Cu_model = FineTune(config,dataset,model_path=pre_train_path)
        Cu_model.train()
    #     result_list.append(Cu_model.mae)
    # df  = pd.DataFrame(result_list)
    # df.to_csv(
    #             config['experiment']['path']+'/result.csv',
    #             mode='a',index=False, header=False
    #         )
