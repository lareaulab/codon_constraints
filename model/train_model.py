import time
import os, sys
import json #check if exists
print(os.path.dirname(sys.executable))
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import shutil
from contextlib import redirect_stdout
#from torchvision import transforms, utils
from sklearn.linear_model import LogisticRegression


from process_data import HDF5_dataset, get_dataset_db, collate_1d, collate_all, FEATURE_SETS, FEATURE_SETS_1D

DATA_DIR = '../data/model_training/'
HDF5_NAME = DATA_DIR + 'many_yeast.hdf5'

#currently binary
class GCELoss(nn.Module):
    def __init__(self, q=0.7, reduction='sum'):
        super(GCELoss, self).__init__()
        self.q = q
        self.reduction = reduction
        assert(self.reduction in ['sum','none','mean'])

    def forward(self, logits, targets):
        #expect targets to be 2d
        #targets needs to be int64
        targets = targets.type(torch.int64)
        pred = torch.sigmoid(logits)
        pred = torch.cat((1 - pred, pred), dim=-1)
        Yg = torch.gather(pred, 1, targets)
        loss = ((1-(Yg**self.q))/self.q)
        if self.reduction == 'mean':
            loss = (loss + self.eps).mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        #if none, just return loss
        return loss

class Linear(nn.Module):
    def __init__(self, input_1d_shape=(1367,) , input_2d_shape = (88, 512), output_size=1, #if not binary, output_size should be set to 3
                  model_parameters={}, 
                 include_1d = True, include_2d = True, binary=True):
        super().__init__()
        self.linear = nn.Linear(input_1d_shape[0], 1)
        #if not binary, no activation, use hinge loss
        self.binary = binary

    def forward(self, x):
        c = self.linear(x)
        if self.binary:
            c = nn.functional.sigmoid(c)
        return c 


class MultiInputNet(nn.Module):
    def __init__(self, input_1d_shape=(1367,) , input_2d_shape = (88, 512), output_size=1,  #if not binary, output_size should be set to 3
                  model_parameters={},
                 include_1d = True, include_2d = True, binary=True):
        super().__init__()
        params = {'dropout_rate':0.1, 'layers_1d':2, 'layers_1d_size':128, 
             'dilation_cycles':6, 'dilations_per_cycle':5, 'conv_features':128, 'filter_size':8,
             'layers_combined':8, 'layers_combined_size':128, 'avgpool_output':64, 'batch_norm':True} 
        for p in model_parameters:
            params[p] = model_parameters[p] #override
        print("Model parameters:", params)

        self.include_1d = include_1d
        self.include_2d = include_2d

        #1d block
        dropout_rate = params['dropout_rate']
        layers_1d = params['layers_1d']
        layers_1d_size = params['layers_1d_size']
        if self.include_1d:
            self.layers1d = nn.Sequential()
            input_size = input_1d_shape[0]
            for l in range(layers_1d):
                self.layers1d.append(nn.Linear(input_size, layers_1d_size))
                if params['batch_norm'] == True:
                    self.layers1d.append(nn.BatchNorm1d(layers_1d_size))
                input_size = layers_1d_size #after first layer, size is the same
                self.layers1d.append(nn.ReLU())
                if dropout_rate != 0: #no dropout layers if dropout is zero
                    self.layers1d.append(nn.Dropout(p=dropout_rate))
        
        #2d dilation conv_net block
        dilation_cycles = params['dilation_cycles']
        dilations_per_cycle = params['dilations_per_cycle']
        conv_features = params['conv_features']
        filter_size = params['filter_size']
        avgpool_output = params['avgpool_output']
        if self.include_2d:
            self.conv = DilatedConvNet(input_2d_shape, dilation_cycles, dilations_per_cycle, conv_features, filter_size)
            self.avgpool = nn.AdaptiveAvgPool1d(avgpool_output)
        
        #combination_block
        combined_input_size = (self.include_1d*layers_1d_size 
                               + self.include_2d*avgpool_output*conv_features)
        layers_combined = params['layers_combined']
        layers_combined_size = params['layers_combined_size']
        self.combined = nn.Sequential()
        for i in range(layers_combined):
            self.combined.append(nn.Linear(combined_input_size, layers_combined_size))
            combined_input_size = layers_combined_size #after first layer, size is the same
            self.combined.append(nn.ReLU())
            if dropout_rate != 0:
                self.combined.append(nn.Dropout(p=dropout_rate))

        self.combined.append(nn.Linear(layers_combined_size, output_size))
        self.binary = binary
        

    def forward(self, x):
        if self.include_1d and self.include_2d:
            i1, i2 = x #unpack
            o1 = self.layers1d(i1)
            o2 = self.avgpool(self.conv(i2))
            c = torch.cat((o1.view(o1.size(0), -1),
                          o2.view(o2.size(0), -1)), dim=1)
        elif self.include_1d:
            c = self.layers1d(x)
        elif self.include_2d:
            c = self.avgpool(self.conv(x))
        else:
            raise ValueError('neither 1d nor 2d input??')
        c = self.combined(c)
        if self.binary:
            c = nn.functional.sigmoid(c) 
        return c

class DilatedConvNet(nn.Module):
    def __init__(self, input_2d_shape, dilation_cycles, dilations_per_cycle, conv_features, filter_size):
        super().__init__()
        self.conv_layers = nn.Sequential()
        input_channels = input_2d_shape[0]
        sequence_length = input_2d_shape[1]
        pad = 'same' #padding does not change! avgpool reduces size later
        for i in range(dilation_cycles):
            for j in range(dilations_per_cycle):
                self.conv_layers.append(AddNormConvBlock(input_channels=input_channels, sequence_length=sequence_length,
                                                         output_channels=conv_features, kernel_size=filter_size,
                                                         padding=pad, dilation=2**j))
                input_channels = conv_features #change after first
                
    def forward(self, x):
        return self.conv_layers(x)
    
class AddNormConvBlock(nn.Module):
    def __init__(self, input_channels, sequence_length, output_channels, kernel_size, padding, dilation):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_channels, out_channels=output_channels,
                            kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.layernorm = nn.LayerNorm([output_channels, sequence_length])

    def forward(self, x):
        #only add if in and out are the same
        if self.conv.in_channels == self.conv.out_channels:
            x = x + self.conv(x)
        else:
            x = self.conv(x)
        x = F.relu(self.layernorm(x))
        return x
    
#set device, print some useful info
def set_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type=="cuda":
        print(torch.cuda.get_device_name(0))
        print("Device count:", torch.cuda.device_count(), 
              "Current device:", torch.cuda.current_device())
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    sys.stdout.flush()
    return device

def train_network(model, trainloader, testloader, epochs=10, task="binary", loss_function='default', return_test = False,
                  save_dir=DATA_DIR+'/test_model/', restrict_size=False, pos_per_gene=100, 
                  return_best=True, eval=True):
    print(f"Training network - task is {task}, saving results to {save_dir}")
    sys.stdout.flush()
    device = set_device()
    model_save_path = save_dir + "model.pt" #periodically save the model here while training
    #save the models from the end of each epoch
    epoch_models_save_dir = save_dir + "epoch_models/"
    history_save_path = save_dir + "history.json"
    best_model_save_path = save_dir + "best_model.pt"

    if testloader.dataset.include_index:
        sample_labels_save_path = save_dir + 'sample_indices.pt'
        sample_results_save_path = save_dir + 'sample_results.pt'
        best_sample_labels_save_path = save_dir + 'best_sample_indices.pt'
        best_sample_results_save_path = save_dir + 'best_sample_results.pt'

    if restrict_size:
        print("Restricting size.")
        sys.stdout.flush()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(epoch_models_save_dir):
        os.makedirs(epoch_models_save_dir)

    reduction = "sum"
    if trainloader.dataset.has_weights:
        reduction = 'none' #will multiply by weights later to reduce

    if task == "classifier":
        criterion = nn.CrossEntropyLoss(reduction=reduction).to(device) 
    elif task == "binary":
        if loss_function == 'default':
            criterion = nn.BCELoss(reduction=reduction).to(device)
        elif loss_function == "GCE": #generalized crossentropy, use q=0.2
            criterion = GCELoss(reduction=reduction, q=0.2).to(device)
            print('Using GCELoss, q=0.2')
        else:
            raise ValueError('unrecognized loss_function setting for binary task')
    elif task == "hinge":
        criterion = nn.HingeEmbeddingLoss(reduction=reduction).to(device)
    else:
        raise Exception("Unsupported task")
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        sys.stdout.flush()
        model = nn.DataParallel(model)
    model.to(device)

    epoch_history = {"train_loss":[], "test_loss":[], "train_acc":[], "test_acc":[], 'train_balanced_accuracy':[], 'train_negative_lr':[], 'train_positive_lr':[]}
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    start_time = time.time()
    data_load_time = 0
    transfer_time = 0
    training_time = 0

    best_test_acc = 0
    best_test_loss = np.inf #save best loss
    best_sample_labels = None
    best_sample_results = None
    best_epoch = 0
    for epoch in range(epochs):
        if eval: #eval should always be True, this is just a test
            print('Switch model to train')
            model.train()
        epoch_start = time.time()
        running_loss = 0
        running_samples = 0
        running_slow = 0
        running_fast = 0
        running_slow_correct = 0
        running_fast_correct = 0
        running_correct = 0
        data_load_start = time.time()
        for i, data in enumerate(trainloader):
            data_load_time += time.time() - data_load_start #data load start
            inputs, labels = data
            if labels.shape[0] == 0:
                print(f'Input/output is empty!!! For index {i} from trainloader. Skipping')
                sys.stdout.flush()
                continue
            if restrict_size: 
                if labels.shape[0] > trainloader.batch_size*pos_per_gene:
                    print(f"Size exceeded for batch {i} with length {labels.shape[0]}")
                    sys.stdout.flush()
                    labels = labels[:trainloader.batch_size*pos_per_gene]
                    print(f"Shortened batch to length {labels.shape[0]}")
                    sys.stdout.flush()
                    if type(inputs) is torch.Tensor:
                        inputs = inputs[:trainloader.batch_size*pos_per_gene]
                    elif len(inputs) == 2:
                        inputs = (inputs[0][:trainloader.batch_size*pos_per_gene], inputs[1][:trainloader.batch_size*pos_per_gene])
                    else:
                        raise Exception("Inputs is more than two tensors!")
            transfer_start = time.time()
            if type(inputs) is torch.Tensor:
                inputs = inputs.to(device) 
            elif len(inputs) == 2:
                inputs = (inputs[0].to(device), inputs[1].to(device))
            else:
                raise Exception("Inputs is more than two tensors!")
            labels = labels.to(device) 
            transfer_time += time.time() - transfer_start
            if task == "hinge":
                #for hinge, 0/1 labels should be -1/1, so multiply by 2, subtract 1
                labels = (labels*2) - 1
            training_start = time.time()
            optimizer.zero_grad()
           
            if trainloader.dataset.has_weights:
                if type(inputs) is torch.Tensor:
                    sample_weights = torch.flatten(inputs[:,-1]) #last column
                    inputs = inputs[:,:-1] #all but last column
                elif len(inputs) == 2:
                    sample_weights = torch.flatten(inputs[0][:,-1]) #last column
                    inputs[0] = inputs[0][:,:-1] #all but last column
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if trainloader.dataset.has_weights:
                loss = (loss.T*sample_weights).sum() 
            loss.backward()
            optimizer.step()
            training_time += time.time() - training_start

            if task=="binary":
                predicted = torch.gt(outputs, 0.5) #everything greater than 0.5 is fast
            elif task=="hinge":
                predicted = torch.gt(outputs, 0) #hinge classifies as 1/-1
            else:
                _, predicted = torch.max(outputs, 1) #take max val
            running_loss += loss.item()
            running_samples += labels.shape[0]
            if task == "binary":
                running_fast += (labels == 1).sum().item() 
                running_slow += (labels == 0).sum().item() 
                running_fast_correct += ((labels == 1) & (predicted == labels)).sum().item() 
                running_slow_correct += ((labels == 0) & (predicted == labels)).sum().item() 
            running_correct += (predicted == labels).sum().item()
            #loss is sum
            if len(trainloader) <= 20 or i % (len(trainloader)//20) == 0:
                sensitivity = running_fast_correct/running_fast
                specificity = running_slow_correct/running_slow
                ba = (sensitivity+specificity)/2
                print(f'Epoch {epoch}, {i}/{len(trainloader)} batches. Loss : {running_loss/running_samples}. Accuracy: {running_correct/running_samples}, Balanced accuracy: {ba}')
                print(f'    Time:{(time.time()-start_time)//60} minutes. Data Loading: {data_load_time/60} minutes, Training: {training_time/60} minutes, Transfer: {transfer_time/60}')
                if device.type=="cuda":
                    print('Memory Usage: Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1),
                           'GB. Cached:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
                sys.stdout.flush()
                #save
                torch.save(model.state_dict(), model_save_path)
            data_load_start = time.time()
        #validation
        print(f'\nEpoch {epoch} complete in {(time.time()-epoch_start)//60} minutes')
        print(f'Samples in epoch: {running_samples}, slow samples in epoch: {running_slow}')
        sys.stdout.flush()
        if testloader.dataset.include_index == True:
            test_loss, test_acc, sample_labels, sample_results = test_network(model, testloader, device, task=task, 
                                    pos_per_gene=pos_per_gene, eval=eval, loss_function=loss_function,
                                    restrict_size=restrict_size, sample_label_rows=testloader.dataset.num_index_rows)
            test_metrics = get_metrics(torch.flatten(sample_labels[:,-1]), torch.flatten(sample_results))
            print('Metrics are:', test_metrics)
            print("Saving sample results to ", sample_results_save_path)
            print("Saving sample labels to ", sample_labels_save_path)
            torch.save(sample_labels, sample_labels_save_path)
            torch.save(sample_results, sample_results_save_path)
            if test_loss < best_test_loss:
                print(f"Loss {test_loss} is better than previous best {best_test_loss}, overwriting best.")
                best_test_loss = test_loss
                best_test_acc = test_acc
                if 'balanced_accuracy' in test_metrics:
                    best_test_acc = test_metrics['balanced_accuracy'] #use balanced accuracy instead of accuracy - easier to understand
                best_epoch = epoch
                best_sample_labels = sample_labels
                best_sample_results = sample_results
                torch.save(model.state_dict(), best_model_save_path)
                torch.save(best_sample_labels, best_sample_labels_save_path)
                torch.save(best_sample_results, best_sample_results_save_path)
            test_output = (test_loss, test_acc, sample_labels, sample_results)
            if return_best:
                test_output =  (best_test_loss, best_test_acc, best_sample_labels, best_sample_results)
        else:
            test_loss, test_acc = test_network(model, testloader, device, task=task, loss_function=loss_function, restrict_size=restrict_size, eval=eval)
            test_output = (test_loss, test_acc)
        epoch_history['train_loss'].append(running_loss/running_samples)

        sensitivity = running_fast_correct/running_fast
        specificity = running_slow_correct/running_slow
        ba = (sensitivity+specificity)/2
        epoch_history['train_balanced_accuracy'].append(ba)
        if specificity != 0:
            epoch_history['train_negative_lr'].append((1-sensitivity)/specificity)
        else:
            epoch_history['train_negative_lr'].append(np.nan)
        if specificity != 1:
            epoch_history['train_positive_lr'].append(sensitivity/(1-specificity))
        else:
            epoch_history['train_positive_lr'].append(np.nan)


        epoch_history['test_loss'].append(test_loss)
        epoch_history['train_acc'].append(running_correct/running_samples)
        epoch_history['test_acc'].append(test_acc)
        if testloader.dataset.include_index == True:
            for metric in ['balanced_accuracy', 'negative_lr', 'positive_lr']:
                if metric in test_metrics:
                    if "test_"+metric not in epoch_history:
                        epoch_history['test_'+metric] = []
                    epoch_history['test_'+metric].append(test_metrics[metric])
        torch.save(model.state_dict(), model_save_path)
        #save model for the epoch
        epoch_model_save_path = epoch_models_save_dir + 'epoch_'+str(epoch)+'_model.pt'
        torch.save(model.state_dict(), epoch_model_save_path)
        with open(history_save_path, 'w') as f:
            json.dump(epoch_history, f)
        print()
    print(f"Training complete! Best epoch was {best_epoch}")
    sys.stdout.flush()
    torch.save(model.state_dict(), model_save_path)
    if return_test:
        return test_output
    

def load_model_from(model_save, device='cpu', params = None, pca=False, model_size='medium', input_1d_shape=None):
    if params is None:
        if model_size=='small':
            params = {'dropout_rate':0.5, 'layers_1d_size':64, 'layers_combined':2, 'layers_combined_size':32, 'batch_norm':False}
        elif model_size=='medium': 
            params = {'dropout_rate':0.5, 'layers_1d_size':128, 'layers_combined':2, 'layers_combined_size':64, 'batch_norm':False}
        elif model_size=='big':
            params = {'dropout_rate':0.5, 'layers_1d_size':256, 'layers_combined':4, 'layers_combined_size':128, 'batch_norm':False}
        elif model_size=='tiny':
            params = {'dropout_rate':0.5, 'layers_1d':1, 'layers_1d_size':16, 'layers_combined':1, 'layers_combined_size':8, 'batch_norm':False}
        else:
            raise ValueError()
    if input_1d_shape is None:
        if pca:
            input_1d_shape = (128,)
        else:
            input_1d_shape = (1280,)
    model = MultiInputNet(include_2d=False, input_1d_shape=input_1d_shape, model_parameters=params)
    model.load_state_dict(torch.load(model_save, map_location=device))
    model.eval()
    return model


def get_metrics(labels, results, th=0.5):
    #return a dictionary of metrics
    #expects tensors as inputs
    labels = labels.cpu().numpy()
    results = results.cpu().numpy() 
    tp = np.sum((labels == 1) & (results >= th))
    tn = np.sum((labels == 0) & (results < th))
    fp = np.sum((labels == 0) & (results >= th))
    fn = np.sum((labels == 1) & (results < th))
    tpr = tp/(tp+fn)
    tnr = tn/(tn+fp)
    fpr = 1-tnr
    fnr = 1-tpr
    positive_lr = tpr/fpr
    negative_lr = fnr/tnr
    accuracy = (tp+tn)/(tp+tn+fp+fn)
    balanced_accuracy = (tpr + tnr)/2
    return {'accuracy':accuracy, 'balanced_accuracy':balanced_accuracy, 
            'positive_lr':positive_lr, 'negative_lr':negative_lr, 
            'sensitivity':tpr, 'specificity':tnr}

def test_network(model, testloader, device, task="binary", loss_function='default', restrict_size=False, pos_per_gene=100, sample_label_rows=0, verbose=True, eval=True):
    start_time = time.time()
    if eval: #eval should always be True, this is to test
        print('switch model to eval')
        model.eval()
    loss = 0
    num_samples = 0
    correct = 0
    num_slow = 0

    reduction = "sum"
    if testloader.dataset.has_weights:
        reduction = 'none' #will multiply by weights later to reduce

    if task == "classifier":
        criterion = nn.CrossEntropyLoss(reduction=reduction).to(device) #set to sum for batch, rather than mean
    elif task == "binary":
        if loss_function == 'default':
            criterion = nn.BCELoss(reduction=reduction).to(device)
        elif loss_function == "GCE": #generalized crossentropy, use q=0.2
            criterion = GCELoss(reduction=reduction, q=0.2).to(device)
            print('Using GCELoss, q=0.2')
        else:
            raise ValueError('unrecognized loss_function setting for binary task')
    elif task == "hinge":
        criterion = nn.HingeEmbeddingLoss(reduction=reduction).to(device)
    else:
        raise Exception("Unsupported task")
    if sample_label_rows != 0:
        sample_labels = torch.empty((0,sample_label_rows)).to(device) 
        _, labels = testloader.dataset[0]
        if len(labels.shape) == 1:
            sample_results = torch.empty((0,)).to(device)
        elif len(labels.shape) == 2:
            sample_results = torch.empty((0, labels.shape[1])).to(device)
        else:
            raise ValueError('something is wrong with shape of labels from testloader')
    with torch.no_grad():
        for i, data in enumerate(testloader):
            inputs, labels = data
            if labels.shape[0] == 0:
                print(f'Input/output is empty!!! For index {i} from testloader. Skipping')
                sys.stdout.flush()
                continue
            if restrict_size: 
                if labels.shape[0] > testloader.batch_size*pos_per_gene:
                    print(f"Size exceeded for batch {i} with length {labels.shape[0]}")
                    sys.stdout.flush()
                    labels = labels[:testloader.batch_size*pos_per_gene]
                    print(f"Shortened batch to length {labels.shape[0]}")
                    sys.stdout.flush()
                    if type(inputs) is torch.Tensor:
                        inputs = inputs[:testloader.batch_size*pos_per_gene]
                    elif len(inputs) == 2:
                        inputs = (inputs[0][:testloader.batch_size*pos_per_gene], inputs[1][:testloader.batch_size*pos_per_gene])
                    else:
                        raise Exception("Inputs is more than two tensors!")
            if type(inputs) is torch.Tensor:
                inputs = inputs.to(device) 
            elif len(inputs) == 2:
                inputs = (inputs[0].to(device), inputs[1].to(device))
            else:
                raise Exception("Inputs is more than two tensors!")
            if sample_label_rows != 0:
                if type(inputs) is torch.Tensor: 
                    sample_labels = torch.concat((sample_labels, inputs[:,:sample_label_rows])) 
                    inputs = inputs[:,sample_label_rows:]
                elif len(inputs) == 2:
                    if verbose and not type(inputs) is torch.Tensor:
                        print('SIZES labels inputs')
                        print(labels.size(), inputs[0].size(), inputs[1].size())
                        print('Types')
                        print(labels.type(), inputs[0].type(), inputs[1].type())
                        sys.stdout.flush()
                    sample_labels = torch.concat((sample_labels, inputs[0][:,:sample_label_rows]))
                    inputs = (inputs[0][:,sample_label_rows:], inputs[1])
            
            if testloader.dataset.has_weights:
                if type(inputs) is torch.Tensor:
                    sample_weights = torch.flatten(inputs[:,-1]) #last column
                    inputs = inputs[:,:-1] #all but last column
                elif len(inputs) == 2:
                    sample_weights = torch.flatten(inputs[0][:,-1]) #last column
                    inputs[0] = inputs[0][:,:-1] #all but last column
           
            labels = labels.to(device)
            if task == "hinge":
                #for hinge, 0/1 labels should be -1/1, so *2 - 1
                labels = (labels*2) - 1
            if verbose:
                if not type(inputs) is torch.Tensor:
                    print("Inputs device", inputs[0].device, inputs[1].device)
                print("Model device", next(model.parameters()).device, "inputs device", inputs.device)
            outputs = model(inputs) #these are logits
            if sample_label_rows != 0:
                sample_results = torch.concat((sample_results, outputs))
            if verbose:
                print("task is", task)
                print("outputs size and type", outputs.size(), outputs.type(), outputs.device)
                print("labels size and type", labels.size(), labels.type(), labels.device)
                if sample_label_rows != 0:
                    print("sample labels", sample_labels.size(), sample_labels.type(), sample_labels.device)
                if not type(inputs) is torch.Tensor:
                    print("inputs", inputs[0].size(), inputs[0].type, inputs[1].size, inputs[1].type)
                print("Example of outputs - first 20")
                print(outputs[0:20,:])
                print("Example of labels - first 20")
                print(labels[0:20])

            if testloader.dataset.has_weights:
                loss += ((criterion(outputs, labels)).T*sample_weights).sum().item()
            else:
                loss += criterion(outputs, labels).item()

            if task=="binary":
                predicted = torch.gt(outputs.data, 0.5) #everything greater than 0.5 is fast
            elif task == "hinge":
                predicted = torch.gt(outputs.data, 0) #everything greater than 0 is fast
            else:
                _, predicted = torch.max(outputs.data, 1) #take max val
            correct += (predicted == labels).sum().item()
            if verbose:
                print("Example of predicted - first 20")
                print(predicted[0:20])
                print("Example of correct - first 20")
                print((predicted == labels)[0:20])
            num_samples += labels.shape[0]
            if task == 'binary':
                num_slow += (labels == 0).sum().item()
            else:
                #0 for slow, 1 for fast, 2 for neither
                num_slow += (labels == 0).sum().item()
            verbose = False #so you only do it for first
    print(f'Test loss: {loss/num_samples}. Test accuracy: {correct/num_samples}. Time to test:{(time.time() - start_time)//60} minutes.')
    print(f'Test samples: {num_samples}, slow samples {num_slow}')
    sys.stdout.flush()
    if sample_label_rows != 0:
        return loss/num_samples, correct/num_samples, sample_labels, sample_results
    return loss/num_samples, correct/num_samples

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#combining preloading and chunk-making
#preload_data saves f0_c0_labels.pt, f0_c1_labels.pt, f1_c0_labels.pt, f1_labels_c1.pt ... so on. 
#where f_ is fold number, c_ is chunk number in the fold
#ChunkLoader loads chunks 

#this function currently works only for 1D datasets
#lazy = True will use saved dataset if one exists (and will assume dataset exists if save_dir directory exists)
#lazy=False will erase any existing dataset
def preload_data(dataloader, save_dir, gb_target=5, return_loader=False, batch_size=128, normalize=False, weight_by_aa_not_gene=False, lazy=False):
    #want to save data as chunks
    #aim to have about 10G chunks (ends up being closer to 5G)
    #this is around 1 million positions * 1280 features * 4 bytes [for float32]
    logistic_regression = True
    print('\npreloading data, splitting into chunks')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        if lazy:
            assert(not return_loader)
            print('\nDirectory already exists! no need to preload')
            return
    sys.stdout.flush()
    st = time.time()
    print('make loader')
    sys.stdout.flush()
    loader = iter(dataloader)
    print('Request next from loader')
    sys.stdout.flush()
    input_1d_shape = next(loader)[0].shape
    labels_shape = next(loader)[1].shape
    pos_per_chunk = gb_target * (10**9) // (input_1d_shape[1] * 4)
    print(f'{input_1d_shape[1]} features per position, {pos_per_chunk} positions per chunk')
    sys.stdout.flush()
    chunk_inputs = torch.empty((0, input_1d_shape[1]))
    chunk_labels = torch.empty((0, labels_shape[1]))
    c = 0
    for i, data in enumerate(dataloader):
        inputs, labels = data
        chunk_inputs = torch.concat((chunk_inputs, inputs))
        chunk_labels = torch.concat((chunk_labels, labels))
        if chunk_labels.shape[0] >= pos_per_chunk or i == len(dataloader) - 1: #time to save chunk
            print(f'Saving chunk {c} of size {chunk_inputs.shape} to {save_dir}, time: {(time.time()-st)/60} minutes')
            chunk_std, chunk_mean = torch.std_mean(chunk_inputs[:,dataloader.dataset.num_index_rows:], dim=0) #but ignore indices
            #these values are printed to check that mean, variance across chunks is roughly the same
            #this should be taken into consideration if you want to normalize data by chunk
            print(f'    Means of features in chunk range from {min(chunk_mean)} to {max(chunk_mean)} (mean {torch.mean(chunk_mean)})')
            print(f'    Std ranges from {min(chunk_std)} to {max(chunk_std)} (mean {torch.mean(chunk_std)})')
            sys.stdout.flush()
            if logistic_regression: #run logistic regression on chunk as a comparison
                try:
                    st_lr = time.time()
                    #scale values
                    #if std is 0, want X to be 0
                    #so change std where std is 0 to be equal to 1 [input wil stay at 0 either way] this affects normalize, too
                    chunk_std[chunk_std == 0.0] = 1.0
                    X = torch.divide(torch.subtract(chunk_inputs[:,dataloader.dataset.num_index_rows:],chunk_mean),chunk_std)
                    y = chunk_labels.reshape((-1,))
                    lr = LogisticRegression(max_iter=100000).fit(X,y)
                    print(f'Ran logistic regression on chunk - took {((1000*(time.time() - st_lr))//60)/1000} minutes')
                    #score is accuracy
                    print(f'Accuracy is {lr.score(X, y.numpy())}')
                    sys.stdout.flush()
                except Exception as e:
                    print('failed to run logistic regression')
                    print(e)
                    sys.stdout.flush()
            if normalize:
                print('subtracting mean and dividing by stdev')
                chunk_inputs = torch.cat((chunk_inputs[:,:dataloader.dataset.num_index_rows], 
                                torch.divide(torch.subtract(chunk_inputs[:,dataloader.dataset.num_index_rows:],chunk_mean), 
                                             chunk_std)), dim=1)
                chunk_std, chunk_mean = torch.std_mean(chunk_inputs[:,dataloader.dataset.num_index_rows:], dim=0) #but ignore indices
                #mean and stdev should now be equal to 1
                print(f'    Means of features in chunk range from {min(chunk_mean)} to {max(chunk_mean)} (mean {torch.mean(chunk_mean)})')
                print(f'    Std ranges from {min(chunk_std)} to {max(chunk_std)} (mean {torch.mean(chunk_std)})')

            if weight_by_aa_not_gene: #weight only by amino acid, not by gene
                print('Adding weights')
                assert(dataloader.dataset.num_index_rows == 5), "Cannot set weights by aa - no codon info?"
                assert(dataloader.dataset.has_weights == False), "Dataset already has weights"
                assert(dataloader.dataset.balance_by_aa == False), "Dataset already balanced"
                #dataset index rows are expected to be gene_id, position, species, codon, label --> codon is 3
                #need a dictionary of codons to amino acids
                with open(dataloader.dataset.hdf5_fname.split('.')[0]+'_meta.json', 'r') as f:
                    meta = json.load(f)
                    codon_to_int_label = meta['codon']
                    aa_to_int_label = meta['aa']
                #"MSK" and "PAD" are also potential "codon" tokens
                c_to_aa =   {'TGT': 'C', 'TGC': 'C', 'GAT': 'D', 'GAC': 'D', 
                                'TCT': 'S', 'TCG': 'S', 'TCA': 'S', 'TCC': 'S', 'AGC': 'S', 'AGT': 'S', 
                                'CAA': 'Q', 'CAG': 'Q', 'ATG': 'M', 'AAC': 'N', 'AAT': 'N', 
                                'CCT': 'P', 'CCG': 'P', 'CCA': 'P', 'CCC': 'P', 'AAG': 'K', 'AAA': 'K', 
                                'ACC': 'T', 'ACA': 'T', 'ACG': 'T', 'ACT': 'T', 
                                'TTT': 'F', 'TTC': 'F', 'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A', 
                                'GGT': 'G', 'GGG': 'G', 'GGA': 'G', 'GGC': 'G', 'ATC': 'I', 'ATA': 'I', 'ATT': 'I', 
                                'TTA': 'L', 'TTG': 'L', 'CTC': 'L', 'CTT': 'L', 'CTG': 'L', 'CTA': 'L', 'CAT': 'H', 'CAC': 'H', 
                                'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R', 'AGG': 'R', 'AGA': 'R', 
                                'TGG': 'W', 'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V', 
                                'GAG': 'E', 'GAA': 'E', 'TAT': 'Y', 'TAC': 'Y', 
                                'TAG': 'X', 'TAA':'X', 'TGA':'X', 'MSK':'X', 'PAD':'X'} #stop as X, any invalid value is also set to X
                c_int_label_to_aa_int_label = {codon_to_int_label[c]:aa_to_int_label[c_to_aa[c]] for c in codon_to_int_label} #get aa for codon
                aa_encoder = np.vectorize(lambda x: c_int_label_to_aa_int_label.get(x, 'X')) #return X for any invalid value
                chunk_aa = torch.from_numpy(aa_encoder(chunk_inputs[:,3]).astype(int))
                chunk_label = chunk_inputs[:,4] #label
                #chunk_aa and chunk_label should both be tensors
                chunk_weights = torch.zeros(chunk_aa.shape) #same shape as aa labels
                aa_list = torch.unique(chunk_aa)
                for aa in aa_list:
                    fast_idx = ((chunk_aa == aa) & (chunk_label == 1)) #boolean index
                    slow_idx = ((chunk_aa == aa) & (chunk_label == 0))
                    if torch.sum(fast_idx) == 0 or torch.sum(slow_idx) == 0:
                        slow_weight = 0
                        fast_weight = 0
                    else:
                        slow_weight = (torch.sum(fast_idx)+torch.sum(slow_idx))/2/torch.sum(slow_idx) 
                        fast_weight = (torch.sum(fast_idx)+torch.sum(slow_idx))/2/torch.sum(fast_idx)
                    chunk_weights[slow_idx] = slow_weight
                    chunk_weights[fast_idx] = fast_weight
                #set weights so that for each aa sum(aa_slow) = sum(aa_fast) = len(aa)/2
                print("chunk inputs shape", chunk_inputs.shape, "chunk weights shape", chunk_weights.shape)
                print("Example chunk_aa ", chunk_aa[0:20])
                print("Example chunk_label ", chunk_label[0:20])
                print("Example chunk weight", chunk_weights[0:20])
                chunk_inputs = torch.cat((chunk_inputs, chunk_weights.reshape((-1,1))), dim=1) #concat weights
            sys.stdout.flush()
            print()
            torch.save(chunk_inputs, save_dir+'c'+str(c)+'_inputs'+'.pt')
            torch.save(chunk_labels, save_dir+'c'+str(c)+'_labels'+'.pt')
            chunk_inputs = torch.empty((0, input_1d_shape[1]))
            chunk_labels = torch.empty((0, labels_shape[1]))
            if  i == len(dataloader) - 1:
                print(f'saved {c+1} chunks to {save_dir}, took {(time.time()-st)//60} minutes')
            c += 1 
    if return_loader:
        assert(weight_by_aa_not_gene == False), "did not account for weight_by_aa_not_gene - need to set dataset to have weights"
        #batch size is specified in number of positions (not in the number of genes, as elsewhere)
        print(f'making chunkloader with batch size {batch_size}')
        return ChunkLoader(save_dir, batch_size=batch_size, shuffle=True, dataset = dataloader.dataset)

#specifying has_weights overrides dataset.has_weights
#note that batchsize means number of positions, not genes
#if test is True, load only excluded_fold, otherwise load all fold except excluded_fold
#if always_exclude is True, test_species is excluded from both training and validation fold
class ChunkLoader():
    def __init__(self, save_dir, batch_size=128, dataset=None, shuffle='gene_order', has_weights = None,
                  folds=True, excluded_fold=0, test=False, num_index_rows=None, test_species=None, always_exclude=False):
        #options for shuffle:
        #shuffle all positions #full
        #shuffle genes and positions within genes, #gene_and_pos_order
        #shuffle genes but not positions within genes #gene_order
        #shuffle batches #batch_order
        #shuffle nothing #none
        #shuffle chunks #chunk_order
        #shuffle positions in genes, but not genes #pos_order
        assert(shuffle in ['gene_order','batch_order','chunk_order','gene_and_pos_order', 'pos_order', 'none','full'])
        self.batch_size = batch_size
        self.save_dir = save_dir
        self.dataset = dataset #in case we want to reference the dataset it was made from
        if not num_index_rows is None:
            assert(self.dataset is None), "Can't specify both num index rows and dataset"
            self.num_index_rows = num_index_rows
        elif not self.dataset is None:
            self.num_index_rows = self.dataset.num_index_rows
        else:
            self.num_index_rows = 0
        self.has_weights = False
        if not self.dataset is None:
            self.has_weights = self.dataset.has_weights
        if not has_weights is None:
            #specifying has_weights overrides dataset.has_weights 
            self.has_weights = has_weights
    

        self.test = test
        self.chunk_sizes = {}
        for fname in os.listdir(save_dir):
            if fname[-9:] == 'labels.pt':
                if folds and test and fname.split('_')[0] == 'f'+str(excluded_fold):
                    assert(fname[-10:] == '_labels.pt')
                    chunk_name = fname[:-10]
                    self.chunk_sizes[chunk_name] = torch.load(save_dir+fname).shape[0]
                elif folds and (not test) and fname.split('_')[0] != 'f'+str(excluded_fold):
                    assert(fname[-10:] == '_labels.pt')
                    chunk_name = fname[:-10] 
                    self.chunk_sizes[chunk_name] = torch.load(save_dir+fname).shape[0]
                    #for training, these sizes will be incorrect, because we haven't removed s_cer sequences yet
                    #(size will be fixed once chunk is loaded)

        self.n_chunks = len(self.chunk_sizes)
        self.chunk_names = sorted([x for x in self.chunk_sizes]) 
        print(f'Making chunkloader. Excluded fold is {excluded_fold}, test is {test}, Chunk names are {self.chunk_names}, batch size is {batch_size}')
        self.chunk_idx = 0 #idx of current_chunk IN chunk order [note chunk order can be shuffled]
        self.labels = None #this is where the chunk we are currently loading from will be stored
        self.inputs = None
        self.pos_in_chunk = 0 #idx of current pos in chunk?
        self.shuffle = shuffle
        if self.shuffle in ["gene_order", "gene_and_pos_order", 'pos_order']:
            assert(self.num_index_rows > 0), "Gene or pos shuffle requires index rows in dataset"
        print(f'ChunkLoader shuffle type is {shuffle}')
        self.test_species = test_species #the species to remove from the training set (default: 13 -- S. cerevisiae)
        self.check_sanity = True #can set to False to spend a little less time
        self.always_exclude = always_exclude

    def __len__(self):
        return int(sum([np.ceil(self.chunk_sizes[x]/self.batch_size) for x in self.chunk_sizes]))
        #NOTE that during training this might be incorrect, because we haven't removed s_cer yet
                    #(size will be fixed once all chunks are loaded)
        

    def __iter__(self):
        if self.shuffle != "none":
            np.random.shuffle(self.chunk_names)
            print(f"Chunk order is now {self.chunk_names}")
        self.set_to_chunk(0)
        return self
    
    
    def set_to_chunk(self, chunk_idx):
        self.chunk_idx = chunk_idx
        prefix = str(self.chunk_names[self.chunk_idx])
        st = time.time()
        print(f'Setting chunk in chunk_loader to {prefix}')
        self.labels = torch.load(self.save_dir + prefix+'_labels.pt')
        self.inputs = torch.load(self.save_dir + prefix+'_inputs.pt') 
        if ((not self.test) or self.always_exclude) and self.num_index_rows >= 3 and not (self.test_species is None):
            #if training set, want to remove index rows 
            #assumes index is gene pos species codon label
            size_before = self.inputs.shape[0]
            self.inputs = self.inputs[self.inputs[:,2] != self.test_species]
            print(f"After removing species {self.test_species}, input size changed from {size_before} to {self.inputs.shape[0]}")
            #chunk lengths are now incorrect - need to fix this:
            self.chunk_sizes[self.chunk_names[self.chunk_idx]] = self.inputs.shape[0]

        self.batches_in_chunk = np.ceil(self.chunk_sizes[self.chunk_names[self.chunk_idx]]/self.batch_size)
        if self.shuffle == "full":
            shuff_t = time.time()
            self.indices = np.random.permutation(self.chunk_sizes[self.chunk_names[self.chunk_idx]]) 
            print(f'Shuffling indices took {(time.time()-shuff_t)/60} minutes')
        #note that positions in the dataset do not necesarily start off as being in order
        #for balance by aa, the order is  fast_idx_A, slow_idx_A, fast_idx_B, slow_idx_B .... etc. etc for each aa
        elif self.shuffle == 'gene_order' or self.shuffle == 'gene_and_pos_order' or self.shuffle == 'pos_order':
            assert(self.num_index_rows > 0), "unexpected number of index rows"
            shuff_t = time.time()
            #we don't change inputs, rather, shuffle indices
            self.indices = np.arange(self.chunk_sizes[self.chunk_names[self.chunk_idx]])
            if self.shuffle == 'gene_and_pos_order' or self.shuffle == 'pos_order':
                np.random.shuffle(self.indices)
            #start indices as permuted so pos in genes not in order
            gene_vals = np.unique(self.inputs[:,0])
            if self.shuffle == 'pos_order':
                #permutation dict does not change values, genes will be in sorted order
                #NOTE if original order of genes was not sorted, this will still have changed order of genes
                permutation_dict = dict(zip(gene_vals, gene_vals))
            else:
                permutation_dict = dict(zip(gene_vals, np.random.permutation(gene_vals)))
            #change the labels of the genes, so that when sorted they go in a random order
            self.indices = self.indices[np.argsort(np.vectorize(permutation_dict.get)(self.inputs[self.indices,0]), kind='stable')]
            #sorted, the genes are in a random order, but all datapoints in the gene are together
            #positions in a gene in a random order
            #NOTE: this means that each batch can get more than one gene, but generally contains one gene 
            #eg: AAAA|AABB|BBCC|CCCC|CCCC|CDDD|DDDD
            #if batch size is very large, batches will generally contain more than one gene
            #NOTE: Note that here indices are a numpy on cpu, while input is a tensor on gpu
            print(f'Shuffling genes/positions took {(time.time()-shuff_t)/60} minutes')
        elif self.shuffle == 'batch_order':
            #for this, drop the last incomplete batch, floor instead of ceil
            shuff_t = time.time()
            self.batches_in_chunk = int(np.floor(self.chunk_sizes[self.chunk_names[self.chunk_idx]]/self.batch_size))
            batch_order = np.random.permutation(self.batches_in_chunk)
            self.indices = np.repeat(batch_order, self.batch_size)*self.batch_size + np.tile(np.arange(self.batch_size), self.batches_in_chunk)
            print(f'Shuffling batches took {(time.time()-shuff_t)/60} minutes')
        elif self.shuffle == 'none' or self.shuffle == 'chunk_order':
            self.indices = np.arange(self.chunk_sizes[self.chunk_names[self.chunk_idx]])
        else:
            raise ValueError('Shuffle is not set to an allowed value')
        if self.check_sanity:
            print(f"Checking sanity for shuffle ({self.shuffle})")
            if self.shuffle == 'gene_order' or self.shuffle == 'gene_and_pos_order' or self.shuffle == 'pos_order':
                genes = self.inputs[self.indices, 0]
                num_genes = len(np.unique(genes))
                #genes are together - gene value changes num_genes - 1 times
                assert(torch.sum(genes[1:] != genes[:-1]) == num_genes - 1)
                if self.shuffle == 'gene_order':
                    #indices order changes as many times as there are genes - 1 (positions are not shuffled)
                    #but it could be less if two genes end up in the same order after shuffling
                    print(f"Number of changes in gene order: {np.sum(self.indices[1:]-self.indices[:-1] != 1)}, {num_genes} genes")
                    if (np.sum(self.indices[1:]-self.indices[:-1] != 1) > num_genes - 1):
                        print('\n Failed Indices:')
                        print(self.indices[0:1000])
                        print('\n Failed inputs:')
                        print(self.inputs[self.indices[0:1000],0:3])
                    assert(np.sum(self.indices[1:]-self.indices[:-1] != 1) <= num_genes - 1)
                if self.shuffle == 'pos_order':
                    #if only positions are shuffled, genes are in ascending order
                    assert(torch.all(genes[1:] >= genes[:-1]))
            if self.shuffle == 'batch_order':
                #indices order changes as many times as there are batches - 1
                #but it could be less if two batches end up in the same order after shuffling
                print(f"Number of changes in batch order: {np.sum(self.indices[1:]-self.indices[:-1] != 1)}, {self.batches_in_chunk} batches")
                assert(np.sum(self.indices[1:]-self.indices[:-1] != 1) <= self.batches_in_chunk - 1)
            print("Shuffle seems correct.")
        self.batch_num = 0

        #Remove index rows from training set.
        if (not self.test) and self.num_index_rows >= 3:
            self.inputs = self.inputs[:, self.num_index_rows:]
        
        print(f'Chunk input shape is {self.inputs.shape}')
        print(f'Setting chunk took {(time.time()-st)/60} minutes total')

    def __next__(self):
        if self.batch_num > self.batches_in_chunk:
            new_chunk_idx = self.chunk_idx + 1
            if new_chunk_idx >= self.n_chunks:
                raise StopIteration 
            else:
                self.set_to_chunk(new_chunk_idx)
        indices = self.indices[self.batch_num*self.batch_size:(self.batch_num+1)*self.batch_size]
        labels = self.labels[indices]
        inputs = self.inputs[indices]
        self.batch_num += 1
        return inputs, labels
    
def combine_folds(results_dir, fix_gene_labels=False):
    db_folds = pd.read_csv(results_dir+'db_folds.csv') #now indexed by number
    #which is the index we want to use
    num_folds = max(db_folds.fold)+1
    indices_list = []
    results_list = []
    for i in range(num_folds):
        print('Fold', i)
        fold_name = 'fold'+str(i)+'_'  
        results = torch.load(results_dir+fold_name+'best_sample_results.pt', map_location=torch.device('cpu'))[:,0].numpy()
        indices = torch.load(results_dir+fold_name+'best_sample_indices.pt', map_location=torch.device('cpu')).numpy()
        if fix_gene_labels:
            indices[:,0] = db_folds[db_folds.fold==i].index[indices[:,0].astype(int)]
        indices_list.append(indices)
        results_list.append(results)
    indices = np.concatenate(indices_list)
    results = np.concatenate(results_list)
    return indices, results
    
#modified Subset so that it would transfer attributes like
#include_index and n_sample_labels from dataset
class MySubset(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.include_index = self.dataset.include_index
        self.has_weights = self.dataset.has_weights
        self.balance_by_aa = self.dataset.balance_by_aa
        self.hdf5_fname = self.dataset.hdf5_fname 
        if self.dataset.include_index:
            self.num_index_rows = self.dataset.num_index_rows

#lazy: will attempt to use existing preloaded data if the directory is there
def run_kfold(dataset_params, train_params, db_clusters, lazy=False):
    print("\nDataset params:", dataset_params)
    print("\nTrain params:",train_params)

    #unravel dataset_params
    features = dataset_params['features']
    pos_name = dataset_params['pos_name']
    num_species = dataset_params['num_species']
    esm_name =  dataset_params['esm_name']
    apply_pca =  dataset_params['apply_pca']
    binary = dataset_params['binary']
    avg_esm = dataset_params['avg_esm']
    label_mode = dataset_params['label_mode']
    filter_by_aa = dataset_params['filter_by_aa']
    change_species = dataset_params['change_species']
    weight_by_aa = dataset_params['weight_by_aa']
    balance_type = dataset_params['balance_type']

    test_species = None #species to exclude from training folds - done in ChunkLoader
    #NOTE: currently, S. cer is simply dropped entirely from db_clusters
    #test_species is specified by change_species
    if len(change_species) != 0:
        assert(len(change_species) == 1)
        test_species = list(change_species.keys())[0]

    #unravel train_params
    fold = train_params['fold']
    params = train_params['params']
    epochs = train_params['epochs'] 
    save_dir = train_params['save_dir'] #changes
    restrict_size = train_params['restrict_size']
    batch_size = train_params['batch_size']
    pos_per_gene = train_params['pos_per_gene']
    task = train_params['task']
    loss_function = train_params['loss_function']
    linear = train_params['linear']
    preload = train_params['preload']
    preload_batch_size = train_params['preload_batch_size']
    model_name = train_params['model_name']
    short = train_params['short']
    shuffle = train_params['shuffle']
    eval = train_params['eval']
    #shuffle is a string, telling chunkloader how to shuffle

    if train_params['weight_by_aa_not_gene']:
        assert(train_params['preload']), "WEIGHTING BY AA NOT GENE REQUIRES PRELOADING"

    st = time.time()

    #make save_dir - for saving new db_clusters (has folds info)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    genes = db_clusters.index
    if short:
        genes = np.random.choice(genes, 1000, replace=False)
        epochs = 5
        print("Shorten! Length genes:", len(genes), 'epochs:', epochs) #for testing purposes
        sys.stdout.flush()
        db_clusters = db_clusters.loc[genes].copy()

    if 'fold' not in db_clusters.columns:
        if os.path.exists(train_params['save_dir'] + 'db_folds.csv') and lazy:
            print('Used existing db_folds.csv')
            db_clusters = pd.read_csv(train_params['save_dir'] + 'db_folds.csv', index_col=0)
            fold_series = db_clusters.fold
            n_folds = len(fold_series.unique())
            assert(n_folds == max(fold_series)+1)
        else:
            cluster_reps = np.array(db_clusters.cluster_rep.unique())
            np.random.shuffle(cluster_reps)
            cluster_reps = pd.DataFrame(index=cluster_reps)
            n_folds = 5
            cluster_reps['fold'] = [0]*(len(cluster_reps)%n_folds) + [f for f in range(n_folds) for i in range(len(cluster_reps)//n_folds)]
            db_clusters['fold'] = db_clusters.cluster_rep.map(cluster_reps.fold)
            print('Saved new folds')
            db_clusters.to_csv(train_params['save_dir'] + 'db_folds.csv')
    else: #could specify preemptively so all runs use same split, I guess.
        fold_series = db_clusters.fold
        n_folds = len(fold_series.unique())
        print('Used existing folds in clusters_db')
        assert(n_folds == max(fold_series)+1)

    assert(features in FEATURE_SETS_1D), "only 1d for now"
    #need to check collate fn, etc. and include 2d if we allow it 
    include_2d = False
    collate_fn = collate_1d

    
    all_test_acc = []
    all_test_loss = []

    unmixed=False
    positions_in_order = True
    if shuffle=='none' or shuffle=='batch_order' or shuffle=='gene_order_by_aa' or shuffle=='gene_order_unmixed_aa':
        positions_in_order = False
    if shuffle == 'gene_order_by_aa':
        shuffle = 'gene_order' #organize by aa as before, but shuffle genes 
    if shuffle == 'gene_order_unmixed_aa':
        shuffle = 'gene_order'
        unmixed = True
    print(f'unmixed is {unmixed}, positions in order is {positions_in_order}, shuffle is {shuffle}')
    
    train_dataset = HDF5_dataset(HDF5_NAME, gene_list = genes, features=features,  pos_name=pos_name, positions_in_order=positions_in_order,
                                         num_species=num_species, esm_name=esm_name, apply_pca=apply_pca, unmixed=unmixed,
                                         binary=binary, avg_esm=avg_esm, shuffle=False, label_mode=label_mode, change_species=change_species, balance_type=balance_type,
                                         include_index=False, filter_by_aa=filter_by_aa, balance_by_aa=dataset_params["balance_by_aa"], weight_by_aa=weight_by_aa)
    
    test_dataset = HDF5_dataset(HDF5_NAME, gene_list = genes,features=features,pos_name=pos_name,  positions_in_order=positions_in_order,
                                         num_species=num_species, change_species=change_species, esm_name=esm_name, apply_pca=apply_pca, unmixed=unmixed,
                                         binary=binary, avg_esm=avg_esm, shuffle=False, label_mode=label_mode, balance_type=balance_type,
                                         include_index=True, filter_by_aa=filter_by_aa, balance_by_aa=dataset_params["balance_by_aa"], weight_by_aa=weight_by_aa)
    if preload:
        for fold_num in range(n_folds):
            fold_dataset = MySubset(test_dataset, np.where((db_clusters.fold == fold_num))[0])
            #chunkloader will remove test_species and index rows
            preload_save_dir = DATA_DIR + 'tmp/'+model_name+'/' 
            loader = DataLoader(fold_dataset, batch_size=batch_size, collate_fn=collate_fn, persistent_workers=True, num_workers=8, pin_memory=True)
            preload_data(loader, preload_save_dir+'f'+str(fold_num)+'_', return_loader=False, normalize=train_params['normalize'], 
                         weight_by_aa_not_gene=train_params['weight_by_aa_not_gene'], lazy=lazy)
        if train_params["weight_by_aa_not_gene"]:
            train_dataset.has_weights = True
            test_dataset.has_weights = True #train/test need to know there are weights
            print("Changed dataset to 'have_weights' - note that dataset WILL NOT generate weights - weights generated during preload") 

    if not fold:
        n_folds = 1    
    for fold_num in range(n_folds):
        print('\n\nRunning fold', fold_num, 'out of', n_folds)
        sys.stdout.flush()
        fold_save_dir = save_dir+'fold'+str(fold_num)+'_'
        if not fold:
            fold_save_dir = save_dir
        if preload:
            if shuffle == "none":
                print("Chunkloader shuffling turned off")
            trainloader = ChunkLoader(preload_save_dir, test=False, batch_size=preload_batch_size, shuffle=shuffle,
                                      dataset =test_dataset, test_species=test_species, #test_species and index rows will be excluded in chunkloader
                                      excluded_fold=fold_num)
            testloader = ChunkLoader(preload_save_dir, test=True, batch_size=preload_batch_size, shuffle='none', dataset = test_dataset, 
                                     test_species=test_species, excluded_fold=fold_num) 
            #if the desired behavior is to exclude test_species in validation folds, set always_exclude to True
        else:
            #it is preferred to use preload_data
            if 'species' in db_clusters.columns: 
                #TODO this should be specified elsewhwere, by test_species, not hardcoded
                #currently, use preload = True instead
                fold_train_dataset = MySubset(train_dataset, np.where((db_clusters.species != 's_cer') & (db_clusters.fold != fold_num))[0])
            else:
                fold_train_dataset = MySubset(train_dataset, np.where((db_clusters.fold != fold_num))[0])
            fold_test_dataset = MySubset(test_dataset, np.where((db_clusters.fold == fold_num))[0]) 
            n_workers = 4 
            trainloader = DataLoader(fold_train_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, persistent_workers=True, num_workers=n_workers, pin_memory=True)
            testloader = DataLoader(fold_test_dataset, batch_size=batch_size, collate_fn=collate_fn, persistent_workers=True, num_workers=n_workers, pin_memory=True)

        print("Train genes:", train_dataset.genes[0:5])
        X, y = train_dataset[0] 
        if type(X) is torch.Tensor:
            input_1d_shape = (X.shape[1],)
        elif len(X) == 2:
            input_1d_shape = (X[0].shape[1],)
        else:
            raise ValueError('X has more than two tensors')
        if train_dataset.weight_by_aa: #check if dataset will crate weights
            input_1d_shape = (input_1d_shape[0] - 1,) #if there are weights, they are the last column
        
        print('Input_1d shape is ', input_1d_shape)
        sys.stdout.flush()

        if not linear:
            model = MultiInputNet(include_2d=include_2d, input_1d_shape=input_1d_shape, model_parameters=params)
        else:
            model = Linear(input_1d_shape=input_1d_shape)
        print(f"Model has {count_parameters(model)} trainable parameters")
        print('Starting training')
        test_loss, test_acc = train_network(model, trainloader, testloader, task=task, epochs=epochs, 
                                                                           loss_function=loss_function, save_dir=fold_save_dir, 
                                                restrict_size=restrict_size, pos_per_gene=pos_per_gene, return_test=True, eval=eval)
        print(f'Finished training k_fold. Final test loss is {test_loss} and final test acc is {test_acc}')
        all_test_loss.append(test_loss)
        all_test_acc.append(test_acc)

    print('\n\nK-fold accuracies:', all_test_acc)
    print('Mean accuracy across folds:', np.mean(all_test_acc))
    print('K-fold losses:', all_test_loss)
    print('Mean loss across folds:', np.mean(all_test_loss))
    sys.stdout.flush()
    if fold:
        print('Combining fold results')
        sys.stdout.flush()
        indices, results = combine_folds(save_dir)
        torch.save(indices, save_dir+'best_sample_indices.pt')
        torch.save(results, save_dir + 'best_sample_results.pt')
    if preload:
        print(f'removing preloaded data')
        shutil.rmtree(preload_save_dir)

    print(f'finished fold run in {(time.time()-st)//60} minutes')
    sys.stdout.flush()
    
if __name__ == '__main__':
    o_st = time.time()

    model_name = 'test_model' #model name
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    record_name = DATA_DIR + 'test_model_training_record.txt' #file to print output to
    if len(sys.argv) > 2:
        record_name = sys.argv[2]

    save_name = DATA_DIR + model_name+'/' #directory to save model to
    if len(sys.argv) > 3:
        save_name = sys.argv[3]

    mode = "" #can specify extra parameters 
    # "short" to run only 1000 genes as a test
    # "unweighted" to not balance slow/fast by gene and amino acid
    if len(sys.argv) > 4:
        mode = sys.argv[4]


    with open(record_name, 'w') as stdout_file:
        with redirect_stdout(stdout_file):  
            print(os.path.dirname(sys.executable))
            print(torch.__version__)
            print(torch.__file__)
            print("Running", mode, ", model name is", model_name)
            print("Saving to", save_name)

            sys.stdout.flush()
            if 'cpu' in mode: #linear will be run on cpu, as well
                print('set device now returns cpu')
                set_device = lambda : torch.device('cpu')

            print('Dataset is transistive-clusters 70, jan24_many_yeast_db_70_cm1.csv')
            print('Clusters were generated using mmseqs cluster --cluster-mode 1 --min-seq-id 0.7 -c 0.8 -s 7.5')
            db = pd.read_csv(DATA_DIR+'jan24_many_yeast_db_70_cm1.csv', index_col=0)
            #remove s_cer from db - save for test set, dop not use in k-fold cross-validation
            db = db[db.species != 's_cer']

            params = {'dropout_rate':0.5, 'layers_1d_size':128, 'layers_combined':2, 'layers_combined_size':64, 'batchnorm':True} #default
            if 'big' in mode:
                params = {'dropout_rate':0.5, 'layers_1d_size':256, 'layers_combined':4, 'layers_combined_size':128}
            if 'tiny' in mode:
                params = {'dropout_rate':0.5, 'layers_1d':1, 'layers_1d_size':16, 'layers_combined':1, 'layers_combined_size':8}
            if 'batchnorm' in mode:
                params['batch_norm'] = True
            if 'nodropout' in mode:
                params['dropout_rate'] = 0

            balance_by_aa = True
            #if running unweighted
            if 'unweighted' in mode:
                balance_by_aa = False

            pos_name = 'all'
            dataset_params = {'pos_name':pos_name, 
                             'num_species':13, 
                             'features':'only_esm',
                             'esm_name':'esm', 
                             'apply_pca':False,
                              'binary':True, 
                              'avg_esm':False, 
                              'label_mode':'multi_er',
                              'filter_by_aa':["C","I","P","G", "R", "A", "S", "L"], 
                              'balance_by_aa':balance_by_aa, 
                              'change_species':{13:10},  #13 is s.cerevisiae. Current setup is to drop from db instead of change
                              'weight_by_aa':False,
                              'balance_type':'undersample'}
            
            train_params = {'params':params,
                            'epochs':15, 
                            'save_dir':save_name, 
                            'restrict_size':True,
                            'shuffle':'full',
                            'batch_size':128,
                            'task':'binary',
                            'loss_function':'default',
                            'linear':False,
                            'pos_per_gene':1000,
                            'preload':True,
                            'preload_batch_size':128,
                            'model_name':model_name,
                            'fold':True,
                            'short':('short' in mode),
                            'eval':True,
                            'normalize':False,
                            'weight_by_aa_not_gene':('weightAA' in mode)}

            print(f'Positions name is {pos_name}')

            run_kfold(dataset_params, train_params, db)

            print(f"Total runtime: {(time.time() - o_st)//60} minutes")
            sys.stdout.flush()
            