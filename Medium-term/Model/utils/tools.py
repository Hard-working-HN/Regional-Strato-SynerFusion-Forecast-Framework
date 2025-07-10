import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def adjust_learning_rate(optimizer, epoch, args):
    if args.lradj=='type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

def tslib_data_loader(window, length_size, batch_size, data, data_mark, samples_per_county, start_offset, shuffle=True, device=device):
    seq_len = window
    sequence_length = seq_len + length_size

    total_samples = len(data)
    total_counties = total_samples // samples_per_county

    result = []
    result_mark = []
    label_indices = []

    for c in range(total_counties):
        start_index = c * samples_per_county
        end_index = start_index + samples_per_county
        county_data = data[start_index:end_index]
        county_data_mark = data_mark[start_index:end_index]

        for i in range(len(county_data) - sequence_length + 1):
            result.append(county_data[i: i + sequence_length])
            result_mark.append(county_data_mark[i: i + sequence_length])
            global_label_index = start_offset + start_index + i + seq_len
            label_indices.append(global_label_index)

    result = np.array(result)
    result_mark = np.array(result_mark)
    label_indices = np.array(label_indices)

    x_temp = result[:, :-length_size]
    y_temp = result[:, -(length_size + int(window / 2)):]
    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    label_data = np.array(data)[:, -1]
    label_lstm = label_data[label_indices - start_offset] 

    x_temp = torch.tensor(x_temp).type(torch.float32).to(device)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32).to(device)
    y_temp = torch.tensor(y_temp).type(torch.float32).to(device)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32).to(device)
    label_lstm = torch.tensor(label_lstm).type(torch.float32).to(device).unsqueeze(-1)
    label_indices = torch.tensor(label_indices).to(device)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark, label_lstm, label_indices)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark, label_lstm, label_indices

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False        
        self.val_loss_min = np.Inf      
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        self.val_loss_min = val_loss

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
       
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
      
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        if data.shape[-1] != mean.shape[-1]:
            mean = mean[-1:]
            std = std[-1:]
        return (data * std) + mean
    
def tslib_data_loader_medium(window, length_size, batch_size, data, data_mark, samples_per_county, start_offset, shuffle=True, device=device):
    seq_len = window
    sequence_length = seq_len + length_size 

    total_samples = len(data)
    total_counties = total_samples // samples_per_county

    result = []
    result_mark = []
    label_indices = []

    for c in range(total_counties):
        start_index = c * samples_per_county
        end_index = start_index + samples_per_county
        county_data = data[start_index:end_index]
        county_data_mark = data_mark[start_index:end_index]

        for i in range(len(county_data) - sequence_length + 1):
            result.append(county_data[i: i + sequence_length])
            result_mark.append(county_data_mark[i: i + sequence_length])
            global_label_index = start_offset + start_index + i + seq_len
            label_indices.append(global_label_index)

        for i in range(len(county_data) - sequence_length + 1, len(county_data) - seq_len):
            temp_seq = county_data[i:].tolist()
            temp_mark_seq = county_data_mark[i:].tolist()

            pad_size = sequence_length - len(temp_seq)
            temp_seq.extend([temp_seq[-1]] * pad_size)  
            temp_mark_seq.extend([temp_mark_seq[-1]] * pad_size)

            result.append(np.array(temp_seq))
            result_mark.append(np.array(temp_mark_seq))
            global_label_index = start_offset + start_index + i + seq_len
            label_indices.append(global_label_index)

    result = np.array(result)
    result_mark = np.array(result_mark)
    label_indices = np.array(label_indices)

    x_temp = result[:, :-length_size]  
    y_temp = result[:, -(length_size + int(window / 2)):]  
    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    label_data = np.array(data)[:, -1]
    label_lstm = label_data[label_indices - start_offset]  

    x_temp = torch.tensor(x_temp).type(torch.float32).to(device)
    x_temp_mark = torch.tensor(x_temp_mark).type(torch.float32).to(device)
    y_temp = torch.tensor(y_temp).type(torch.float32).to(device)
    y_temp_mark = torch.tensor(y_temp_mark).type(torch.float32).to(device)
    label_lstm = torch.tensor(label_lstm).type(torch.float32).to(device).unsqueeze(-1)
    label_indices = torch.tensor(label_indices).to(device)

    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark, label_lstm, label_indices)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark, label_lstm, label_indices

def tslib_data_loader_long(window, length_size, batch_size, data, data_mark, samples_per_county, start_offset, shuffle=True, device_param=None):
    seq_len = window
    sequence_length = seq_len + length_size  

    total_samples = len(data)
    total_counties = total_samples // samples_per_county

    result = []
    result_mark = []
    label_indices = []

    for c in range(total_counties):
        start_index = c * samples_per_county
        end_index = start_index + samples_per_county
        county_data = data[start_index:end_index]
        county_data_mark = data_mark[start_index:end_index]

        for i in range(len(county_data) - sequence_length + 1):
            result.append(county_data[i: i + sequence_length])
            result_mark.append(county_data_mark[i: i + sequence_length])
            global_label_index = start_offset + start_index + i + seq_len
            label_indices.append(global_label_index)

        for i in range(len(county_data) - sequence_length + 1, len(county_data) - seq_len):
            temp_seq = county_data[i:].tolist()
            temp_mark_seq = county_data_mark[i:].tolist()

            pad_size = sequence_length - len(temp_seq)
            temp_seq.extend([temp_seq[-1]] * pad_size) 
            temp_mark_seq.extend([temp_mark_seq[-1]] * pad_size)

            result.append(np.array(temp_seq))
            result_mark.append(np.array(temp_mark_seq))
            global_label_index = start_offset + start_index + i + seq_len
            label_indices.append(global_label_index)

    result = np.array(result)
    result_mark = np.array(result_mark)
    label_indices = np.array(label_indices)

    x_temp = result[:, :-length_size]  
    y_temp = result[:, -(length_size + int(window / 2)):] 
    x_temp_mark = result_mark[:, :-length_size]
    y_temp_mark = result_mark[:, -(length_size + int(window / 2)):]

    label_data = np.array(data)[:, -1]
    label_lstm = label_data[label_indices - start_offset]

    x_temp = torch.tensor(x_temp, dtype=torch.float32)
    x_temp_mark = torch.tensor(x_temp_mark, dtype=torch.float32)
    y_temp = torch.tensor(y_temp, dtype=torch.float32)
    y_temp_mark = torch.tensor(y_temp_mark, dtype=torch.float32)
    label_lstm = torch.tensor(label_lstm, dtype=torch.float32).unsqueeze(-1)
    label_indices = torch.tensor(label_indices)
    
    ds = TensorDataset(x_temp, y_temp, x_temp_mark, y_temp_mark, label_lstm, label_indices)
    dataloader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    return dataloader, x_temp, y_temp, x_temp_mark, y_temp_mark, label_lstm, label_indices
