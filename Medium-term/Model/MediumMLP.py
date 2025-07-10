import pandas as pd
import numpy as np
import random
import time
import os
import glob
import gc 
import torch
import torch.nn as nn
from utils.timefeatures import time_features
from utils.calculate_metrics import *
from utils.data_process import *
from utils.tools import *
import warnings
warnings.filterwarnings("ignore")

start_total = time.perf_counter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

window = 90        
length_size = 30     
batch_size = 1024
loss_function = "MSE"
if loss_function == 'MSE':
    loss_func = nn.MSELoss(reduction='mean')

learning_rate = 0.001
epochs = 999        

samples_per_county = 3652  

parquet_files = glob.glob(r"E:\VScode\Article\Com_Model\data\othermodel\*.parquet")

for parquet_file in parquet_files:
    file_start_time = time.perf_counter()

    file_name = os.path.basename(parquet_file).split('.')[0]

    df = pd.read_parquet(parquet_file)

    data_target = df.iloc[:, -1]
    data_features = df.iloc[:, 1:-1]
    date_col = df.iloc[:, 0]

    df_stamp = pd.DataFrame({'date': pd.to_datetime(date_col)})
    data_stamp = time_features(df_stamp, timeenc=1, freq='D')

    if data_features.shape[1] == 0:
        data_all = data_target.to_frame()
    else:
        data_all = pd.concat([data_features, data_target], axis=1)
    data_all_np = np.array(data_all)
    if data_all_np.ndim == 1:
        data_all_np = data_all_np.reshape(-1, 1)

    scaler = MinMaxScaler()
    data_inverse = scaler.fit_transform(data_all_np)

    total_samples = len(data_inverse)
    total_counties = total_samples // samples_per_county

    train_counties = 30
    val_counties = 6
    test_counties = total_counties - train_counties - val_counties

    train_size = train_counties * samples_per_county
    val_size = (train_counties + val_counties) * samples_per_county
    test_size = total_samples - val_size

    data_train = data_inverse[:train_size, :]
    data_val = data_inverse[train_size:val_size, :]
    data_test = data_inverse[val_size:, :]
    data_stamp_train = data_stamp[:train_size, :]
    data_stamp_val = data_stamp[train_size:val_size, :]
    data_stamp_test = data_stamp[val_size:, :]

    train_loader, x_train, y_train, x_train_mark, y_train_mark, label_train, label_indices_train = tslib_data_loader(
        window, length_size, batch_size, data_train, data_stamp_train, 
        samples_per_county=samples_per_county, shuffle=True, device=device, start_offset=0)

    val_loader, x_val, y_val, x_val_mark, y_val_mark, label_val, label_indices_val = tslib_data_loader(
        window, length_size, batch_size, data_val, data_stamp_val, 
        samples_per_county=samples_per_county, shuffle=False, device=device, start_offset=train_size)

    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim=30):
            super(SimpleMLP, self).__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, output_dim)
            self.relu = nn.ReLU()
        def forward(self, x, x_mark):
            combined = torch.cat((x, x_mark), dim=2)  
            flat = combined.view(combined.size(0), -1)  
            out = self.relu(self.fc1(flat))
            out = self.relu(self.fc2(out))
            out = self.fc3(out)
            return out
        
    sample_feature = x_train[:1]        
    sample_feature_mark = x_train_mark[:1]
    combined_dim = sample_feature.shape[2] + sample_feature_mark.shape[2]
    input_dim = window * combined_dim
    hidden_dim = 128
    mlp_model = SimpleMLP(input_dim, hidden_dim, output_dim=length_size).to(device)

    weights_dir = './MediumMLPWeights'
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)
    file_weights_dir = os.path.join(weights_dir, file_name)
    if not os.path.exists(file_weights_dir):
        os.makedirs(file_weights_dir)

    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    for epoch in range(epochs):
        mlp_model.train()
        train_loss_sum = 0
        for step, (feature_, y_train_batch, feature_mark_, y_train_mark_batch, label_, _) in enumerate(train_loader):
            optimizer.zero_grad()
            pred = mlp_model(feature_, feature_mark_)
            loss = loss_func(pred, label_)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()

        mlp_model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for step, (feature_, y_val_batch, feature_mark_, y_val_mark_batch, label_, _) in enumerate(val_loader):
                pred = mlp_model(feature_, feature_mark_)
                loss = loss_func(pred, label_)
                val_loss_sum += loss.item()

        if val_loss_sum < best_val_loss:
            best_val_loss = val_loss_sum
            torch.save(mlp_model.state_dict(), os.path.join(file_weights_dir, f'{file_name}_best_MLP_model.pth'))
        print(f"[{file_name}] Epoch {epoch+1}/{epochs} - MLP loss: {train_loss_sum:.8f}, val loss: {val_loss_sum:.8f}")

    mlp_model.load_state_dict(torch.load(os.path.join(file_weights_dir, f'{file_name}_best_MLP_model.pth')))
    mlp_model.eval()
    pred_records = [] 

    step = length_size  
    for i in range(test_counties):
        county_offset = val_size + i * samples_per_county  
        start_idx = i * samples_per_county
        end_idx = (i+1) * samples_per_county
        county_data = data_test[start_idx:end_idx, :]           
        county_stamp = data_stamp_test[start_idx:end_idx, :]     
        n = samples_per_county
        for start in range(0, n - window + 1, step):
            remaining = n - (start + window)
            if remaining <= 0:
                break
            L = step if remaining >= step else remaining
            x_seq = county_data[start : start+window, :]          
            x_mark_seq = county_stamp[start : start+window, :]       
            label_seq = county_data[start+window : start+window+L, -1] 
            
            x_tensor = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0).to(device)
            x_mark_tensor = torch.tensor(x_mark_seq, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_full = mlp_model(x_tensor, x_mark_tensor)  
            pred_full = pred_full.cpu().numpy().squeeze(0)    
            pred_seq = pred_full[:L]
            
            for j in range(L):
                global_idx = county_offset + start + window + j 
                date_val = df_stamp.iloc[global_idx]['date']
                county_index = (global_idx // samples_per_county) + 1
                pred_records.append({
                    'date': date_val,
                    'county_index': county_index,
                    'true_value_norm': label_seq[j],
                    'pred_value_norm': pred_seq[j]
                })
    df_result = pd.DataFrame(pred_records)

    n_rec = df_result.shape[0]
    temp_pred = np.zeros((n_rec, data_all_np.shape[1]))
    temp_label = np.zeros((n_rec, data_all_np.shape[1]))
    temp_pred[:, -1] = df_result['pred_value_norm'].values
    temp_label[:, -1] = df_result['true_value_norm'].values
    pred_inverse = scaler.inverse_transform(temp_pred)[:, -1]
    label_inverse = scaler.inverse_transform(temp_label)[:, -1]
    df_result['pred_value'] = pred_inverse
    df_result['true_value'] = label_inverse
    df_result.drop(['pred_value_norm', 'true_value_norm'], axis=1, inplace=True)

    pre_dir = './MediumPrediction'
    if not os.path.exists(pre_dir):
        os.makedirs(pre_dir)
    file_pre_dir = os.path.join(pre_dir, file_name)
    if not os.path.exists(file_pre_dir):
        os.makedirs(file_pre_dir)
    df_result.to_csv(os.path.join(file_pre_dir, f'{file_name}_test_predictions.csv'), index=False)

    # 释放内存
    del df, data_train, data_val, data_test, data_stamp_train, data_stamp_val, data_stamp_test
    del x_train, y_train, x_train_mark, y_train_mark, label_train, label_indices_train
    del x_val, y_val, x_val_mark, y_val_mark, label_val, label_indices_val
    del mlp_model, train_loader, val_loader
    del pred_records, df_result
    torch.cuda.empty_cache()
    gc.collect()
    
end_total = time.perf_counter()
