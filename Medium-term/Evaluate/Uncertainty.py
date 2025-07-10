import pandas as pd
import numpy as np
import random
import time
import os
import glob
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from models import STAR
from models.LSTM import LSTMMain
from models.MLP import MLP
from utils.timefeatures import time_features
from utils.data_process import *
from utils.tools import *
import warnings
warnings.filterwarnings("ignore")

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
samples_per_county = 3652
noise_level = 0.05   
num_samples = 128   

uncertainty_root = r'F:\Article1\不确定性分析结果\Medium_uncertainty'
os.makedirs(uncertainty_root, exist_ok=True)

class Config:
    def __init__(self):
        self.seq_len = window
        self.label_len = int(window / 2)
        self.pred_len = length_size
        self.freq = 'D'
        self.batch_size = batch_size
        self.dec_in = None
        self.enc_in = None
        self.c_out = 1
        self.d_model = 128
        self.n_heads = 8
        self.dropout = 0.0
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 128
        self.factor = 5
        self.activation = 'gelu'
        self.channel_independence = 0
        self.top_k = 5
        self.num_kernels = 6
        self.distil = 1
        self.use_norm = True
        self.d_core = 512
        self.embed = 'timeF'
        self.output_attention = 0
        self.task_name = 'medium_term_forecast'

parquet_files = glob.glob("data/newparquet/*.parquet")
for parquet_file in parquet_files:
    file_start_time = time.time()
    file_name = os.path.basename(parquet_file).split('.')[0]
    print(f"\n=== Start processing file: {file_name} at {time.strftime('%Y-%m-%d %H:%M:%S')} ===")

    df = pd.read_parquet(parquet_file)
    dates = pd.to_datetime(df.iloc[:, 0])
    data_features = df.iloc[:, 1:-1]
    data_target = df.iloc[:, -1]
    df_stamp = pd.DataFrame({'date': dates})
    data_stamp = time_features(df_stamp, timeenc=1, freq='D')

    print(f"Data loaded: total samples {df.shape[0]}, feature dims {data_features.shape[1]}, timestamp dims {data_stamp.shape[1]}")

    data_all = pd.concat([data_features, data_target], axis=1).values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data_all)

    total_samples = data_scaled.shape[0]
    total_counties = total_samples // samples_per_county
    train_counties = int(total_counties * 0.5)
    val_counties = int(total_counties * 0.525)
    train_size = train_counties * samples_per_county
    val_size = val_counties * samples_per_county

    data_test = data_scaled[val_size:, :]
    stamp_test = data_stamp[val_size:, :]
    print(f"Test set: samples {data_test.shape[0]}, counties {(data_test.shape[0] // samples_per_county)}")

    test_loader, x_test, y_test, x_test_mark, y_test_mark, label_test_lstm, label_indices = \
        tslib_data_loader_medium(
            window, length_size,
            batch_size,
            data_test, stamp_test,
            samples_per_county=samples_per_county,
            shuffle=False,
            device=device,
            start_offset=0
        )
    print(f"DataLoader created: {len(test_loader)} batches, batch size {batch_size}")

    data_dim = data_features.shape[1] + 1
    lstm_model = LSTMMain(
        input_size=data_dim + data_stamp.shape[1],
        output_len=length_size,
        lstm_hidden=128,
        lstm_layers=3,
        batch_size=batch_size,
        device=device
    ).to(device)
    config = Config()
    config.dec_in = data_dim
    config.enc_in = data_dim
    net = STAR.Model(config).to(device)
    mlp_model = MLP(input_dim=length_size * 2, hidden_dim=64).to(device)

    weights_dir = os.path.join('./medium_weights', file_name)
    lstm_model.load_state_dict(torch.load(os.path.join(weights_dir, f'{file_name}_best_LSTM_model.pth')))
    net.load_state_dict(torch.load(os.path.join(weights_dir, f'{file_name}_best_STAR_model.pth')))
    mlp_model.load_state_dict(torch.load(os.path.join(weights_dir, f'{file_name}_best_MLP_model.pth')))

    lstm_model.eval()
    net.eval()
    mlp_model.eval()
    print("Models loaded and set to eval mode.")

    file_unc_dir = os.path.join(uncertainty_root, file_name)
    os.makedirs(file_unc_dir, exist_ok=True)

    for i in range(1, num_samples + 1):
        sample_seed = seed + i
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)

        sample_start = time.time()
        print(f"  --> Running sample {i}/{num_samples}...")

        all_preds = []
        all_labels = []
        all_indices = []

        with torch.no_grad():
            for feature_, y_batch, mark_, mark_y, label_, idx in test_loader:
                noise = (torch.rand_like(feature_[:, :, :]) - 0.5) * 2 * noise_level
                noise = noise.to(device)

                perturbed = feature_ + noise

                pred_lstm = lstm_model(perturbed, mark_)
                pred_star = net(perturbed, mark_, y_batch, mark_y, None)
                pred_star = pred_star.squeeze(1)
                pred_star = pred_star[:, :, -1]
                combined = torch.cat((pred_lstm, pred_star), dim=1)
                pred_mlp = mlp_model(combined)

                all_preds.append(pred_mlp.cpu().numpy())
                all_labels.append(label_.cpu().numpy())
                all_indices.append(idx.cpu().numpy())

        preds = np.vstack(all_preds)
        labels = np.vstack(all_labels)
        indices = np.hstack(all_indices)

        tmp_pred = np.zeros((preds.shape[0], data_all.shape[1]))
        tmp_label = np.zeros_like(tmp_pred)
        tmp_pred[:, -1] = preds[:, 0]
        tmp_label[:, -1] = labels[:, 0]

        pred_inv = scaler.inverse_transform(tmp_pred)[:, -1]
        label_inv = scaler.inverse_transform(tmp_label)[:, -1]
        county_idx = (indices // samples_per_county) + 1
        date_vals = dates.iloc[indices].values

        df_unc = pd.DataFrame({
            'date': date_vals,
            'county_index': county_idx,
            'true_value': label_inv,
            'pred_value': pred_inv
        })

        out_path = os.path.join(file_unc_dir, f'Prediction_{i}.csv')
        df_unc.to_csv(out_path, index=False)
        sample_end = time.time()
        print(f"    Saved: {out_path} (time: {sample_end - sample_start:.2f}s)")

    file_end_time = time.time()
    print(f"=== Finished processing {file_name}. Total time: {file_end_time - file_start_time:.2f}s ===")