import os
import glob
import time
import gc

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
import pyarrow.parquet as pq

from models import STAR
from models.LSTM import LSTMMain
from models.MLP import MLP
from utils.timefeatures import time_features
from utils.data_process import *
from utils.tools import *
from utils.calculate_metrics import cal_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window = 90
length_size = 30    
batch_size = 4096
samples_per_county = 3652
chunk_counties = 100  

parquet_train_dir   = "./data/newparquet"    
parquet_test_root   = "F:/Article/parquet_test"     
weights_root        = "./medium_weights"   
output_root         = "./data/other_county_medium"  

class Config:
    def __init__(self):
        self.seq_len = window
        self.label_len = window // 2
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

def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()

def test_model_and_save_results():
    os.makedirs(output_root, exist_ok=True)

    for subdir in sorted(glob.glob(os.path.join(parquet_test_root, '*'))):
        subname = os.path.basename(subdir)
        folder_sub_out = os.path.join(output_root, subname)
        os.makedirs(folder_sub_out, exist_ok=True)

        for parquet_file in sorted(glob.glob(os.path.join(subdir, '*.parquet'))):
            fname = os.path.splitext(os.path.basename(parquet_file))[0]
            t0_file = time.perf_counter()

            train_file = os.path.join(parquet_train_dir, f"{fname}.parquet")
            df_train = pd.read_parquet(train_file)
            train_np = df_train.iloc[:, 1:].values 
            scaler = MinMaxScaler()
            scaler.fit(train_np)

            pf = pq.ParquetFile(parquet_file)
            batch_rows = samples_per_county * chunk_counties
            offset_counties = 0

            for batch_idx, rb in enumerate(pf.iter_batches(batch_size=batch_rows)):
                t0_chunk = time.perf_counter()
                df_chunk = rb.to_pandas()
                rows = df_chunk.shape[0]
                n_counties = rows // samples_per_county

                start_c = offset_counties + 1
                end_c   = offset_counties + n_counties
                offset_counties = end_c

                date_col = pd.to_datetime(df_chunk.iloc[:, 0])
                data_np = df_chunk.iloc[:, 1:].values
                data_scaled = scaler.transform(data_np)
                df_stamp = pd.DataFrame({'date': date_col})
                data_stamp = time_features(df_stamp, timeenc=1, freq='D')

                test_loader, _, _, _, _, _, idxs = tslib_data_loader(
                    window, length_size, batch_size,
                    data_scaled, data_stamp,
                    samples_per_county=samples_per_county,
                    shuffle=False, device=device, start_offset=0
                )

                lstm_model = LSTMMain(
                    input_size=data_scaled.shape[1] + data_stamp.shape[1],
                    output_len=length_size,
                    lstm_hidden=128, lstm_layers=3,
                    batch_size=batch_size, device=device
                )
                star_model = STAR.Model(Config())
                mlp_model = MLP(input_dim=length_size * 2, hidden_dim=64)

                wdir = os.path.join(weights_root, fname)
                load_model(lstm_model, os.path.join(wdir, f"{fname}_best_LSTM_model.pth"))
                load_model(star_model, os.path.join(wdir, f"{fname}_best_STAR_model.pth"))
                load_model(mlp_model, os.path.join(wdir, f"{fname}_best_MLP_model.pth"))

                all_preds, all_lbls = [], []
                with torch.no_grad():
                    for i, (feat, yb, feat_mk, yb_mk, lbl, _) in enumerate(test_loader):
                        p_lstm = lstm_model(feat, feat_mk)
                        p_star = star_model(feat, feat_mk, yb, yb_mk, None)
                        print(p_star.shape)
                        if p_star.dim() == 3:
                            p_star = p_star[:, :, -1]
                        comb = torch.cat((p_lstm, p_star), dim=1)
                        p_mlp = mlp_model(comb)
                        all_preds.append(p_mlp.cpu().numpy())
                        all_lbls.append(lbl.cpu().numpy())
                        if (i+1) % 10 == 0 or i == len(test_loader)-1:
                            print(f"{i+1}/{len(test_loader)}")

                preds = np.vstack(all_preds)
                lbls  = np.vstack(all_lbls)

                tmp_p = np.zeros((preds.shape[0], train_np.shape[1]))
                tmp_l = np.zeros_like(tmp_p)
                tmp_p[:, -1] = preds[:, 0]
                tmp_l[:, -1] = lbls[:, 0]
                inv_p = scaler.inverse_transform(tmp_p)[:, -1]
                inv_l = scaler.inverse_transform(tmp_l)[:, -1]

                idxs_np = idxs.detach().cpu().numpy()

                out_dir = os.path.join(folder_sub_out, fname)
                os.makedirs(out_dir, exist_ok=True)
                out_file = os.path.join(out_dir, f"{fname}_{start_c}_{end_c}.csv")
                df_out = pd.DataFrame({
                    'date': date_col.values[idxs_np],
                    'county_index': (idxs_np // samples_per_county) + start_c,
                    'true_value': inv_l,
                    'pred_value': inv_p
                })
                df_out.to_csv(out_file, index=False, encoding='utf-8-sig')

                del df_chunk, data_scaled, data_stamp, test_loader
                del all_preds, all_lbls, preds, lbls, tmp_p, tmp_l, inv_p, inv_l, df_out
                torch.cuda.empty_cache()
                gc.collect()

if __name__ == "__main__":
    test_model_and_save_results()
