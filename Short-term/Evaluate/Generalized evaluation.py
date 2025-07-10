import os
import glob
import time
import gc
import math
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from utils.timefeatures import time_features
from utils.data_process import *
from utils.tools import *
from utils.calculate_metrics import cal_eval
from models.LSTM import LSTMMain
from models import STAR
from models.MLP import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

window = 7               
length_size = 1           
batch_size = 1024
samples_per_county = 3652  

source_root = r"E:/VScode/Article/Com_Model/data/parquet"
test_root   = r"F:/Article/parquet_test"  
output_root = r"./data/other_county_short"

class Config:
    def __init__(self):
        self.seq_len              = window
        self.label_len            = window // 2
        self.pred_len             = length_size
        self.freq                 = 'D'
        self.batch_size           = batch_size
        self.dec_in               = None 
        self.enc_in               = None
        self.c_out                = 1
        self.d_model              = 128
        self.n_heads              = 8
        self.dropout              = 0.0
        self.e_layers             = 2
        self.d_layers             = 1
        self.d_ff                 = 128
        self.factor               = 5
        self.activation           = 'gelu'
        self.channel_independence = 0
        self.top_k                = 5
        self.num_kernels          = 6
        self.distil               = 1
        self.use_norm             = True
        self.d_core               = 512
        self.embed                = 'timeF'
        self.output_attention     = 0
        self.task_name            = 'short_term_forecast'

test_dirs = [
    d for d in glob.glob(os.path.join(test_root, "*"))
    if os.path.isdir(d)
]

for test_dir in test_dirs:
    folder_name = os.path.basename(test_dir)

    pre_root_sub = os.path.join(output_root, folder_name)
    os.makedirs(pre_root_sub, exist_ok=True)

    source_files = glob.glob(os.path.join(source_root, "*.parquet"))
    print(f"Found {len(source_files)} source files for fitting scaler.")

    for src_path in source_files:
        t0 = time.time()
        file_name = os.path.basename(src_path).rsplit('.', 1)[0]
        print(f"\n==== Processing {file_name} ====")

        df_src   = pd.read_parquet(src_path)
        feat_src = df_src.iloc[:, 1:-1].values
        tgt_src  = df_src.iloc[:, -1].to_numpy().reshape(-1, 1)
        scaler   = MinMaxScaler().fit(np.hstack([feat_src, tgt_src]))

        test_path = os.path.join(test_dir, f"{file_name}.parquet")
        if not os.path.exists(test_path):
            print(f"→ Warning: {test_path} not found, skipping.")
            continue

        df_test    = pd.read_parquet(test_path)
        date_test  = pd.to_datetime(df_test.iloc[:, 0])
        feat_test  = df_test.iloc[:, 1:-1].values
        tgt_test   = df_test.iloc[:, -1].to_numpy().reshape(-1, 1)
        stamp_test = time_features(pd.DataFrame({'date': date_test}), timeenc=1, freq='D')

        data_all_test    = np.hstack([feat_test, tgt_test])
        data_scaled_test = scaler.transform(data_all_test)

        test_loader, *_ = tslib_data_loader(
            window, length_size, batch_size,
            data_scaled_test, stamp_test,
            samples_per_county=samples_per_county,
            shuffle=False, device=device,
            start_offset=0
        )

        lstm_model = LSTMMain(
            input_size = data_scaled_test.shape[1] + stamp_test.shape[1],
            output_len = length_size,
            lstm_hidden=128,
            lstm_layers=3,
            batch_size =batch_size,
            device     =device
        ).to(device)

        cfg = Config()
        cfg.dec_in = data_scaled_test.shape[1]
        cfg.enc_in = data_scaled_test.shape[1]
        star_model = STAR.Model(cfg).to(device)

        mlp_model = MLP(input_dim=2, hidden_dim=64).to(device)

        wdir = os.path.join('./short_weights', file_name)
        lstm_model.load_state_dict(torch.load(os.path.join(wdir, f"{file_name}_best_LSTM_model.pth"), map_location=device))
        star_model.load_state_dict(torch.load(os.path.join(wdir, f"{file_name}_best_STAR_model.pth"), map_location=device))
        mlp_model.load_state_dict(torch.load(os.path.join(wdir, f"{file_name}_best_MLP_model.pth"), map_location=device))

        lstm_model.eval(); star_model.eval(); mlp_model.eval()

        preds, labels, dates, counties = [], [], [], []
        with torch.no_grad():
            for feat, y_in, feat_mark, y_mark, label, idx in test_loader:
                out_lstm = lstm_model(feat, feat_mark)
                tmp_star = star_model(feat, feat_mark, y_in, y_mark, None)
                out_star = tmp_star.squeeze(1)[:, -1:]
                fused   = torch.cat([out_lstm, out_star], dim=1)
                out_mlp = mlp_model(fused)

                preds   .append(out_mlp.cpu().numpy())
                labels  .append(label.cpu().numpy())
                dates   .append(date_test.iloc[idx.cpu().numpy()].values)
                counties.append((idx.cpu().numpy() // samples_per_county) + 1)

        preds    = np.vstack(preds)[:, 0]
        labels   = np.vstack(labels)[:, 0]
        dates    = np.concatenate(dates)
        counties = np.concatenate(counties)

        tmp = np.zeros((len(preds), data_scaled_test.shape[1]))
        tmp[:, -1] = preds
        pred_inv   = scaler.inverse_transform(tmp)[:, -1]
        tmp[:, -1] = labels
        label_inv  = scaler.inverse_transform(tmp)[:, -1]

        df_out = pd.DataFrame({
            'date':         dates,
            'county_index': counties,
            'true_value':   label_inv,
            'pred_value':   pred_inv
        })

        save_dir = os.path.join(pre_root_sub, file_name)
        os.makedirs(save_dir, exist_ok=True)

        max_cnty = df_out['county_index'].max()
        groups   = math.ceil(max_cnty / 100)

        for g in range(groups):
            start = g * 100 + 1
            end   = min((g + 1) * 100, max_cnty)
            sub_df = df_out[
                (df_out['county_index'] >= start) &
                (df_out['county_index'] <= end)
            ]

            out_path = os.path.join(
                save_dir,
                f"{file_name}_{start}_{end}.xlsx"
            )
            sub_df.to_excel(out_path, index=False)
            print(f"→ Saved {folder_name}/{file_name}_{start}_{end}.xlsx")

        metrics = cal_eval(label_inv, pred_inv)
        print(f"(elapsed {time.time()-t0:.1f}s) metrics={metrics}")

        del df_src, feat_src, tgt_src
        del df_test, feat_test, tgt_test, stamp_test
        del data_all_test, data_scaled_test
        del test_loader

        del lstm_model, star_model, mlp_model
        del preds, labels, dates, counties
        del tmp, pred_inv, label_inv, df_out

        try:
            del sub_df, groups, g
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()
