import os
import glob
import numpy as np
import pandas as pd
import torch
import gc

from captum.attr import IntegratedGradients
from sklearn.preprocessing import MinMaxScaler

from models.LSTM import LSTMMain
from models import STAR

from utils.timefeatures import time_features
from utils.data_process import *
from utils.tools import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
window = 90
pred_len = 30
batch_size = 16
samples_per_county = 3652

focus_day = 15
focus_idx = focus_day - 1

class Config:
    def __init__(self):
        self.seq_len              = window
        self.label_len            = window // 2
        self.pred_len             = pred_len
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
        self.task_name            = 'medium_term_forecast'

baseline_count = 25600    
test_count     = 2560    

out_root = './data/Captum/Medium'
os.makedirs(out_root, exist_ok=True)

np.random.seed(2025)
torch.manual_seed(2025)

for pq in glob.glob('./data/newparquet/*.parquet'):
    file_name = os.path.basename(pq).rsplit('.', 1)[0]
    print(f"\n=== 处理文件：{file_name} ===")

    df     = pd.read_parquet(pq)
    dates  = pd.to_datetime(df.iloc[:, 0])
    feats  = df.iloc[:, 1:-1].values            
    target = df.iloc[:, -1].values.reshape(-1, 1)
    stamp  = time_features(pd.DataFrame({'date': dates}), timeenc=1, freq='D')
    print(f"Loaded {file_name}: feats={feats.shape}, target={target.shape}, stamp={stamp.shape}")

    data_all    = np.hstack([feats, target])
    data_scaled = MinMaxScaler().fit_transform(data_all)
    print(f"Scaled data: {data_scaled.shape}")

    N         = data_scaled.shape[0]
    C         = N // samples_per_county
    train_end = int(C * 0.7)  * samples_per_county
    val_end   = int(C * 0.85) * samples_per_county

    train_data   = data_scaled[:train_end]
    val_data     = data_scaled[train_end:val_end]
    test_data    = data_scaled[val_end:]
    stamp_train  = stamp[:train_end]
    stamp_val    = stamp[train_end:val_end]
    stamp_test   = stamp[val_end:]
    print(f"Split: counties={C}, train={train_end}, val={val_end-train_end}, test={test_data.shape[0]}")

    train_loader, x_tr, _, x_tr_st, _, _, _ = tslib_data_loader_medium(
        window, pred_len, batch_size,
        train_data, stamp_train,
        samples_per_county=samples_per_county,
        shuffle=True, device=device, start_offset=0
    )
    _, x_va, _, x_va_st, _, _, _ = tslib_data_loader_medium(
        window, pred_len, batch_size,
        val_data, stamp_val,
        samples_per_county=samples_per_county,
        shuffle=False, device=device, start_offset=train_end
    )
    test_loader, x_te, _, x_te_st, _, _, _ = tslib_data_loader_medium(
        window, pred_len, batch_size,
        test_data, stamp_test,
        samples_per_county=samples_per_county,
        shuffle=False, device=device, start_offset=val_end
    )
    print(f"Tensors: x_tr={x_tr.shape}, x_te={x_te.shape}")

    feat_dim  = feats.shape[1]   
    stamp_dim = stamp.shape[1]  

    idx_base   = np.random.choice(x_tr.size(0), size=baseline_count, replace=False)
    base_feats = x_tr[idx_base]
    base_stamp = x_tr_st[idx_base]
    baseline   = torch.cat([base_feats, base_stamp], dim=-1).mean(dim=0, keepdim=True).to(device)
    print(f"Baseline shape: {baseline.shape}")

    idx_test    = np.random.choice(x_te.size(0), size=test_count, replace=False)
    test_feats  = x_te[idx_test]
    test_stamp  = x_te_st[idx_test]
    print(f"Test subset shape: {test_feats.shape}")

    lstm = LSTMMain(
        input_size = feat_dim + 1 + stamp_dim,   
        output_len = pred_len,
        lstm_hidden=128, lstm_layers=3,
        batch_size = batch_size,
        device     = device
    ).to(device)
    lstm.load_state_dict(torch.load(
        f'./medium_weights/{file_name}/{file_name}_best_LSTM_model.pth'
    ))
    print("Loaded LSTM weights")

    cfg = Config()
    cfg.dec_in, cfg.enc_in = feat_dim, feat_dim
    star = STAR.Model(cfg).to(device)
    star.load_state_dict(torch.load(
        f'./medium_weights/{file_name}/{file_name}_best_STAR_model.pth'
    ))
    star.eval()
    star.use_norm = False
    print("Loaded STAR weights & disabled normalization")

    feat_names = (
        list(df.columns[1:-1])     
        + [df.columns[-1]]          
        + [f"time_feature_{i+1}" for i in range(stamp_dim)]
    )

    def lstm_forward(x):
        f = x[..., :feat_dim+1]       
        s = x[..., feat_dim+1:]     
        out = lstm(f, s)              
        return out

    def star_forward(X):
        X_t = X.to(device)
        main = X_t[..., :feat_dim + 1]
        mark = X_t[..., feat_dim + 1:]
        star.eval()
        out = star(main, mark, None, None, None)
        out_last = out[..., -1:]
        out_star = out_last.squeeze(-1)
        return out_star

    def compute_imp(model, forward_fn, feats, stamps, name, train_mode=False):
        print(f"\nComputing {name} attributions…")
        ig = IntegratedGradients(forward_fn)
        if train_mode:
            model.train()
        all_attr = []
        batches = int(np.ceil(feats.size(0) / batch_size))
        for i in range(0, feats.size(0), batch_size):
            fb  = feats[i:i+batch_size].to(device)
            sb  = stamps[i:i+batch_size].to(device)
            inp = torch.cat([fb, sb], dim=-1).clone().detach().requires_grad_(True)

            attr = ig.attribute(inp, baselines=baseline, target=focus_idx, n_steps=50)
            summed = attr.sum(dim=1).detach().cpu().numpy()
            all_attr.append(summed)
            print(f"  {name} batch {i//batch_size+1}/{batches} done")
        if train_mode:
            model.eval()
        all_attr = np.vstack(all_attr)
        imp = all_attr.mean(axis=0)
        print(f"{name} attribution done")
        return imp

    lstm_imp = compute_imp(lstm, lstm_forward, test_feats, test_stamp, "LSTM", train_mode=True)
    star_imp = compute_imp(star, star_forward, test_feats, test_stamp, "STAR", train_mode=False)

    out_df  = pd.DataFrame({
        'feature': feat_names,
        'LSTM':    lstm_imp,
        'STAR':    star_imp
    })
    out_path = os.path.join(out_root, f'{file_name}.csv')
    out_df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")

    del train_loader, test_loader
    del df, feats, target, stamp
    del data_all, data_scaled
    del train_data, val_data, test_data
    del stamp_train, stamp_val, stamp_test
    del x_tr, x_va, x_te, x_tr_st, x_va_st, x_te_st
    del baseline, test_feats, test_stamp
    del lstm, star, out_df

    torch.cuda.empty_cache()
    gc.collect()
