import pandas as pd
import numpy as np
import random
import time
import os
import glob
import gc
import torch
import logging
from sklearn.preprocessing import MinMaxScaler
from models import STAR
from models.LSTM import LSTMMain
from utils.timefeatures import time_features
from utils.tools import tslib_data_loader, tslib_data_loader_medium
import warnings

warnings.filterwarnings("ignore")

output_dir = r"E:\VScode\Article\Com_Model\data\medium_pdp_plots"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

window = 90
length_size = 30
batch_size = 1024
samples_per_county = 3652
test_counties = 4
test_size = samples_per_county * test_counties

logging.basicConfig(filename='pdp_run.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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


def predict_star(net, data_inverse, data_stamp, start, length, scaler):
    end = start + length
    data_test = data_inverse[start:end, :]
    stamp_test = data_stamp[start:end, :]
    loader, *_ = tslib_data_loader(
        window, length_size, batch_size,
        data_test, stamp_test,
        samples_per_county=samples_per_county,
        shuffle=False, device=device,
        start_offset=start
    )
    preds = []
    net.eval()
    with torch.no_grad():
        for feat, y_b, feat_m, y_m, _, _ in loader:
            out = net(feat, feat_m, y_b, y_m, None).squeeze(1)
            last = out[:, :, -1]
            last = last[:, -1]
            preds.append(last.cpu().numpy())
    preds = np.concatenate(preds)
    temp = np.zeros((len(preds), data_inverse.shape[1]))
    temp[:, -1] = preds
    inv = scaler.inverse_transform(temp)[:, -1]
    return inv


def predict_lstm(model, data_inverse, data_stamp, start, length, scaler):
    end = start + length
    data_test = data_inverse[start:end, :]
    stamp_test = data_stamp[start:end, :]
    loader, *_ = tslib_data_loader_medium(
        window, length_size, batch_size,
        data_test, stamp_test,
        samples_per_county=samples_per_county,
        shuffle=False, device=device,
        start_offset=start
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for feat, _, feat_m, _, _, _ in loader:
            out = model(feat, feat_m)
            last = out[:, -1]
            preds.append(last.cpu().numpy())
    preds = np.concatenate(preds)
    temp = np.zeros((len(preds), data_inverse.shape[1]))
    temp[:, -1] = preds
    inv = scaler.inverse_transform(temp)[:, -1]
    return inv


def main():
    parquet_files = glob.glob(r"data/newparquet/*.parquet")

    for pf in parquet_files:
        try:
            t0_file = time.perf_counter()
            name = os.path.basename(pf).split('.')[0]
            subdir = os.path.join(output_dir, name)
            os.makedirs(subdir, exist_ok=True)

            df_check = pd.read_parquet(pf)
            feats_check = df_check.columns[1:-1]
            all_done = all(os.path.exists(os.path.join(subdir, f"{name}_{col}_sensitivity.csv")) for col in feats_check)
            if all_done:
                logging.info(f"{name}")
                continue

            df = pd.read_parquet(pf)
            dates = pd.to_datetime(df.iloc[:, 0])
            feats = df.iloc[:, 1:-1]
            tgt = df.iloc[:, -1]
            stamp = time_features(pd.DataFrame({'date': dates}), timeenc=1, freq='D')

            arr = np.hstack([feats.values, tgt.values.reshape(-1, 1)])
            scaler = MinMaxScaler().fit(arr)
            inv = scaler.transform(arr)

            total = inv.shape[0]
            counties = total // samples_per_county
            val_c = int(counties * 0.75)
            val_off = val_c * samples_per_county

            cfg = Config()
            cfg.dec_in = feats.shape[1] + 1
            cfg.enc_in = cfg.dec_in
            star = STAR.Model(cfg).to(device)
            lstm = LSTMMain(input_size=feats.shape[1] + stamp.shape[1] + 1,
                            output_len=length_size,
                            lstm_hidden=128,
                            lstm_layers=3,
                            batch_size=batch_size,
                            device=device).to(device)
            star.load_state_dict(torch.load(f"./medium_weights/{name}/{name}_best_STAR_model.pth"))
            lstm.load_state_dict(torch.load(f"./medium_weights/{name}/{name}_best_LSTM_model.pth"))

            base_star = predict_star(star, inv, stamp, val_off, test_size, scaler)
            base_lstm = predict_lstm(lstm, inv, stamp, val_off, test_size, scaler)

            for i, col in enumerate(feats.columns):
                path = os.path.join(subdir, f"{name}_{col}_sensitivity.csv")
                if os.path.exists(path):
                    logging.info(f"[{col}]")
                    continue

                ratios = np.round(np.arange(0.5, 1.5001, 0.04), 2)
                rec = []
                t0_fg = time.perf_counter()
                for r in ratios:
                    t0 = time.perf_counter()
                    tmp = df.copy()
                    tmp.iloc[:, i+1] *= r
                    arr_tmp = np.hstack([tmp.iloc[:,1:-1].values, tmp.iloc[:,-1].values.reshape(-1,1)])
                    inv_tmp = scaler.transform(arr_tmp)
                    ps = predict_star(star, inv_tmp, stamp, val_off, test_size, scaler)
                    pl = predict_lstm(lstm, inv_tmp, stamp, val_off, test_size, scaler)
                    ds = ps - base_star
                    dl = pl - base_lstm
                    rec.append({
                        'ratio': r,
                        'star_mean_diff': ds.mean(),
                        'star_min_diff': ds.min(),
                        'star_max_diff': ds.max(),
                        'lstm_mean_diff': dl.mean(),
                        'lstm_min_diff': dl.min(),
                        'lstm_max_diff': dl.max(),
                        'time_secs': time.perf_counter() - t0
                    })
                df_out = pd.DataFrame(rec)
                df_out.to_csv(path, index=False)

            del df, inv, stamp, star, lstm
            torch.cuda.empty_cache()
            gc.collect()

        except Exception as e:
            logging.error(f"{e}", exc_info=True)
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(10)


if __name__ == "__main__":
    while True:
        try:
            main()
            break
        except Exception as e:
            logging.error(f"{e}", exc_info=True)
            torch.cuda.empty_cache()
            gc.collect()
            time.sleep(10)
